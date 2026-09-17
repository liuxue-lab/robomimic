#!/usr/bin/env python3
"""Serve the frozen Lift-PH Vision BC policy from the Stage 8 Jetson."""

import argparse
import hashlib
import json
import os
import random
import signal
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from stage08_protocol import (
    ConnectionClosed,
    ProtocolError,
    load_config,
    pack_message,
    recv_frame,
    send_frame,
    unpack_message,
)


EXPECTED_PARAMETER_COUNT = 23_661_963
DEFAULT_CHECKPOINT = Path(
    "/home/jetson/robomimic-hil/checkpoints/"
    "model_epoch_80_Lift_success_1.0.pth"
)
DEFAULT_CONFIG = Path(
    "/home/jetson/robomimic-hil/runtime/stage08/"
    "stage08-hil-protocol.json"
)
DEFAULT_LOG = Path(
    "/home/jetson/robomimic-hil/runs/"
    "stage08-policy-server.jsonl"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe_error(exc, byte_limit):
    text = f"{type(exc).__name__}: {exc}"
    encoded = text.encode("utf-8")[:byte_limit]
    return encoded.decode("utf-8", errors="ignore") or type(exc).__name__


def ordered_wall_time_ns(previous=0):
    return max(time.time_ns(), int(previous))


def set_reproducible_seed(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class JsonlLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a", encoding="utf-8", buffering=1)

    def write(self, event, **fields):
        record = {
            "event": event,
            "log_time_ns": time.time_ns(),
            **fields,
        }
        self.stream.write(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )

    def close(self):
        self.stream.close()


class SequenceTracker:
    """Enforce contiguous sequence IDs, resetting to zero for each new episode."""

    def __init__(self):
        self.episode_id = None
        self.next_sequence_id = 0

    def accept(self, episode_id, sequence_id):
        new_episode = episode_id != self.episode_id
        expected = 0 if new_episode else self.next_sequence_id
        if sequence_id != expected:
            raise ProtocolError(
                f"sequence mismatch for episode {episode_id!r}: "
                f"expected {expected}, got {sequence_id}"
            )
        if new_episode:
            self.episode_id = episode_id
        self.next_sequence_id = sequence_id + 1
        return new_episode


def fixed_raw_observation():
    pixels = np.arange(84 * 84 * 3, dtype=np.uint32).reshape(84, 84, 3)
    return {
        "agentview_image": ((pixels * 17 + 13) % 256).astype(np.uint8),
        "robot0_eye_in_hand_image": ((pixels * 29 + 7) % 256).astype(np.uint8),
        "robot0_eef_pos": np.array([0.05, -0.10, 0.90], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.02, -0.02], dtype=np.float32),
    }


def load_policy(checkpoint_path, config, warmup_steps, seed):
    """Verify and load the trusted checkpoint, then warm up CUDA."""
    import torch
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    checkpoint_path = Path(checkpoint_path)
    require(checkpoint_path.is_file(), f"checkpoint does not exist: {checkpoint_path}")
    checkpoint_sha = sha256_file(checkpoint_path)
    expected_sha = config["evaluation"]["checkpoint_sha256"]
    require(checkpoint_sha == expected_sha, f"checkpoint SHA256 mismatch: {checkpoint_sha}")
    require(torch.cuda.is_available(), "CUDA is unavailable")

    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")

    load_start = time.monotonic_ns()
    checkpoint_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    policy, restored_checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_dict=checkpoint_dict,
        device=device,
        verbose=False,
    )
    torch.cuda.synchronize()
    load_duration_ns = time.monotonic_ns() - load_start
    del checkpoint_dict, restored_checkpoint

    algorithm = policy.policy
    require(type(algorithm).__name__ == "BC_GMM", "checkpoint did not restore BC_GMM")
    parameter_count = sum(parameter.numel() for parameter in algorithm.nets.parameters())
    require(parameter_count == EXPECTED_PARAMETER_COUNT, f"parameter count mismatch: {parameter_count}")
    require(not algorithm.nets.training, "policy network is not in evaluation mode")
    require(
        {str(parameter.device) for parameter in algorithm.nets.parameters()} == {"cuda:0"},
        "policy parameters are not exclusively on cuda:0",
    )
    require(algorithm.ac_dim == 7, f"unexpected action dimension: {algorithm.ac_dim}")

    raw_obs = fixed_raw_observation()
    processed_obs = ObsUtils.process_obs_dict(raw_obs)
    for _ in range(warmup_steps):
        with torch.inference_mode():
            action = policy(ob=processed_obs)
        require(np.asarray(action).shape == (7,), "warm-up action shape mismatch")
    torch.cuda.synchronize()

    set_reproducible_seed(seed, torch)
    policy.start_episode()
    return {
        "torch": torch,
        "obs_utils": ObsUtils,
        "policy": policy,
        "checkpoint_sha256": checkpoint_sha,
        "load_duration_ns": load_duration_ns,
        "parameter_count": parameter_count,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def infer_action(runtime, raw_obs):
    """Preprocess one raw observation and run one synchronized CUDA action sample."""
    torch = runtime["torch"]
    preprocess_start = time.monotonic_ns()
    processed_obs = runtime["obs_utils"].process_obs_dict(raw_obs)
    preprocess_duration_ns = time.monotonic_ns() - preprocess_start

    inference_start_wall_ns = ordered_wall_time_ns()
    inference_start = time.monotonic_ns()
    with torch.inference_mode():
        raw_action = runtime["policy"](ob=processed_obs)
    torch.cuda.synchronize()
    inference_duration_ns = time.monotonic_ns() - inference_start
    inference_end_wall_ns = ordered_wall_time_ns(inference_start_wall_ns)

    postprocess_start = time.monotonic_ns()
    action = np.asarray(raw_action)
    require(action.dtype == np.dtype("float32"), f"policy action dtype is {action.dtype}, expected float32")
    require(action.shape == (7,), f"policy action shape is {action.shape}, expected (7,)")
    require(bool(np.isfinite(action).all()), "policy action contains nonfinite values")
    action = np.ascontiguousarray(action)
    postprocess_duration_ns = time.monotonic_ns() - postprocess_start

    return action, {
        "preprocess_duration_ns": preprocess_duration_ns,
        "inference_duration_ns": inference_duration_ns,
        "postprocess_duration_ns": postprocess_duration_ns,
        "inference_start_wall_ns": inference_start_wall_ns,
        "inference_end_wall_ns": inference_end_wall_ns,
    }


def build_response_metadata(
    episode_id,
    sequence_id,
    server_receive_ns,
    inference_start_ns,
    inference_end_ns,
    status,
    error_message,
):
    inference_start_ns = max(int(server_receive_ns), int(inference_start_ns))
    inference_end_ns = max(inference_start_ns, int(inference_end_ns))
    server_send_ns = ordered_wall_time_ns(inference_end_ns)
    return {
        "episode_id": episode_id,
        "sequence_id": sequence_id,
        "server_receive_ns": server_receive_ns,
        "inference_start_ns": inference_start_ns,
        "inference_end_ns": inference_end_ns,
        "server_send_ns": server_send_ns,
        "status": status,
        "error_message": error_message,
    }


def send_error_response(conn, config, metadata, server_receive_ns, exc):
    now = ordered_wall_time_ns(server_receive_ns)
    error_text = json_safe_error(exc, config["safety"]["max_error_message_utf8_bytes"])
    response_metadata = build_response_metadata(
        episode_id=metadata["episode_id"],
        sequence_id=metadata["sequence_id"],
        server_receive_ns=server_receive_ns,
        inference_start_ns=now,
        inference_end_ns=now,
        status="error",
        error_message=error_text,
    )
    payload = pack_message(
        config,
        config["response"]["message_type"],
        response_metadata,
        {"action": np.zeros((7,), dtype=np.float32)},
    )
    send_frame(conn, config, payload)
    return response_metadata, len(payload), error_text


def handle_connection(conn, peer, config, runtime, logger, request_budget, stop_event):
    """Serve a single verified client until disconnect, timeout, or request budget."""
    tracker = SequenceTracker()
    handled = 0
    conn.settimeout(config["transport"]["server_idle_timeout_seconds"])
    if config["transport"]["tcp_nodelay"]:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    if config["transport"]["keepalive"]:
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    while not stop_event.is_set():
        receive_start = time.monotonic_ns()
        try:
            payload = recv_frame(conn, config)
            server_receive_ns = ordered_wall_time_ns()
        except socket.timeout:
            logger.write("connection_idle_timeout", peer_ip=peer[0], peer_port=peer[1])
            break
        except ConnectionClosed:
            logger.write("client_disconnected", peer_ip=peer[0], peer_port=peer[1])
            break
        except (OSError, ProtocolError) as exc:
            logger.write("frame_receive_error", peer_ip=peer[0], peer_port=peer[1], error=json_safe_error(exc, 1024))
            break
        receive_duration_ns = time.monotonic_ns() - receive_start

        decode_start = time.monotonic_ns()
        try:
            _, metadata, arrays = unpack_message(
                config,
                payload,
                expected_message_type=config["request"]["message_type"],
            )
        except ProtocolError as exc:
            logger.write(
                "request_decode_error",
                peer_ip=peer[0],
                peer_port=peer[1],
                request_payload_bytes=len(payload),
                receive_duration_ns=receive_duration_ns,
                error=json_safe_error(exc, 1024),
            )
            break
        decode_duration_ns = time.monotonic_ns() - decode_start

        episode_id = metadata["episode_id"]
        sequence_id = metadata["sequence_id"]
        request_start = time.monotonic_ns()

        try:
            new_episode = tracker.accept(episode_id, sequence_id)
            if new_episode:
                runtime["policy"].start_episode()
            action, timing = infer_action(runtime, arrays)
            response_metadata = build_response_metadata(
                episode_id=episode_id,
                sequence_id=sequence_id,
                server_receive_ns=server_receive_ns,
                inference_start_ns=timing["inference_start_wall_ns"],
                inference_end_ns=timing["inference_end_wall_ns"],
                status="ok",
                error_message="",
            )
            encode_start = time.monotonic_ns()
            response_payload = pack_message(
                config,
                config["response"]["message_type"],
                response_metadata,
                {"action": action},
            )
            encode_duration_ns = time.monotonic_ns() - encode_start
            send_start = time.monotonic_ns()
            send_frame(conn, config, response_payload)
            send_duration_ns = time.monotonic_ns() - send_start
            handled += 1
            logger.write(
                "inference_response",
                peer_ip=peer[0],
                peer_port=peer[1],
                episode_id=episode_id,
                sequence_id=sequence_id,
                new_episode=new_episode,
                request_payload_bytes=len(payload),
                response_payload_bytes=len(response_payload),
                receive_duration_ns=receive_duration_ns,
                decode_duration_ns=decode_duration_ns,
                preprocess_duration_ns=timing["preprocess_duration_ns"],
                inference_duration_ns=timing["inference_duration_ns"],
                postprocess_duration_ns=timing["postprocess_duration_ns"],
                encode_duration_ns=encode_duration_ns,
                send_duration_ns=send_duration_ns,
                server_request_duration_ns=time.monotonic_ns() - request_start,
                status="ok",
            )
            print(
                f"InferenceResponse=episode:{episode_id};sequence:{sequence_id};"
                f"inference_ms:{timing['inference_duration_ns'] / 1e6:.3f}",
                flush=True,
            )
        except Exception as exc:
            try:
                response_metadata, response_bytes, error_text = send_error_response(
                    conn,
                    config,
                    metadata,
                    server_receive_ns,
                    exc,
                )
                handled += 1
                logger.write(
                    "inference_error_response",
                    peer_ip=peer[0],
                    peer_port=peer[1],
                    episode_id=episode_id,
                    sequence_id=sequence_id,
                    response_payload_bytes=response_bytes,
                    server_request_duration_ns=time.monotonic_ns() - request_start,
                    status="error",
                    error=error_text,
                )
                print(f"InferenceError=episode:{episode_id};sequence:{sequence_id};error:{error_text}", flush=True)
            except Exception as send_exc:
                logger.write(
                    "error_response_send_failure",
                    peer_ip=peer[0],
                    peer_port=peer[1],
                    episode_id=episode_id,
                    sequence_id=sequence_id,
                    error=json_safe_error(send_exc, 1024),
                )
            break

        if request_budget > 0 and handled >= request_budget:
            break

    return handled


def run_server(args, config, runtime):
    bind_host = config["transport"]["server_host"]
    port = config["transport"]["server_port"]
    expected_client = config["transport"]["expected_client_host"]
    logger = JsonlLogger(args.log_path)
    stop_event = threading.Event()

    def request_stop(signum, _frame):
        logger.write("signal", signal=signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    code_sha = sha256_file(Path(__file__))
    logger.write(
        "server_start",
        pid=os.getpid(),
        bind_host=bind_host,
        port=port,
        expected_client_host=expected_client,
        checkpoint_sha256=runtime["checkpoint_sha256"],
        server_code_sha256=code_sha,
        parameter_count=runtime["parameter_count"],
        gpu=runtime["gpu"],
        torch_version=runtime["torch_version"],
        torch_cuda=runtime["torch_cuda"],
        seed=args.seed,
        warmup_steps=args.warmup_steps,
    )

    accepted_connections = 0
    handled_requests = 0
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((bind_host, port))
        server.listen(config["safety"]["max_connections"])
        server.settimeout(1.0)
        print(f"ServerReady={bind_host}:{port}", flush=True)
        print(f"ServerPID={os.getpid()}", flush=True)
        print(f"ServerLog={args.log_path}", flush=True)

        while not stop_event.is_set():
            if args.max_connections > 0 and accepted_connections >= args.max_connections:
                break
            try:
                conn, peer = server.accept()
            except socket.timeout:
                continue
            except OSError as exc:
                if stop_event.is_set():
                    break
                raise RuntimeError(f"accept failed: {exc}") from exc

            accepted_connections += 1
            logger.write("connection_accepted", peer_ip=peer[0], peer_port=peer[1])
            print(f"ConnectionAccepted={peer[0]}:{peer[1]}", flush=True)
            with conn:
                if peer[0] != expected_client:
                    logger.write(
                        "connection_rejected",
                        peer_ip=peer[0],
                        peer_port=peer[1],
                        reason="unexpected_client_ip",
                    )
                    print(f"ConnectionRejected={peer[0]}", flush=True)
                    continue
                remaining = 0
                if args.max_requests > 0:
                    remaining = max(args.max_requests - handled_requests, 0)
                    if remaining == 0:
                        break
                handled_requests += handle_connection(
                    conn,
                    peer,
                    config,
                    runtime,
                    logger,
                    remaining,
                    stop_event,
                )
            if args.max_requests > 0 and handled_requests >= args.max_requests:
                break
    finally:
        server.close()
        logger.write(
            "server_stop",
            accepted_connections=accepted_connections,
            handled_requests=handled_requests,
        )
        logger.close()
    print(f"ServerStopped=connections:{accepted_connections};requests:{handled_requests}", flush=True)


def run_logic_self_test(config):
    tracker = SequenceTracker()
    require(tracker.accept("episode-a", 0) is True, "first request was not a new episode")
    require(tracker.accept("episode-a", 1) is False, "continued request was treated as new")
    try:
        tracker.accept("episode-a", 1)
    except ProtocolError:
        print("DuplicateSequenceRejection=PASS")
    else:
        raise RuntimeError("duplicate sequence was accepted")
    try:
        tracker.accept("episode-a", 3)
    except ProtocolError:
        print("OutOfOrderSequenceRejection=PASS")
    else:
        raise RuntimeError("out-of-order sequence was accepted")
    require(tracker.accept("episode-b", 0) is True, "new episode did not reset sequence")
    try:
        tracker.accept("episode-c", 1)
    except ProtocolError:
        print("NewEpisodeMustStartAtZero=PASS")
    else:
        raise RuntimeError("new episode with nonzero sequence was accepted")

    metadata = build_response_metadata(
        episode_id="self-test",
        sequence_id=0,
        server_receive_ns=100,
        inference_start_ns=90,
        inference_end_ns=80,
        status="error",
        error_message="self-test",
    )
    require(metadata["server_receive_ns"] == 100, "server receive timestamp changed")
    require(metadata["inference_start_ns"] == 100, "inference start was not clamped")
    require(metadata["inference_end_ns"] == 100, "inference end was not clamped")
    require(metadata["server_send_ns"] >= 100, "server send timestamp is out of order")
    payload = pack_message(
        config,
        config["response"]["message_type"],
        metadata,
        {"action": np.zeros((7,), dtype=np.float32)},
    )
    unpack_message(config, payload, expected_message_type=config["response"]["message_type"])
    print("OrderedErrorResponse=PASS")
    print("Stage08ServerLogicSelfTest=PASS")


def run_model_smoke_test(args, config):
    runtime = load_policy(args.checkpoint, config, args.warmup_steps, args.seed)
    set_reproducible_seed(args.seed, runtime["torch"])
    runtime["policy"].start_episode()
    action, timing = infer_action(runtime, fixed_raw_observation())
    limit = float(config["safety"]["action_absolute_limit"])
    require(bool((np.abs(action) <= limit).all()), f"smoke-test action exceeds {limit}")
    print("CheckpointSHA256=", runtime["checkpoint_sha256"])
    print("AlgorithmClass=", type(runtime["policy"].policy).__name__)
    print("ParameterCount=", runtime["parameter_count"])
    print("GPU=", runtime["gpu"])
    print("TorchVersion=", runtime["torch_version"])
    print("TorchCUDA=", runtime["torch_cuda"])
    print("ModelLoadMs=", round(runtime["load_duration_ns"] / 1e6, 3))
    print("PreprocessMs=", round(timing["preprocess_duration_ns"] / 1e6, 3))
    print("InferenceMs=", round(timing["inference_duration_ns"] / 1e6, 3))
    print("PostprocessMs=", round(timing["postprocess_duration_ns"] / 1e6, 3))
    print(
        "Action=",
        np.array2string(action, precision=9, separator=", ", floatmode="fixed"),
    )
    print("ActionShape=", action.shape)
    print("ActionDType=", action.dtype)
    print("ActionFinite=", bool(np.isfinite(action).all()))
    print("ActionWithinLimit=", bool((np.abs(action) <= limit).all()))
    print("Stage08JetsonServerModelSmokeTest=PASS")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-connections", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--max-requests", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-smoke-test", action="store_true")
    args = parser.parse_args()
    if args.self_test and args.model_smoke_test:
        parser.error("choose only one of --self-test and --model-smoke-test")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be nonnegative")
    if args.max_connections < 0 or args.max_requests < 0:
        parser.error("request and connection limits must be nonnegative")
    return args


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.seed is None:
        args.seed = config["evaluation"]["evaluation_seed"]
    require(args.seed >= 0, "seed must be nonnegative")

    if args.self_test:
        run_logic_self_test(config)
        return
    if args.model_smoke_test:
        run_model_smoke_test(args, config)
        return

    runtime = load_policy(args.checkpoint, config, args.warmup_steps, args.seed)
    print("CheckpointSHA256=", runtime["checkpoint_sha256"], flush=True)
    print("ModelLoadMs=", round(runtime["load_duration_ns"] / 1e6, 3), flush=True)
    print("AlgorithmClass=", type(runtime["policy"].policy).__name__, flush=True)
    print("ParameterCount=", runtime["parameter_count"], flush=True)
    print("GPU=", runtime["gpu"], flush=True)
    run_server(args, config, runtime)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Stage08JetsonPolicyServer=FAIL; Error={type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1)
