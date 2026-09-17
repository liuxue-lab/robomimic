#!/usr/bin/env python3
"""Laptop-side client for the Stage 08 robomimic Lift PH HIL protocol.

Version 3 adds a paced 20 Hz fixed-input latency benchmark in addition to
network-driven MuJoCo rollouts and evidence artifact capture.

This module deliberately separates two observation representations:

* robomimic policy observations: RGB images are CHW float32 in [0, 1]
* Stage 08 wire observations: RGB images are HWC uint8 and state is float32

The conversion is performed with robomimic's ``unprocess_obs_dict`` before a
request is encoded.  The fixed-input probe uses the same deterministic raw
observation that was used for the cross-device checkpoint comparison.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import random
import socket
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

import stage08_protocol as Protocol


SCRIPT_VERSION = 3
# Canonical client-side hash over name, dtype, shape, and bytes in protocol order.
FIXED_WIRE_ARRAYS_SHA256 = "8933a821706e29bf276a9fd00e530e6654639cfc8f542d28de9a000a114d9523"
EXPECTED_FIXED_ACTION = np.asarray(
    [
        0.719693065,
        -0.455692232,
        -0.387767941,
        -0.057090916,
        -0.007736541,
        0.276231438,
        -0.999905407,
    ],
    dtype=np.float32,
)


class Stage08ClientError(RuntimeError):
    """Base class for client-side Stage 08 failures."""


class RemoteInferenceError(Stage08ClientError):
    """The server returned a structurally valid error response."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage08ClientError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_arrays(arrays: Mapping[str, np.ndarray], names: Tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in names:
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def fixed_raw_observation() -> Dict[str, np.ndarray]:
    """Return the deterministic raw observation used by Stage 08 validation."""

    pixels = np.arange(84 * 84 * 3, dtype=np.uint32).reshape(84, 84, 3)

    return {
        "agentview_image": ((pixels * 17 + 13) % 256).astype(np.uint8),
        "robot0_eye_in_hand_image": ((pixels * 29 + 7) % 256).astype(np.uint8),
        "robot0_eef_pos": np.asarray([0.05, -0.10, 0.90], dtype=np.float32),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([0.02, -0.02], dtype=np.float32),
    }


def processed_observation_to_wire(processed_obs: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Convert a robomimic policy observation to the Stage 08 wire format.

    ``EnvRobosuite`` returns processed RGB observations (CHW float32).  This
    function uses robomimic's canonical inverse transform to recover the raw
    HWC uint8 images expected by the network protocol.  Low-dimensional values
    are copied and explicitly converted to little-endian float32.
    """

    required = (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    missing = [name for name in required if name not in processed_obs]
    require(not missing, f"processed observation is missing keys: {missing}")

    try:
        import robomimic.utils.obs_utils as ObsUtils
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise Stage08ClientError(f"could not import robomimic ObsUtils: {exc}") from exc

    selected = {name: np.asarray(processed_obs[name]) for name in required}
    unprocessed = ObsUtils.unprocess_obs_dict(selected)

    wire: Dict[str, np.ndarray] = {}
    for name in ("agentview_image", "robot0_eye_in_hand_image"):
        image = np.asarray(unprocessed[name])
        require(image.shape == (84, 84, 3), f"{name} has wire shape {image.shape}, expected (84, 84, 3)")
        require(image.dtype == np.uint8, f"{name} has wire dtype {image.dtype}, expected uint8")
        wire[name] = np.ascontiguousarray(image)

    for name, shape in (
        ("robot0_eef_pos", (3,)),
        ("robot0_eef_quat", (4,)),
        ("robot0_gripper_qpos", (2,)),
    ):
        value = np.asarray(unprocessed[name], dtype="<f4")
        require(value.shape == shape, f"{name} has wire shape {value.shape}, expected {shape}")
        require(np.all(np.isfinite(value)), f"{name} contains a non-finite value")
        wire[name] = np.ascontiguousarray(value)

    return wire


class RemotePolicyClient:
    """Persistent, ordered Stage 08 TCP policy client."""

    def __init__(
        self,
        config_path: Path,
        *,
        timeout_mode: str = "functional",
        bind_expected_client: bool = True,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.config = Protocol.load_config(self.config_path)
        self.timeout_mode = timeout_mode
        self.bind_expected_client = bind_expected_client
        self.sock: Optional[socket.socket] = None
        self.episode_id: Optional[str] = None
        self.next_sequence = 0
        self.connection_index = 0
        self.last_timing: Optional[Dict[str, Any]] = None

        require(timeout_mode in ("functional", "realtime"), f"unknown timeout mode: {timeout_mode}")

    @property
    def endpoint(self) -> Tuple[str, int]:
        transport = self.config["transport"]
        return str(transport["server_host"]), int(transport["server_port"])

    def _timeout_seconds(self) -> float:
        if self.timeout_mode == "realtime":
            return float(self.config["transport"]["realtime_response_timeout_seconds"])
        return float(self.config["transport"]["functional_response_timeout_seconds"])

    def connect(self) -> None:
        if self.sock is not None:
            return

        host, port = self.endpoint
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            sock.settimeout(float(self.config["transport"]["connect_timeout_seconds"]))
            if self.bind_expected_client:
                expected_host = str(self.config["transport"]["expected_client_host"])
                sock.bind((expected_host, 0))
            sock.connect((host, port))
            sock.settimeout(self._timeout_seconds())
        except Exception:
            sock.close()
            raise

        self.sock = sock
        self.connection_index += 1

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()
            self.sock = None

    def __enter__(self) -> "RemotePolicyClient":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def start_episode(self, episode_id: Optional[str] = None) -> str:
        if episode_id is None:
            episode_id = f"lift-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        require(isinstance(episode_id, str) and episode_id, "episode_id must be a non-empty string")
        self.episode_id = episode_id
        self.next_sequence = 0
        return episode_id

    def request_raw(self, raw_obs: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.sock is None:
            self.connect()
        require(self.sock is not None, "socket is not connected")
        if self.episode_id is None:
            self.start_episode()
        require(self.episode_id is not None, "episode was not initialized")

        sequence = self.next_sequence
        client_send_ns = time.time_ns()
        request_metadata = {
            "episode_id": self.episode_id,
            "sequence_id": sequence,
            "client_send_ns": client_send_ns,
        }

        serialize_start = time.perf_counter_ns()
        request_payload = Protocol.pack_message(
            self.config,
            self.config["request"]["message_type"],
            request_metadata,
            raw_obs,
        )
        serialize_end = time.perf_counter_ns()

        round_trip_start = time.perf_counter_ns()
        Protocol.send_frame(self.sock, self.config, request_payload)
        send_complete = time.perf_counter_ns()
        response_payload = Protocol.recv_frame(self.sock, self.config)
        response_arrival_wall_ns = time.time_ns()
        response_arrival_mono_ns = time.perf_counter_ns()

        _, response_metadata, response_arrays = Protocol.unpack_message(
            self.config,
            response_payload,
            expected_message_type=self.config["response"]["message_type"],
        )
        decode_end = time.perf_counter_ns()

        require(response_metadata["episode_id"] == self.episode_id, "response episode_id does not match request")
        require(int(response_metadata["sequence_id"]) == sequence, "response sequence does not match request")

        timing: Dict[str, Any] = {
            "episode_id": self.episode_id,
            "sequence": sequence,
            "connection_index": self.connection_index,
            "client_send_ns": client_send_ns,
            "client_receive_ns": response_arrival_wall_ns,
            "serialize_ns": serialize_end - serialize_start,
            "send_call_ns": send_complete - round_trip_start,
            "receive_wait_ns": response_arrival_mono_ns - send_complete,
            "decode_ns": decode_end - response_arrival_mono_ns,
            "round_trip_ns": response_arrival_mono_ns - round_trip_start,
            "server_receive_ns": int(response_metadata["server_receive_ns"]),
            "server_inference_start_ns": int(response_metadata["inference_start_ns"]),
            "server_inference_end_ns": int(response_metadata["inference_end_ns"]),
            "server_send_ns": int(response_metadata["server_send_ns"]),
        }
        timing["estimated_upload_and_queue_ns"] = timing["server_receive_ns"] - client_send_ns
        timing["server_pre_inference_ns"] = timing["server_inference_start_ns"] - timing["server_receive_ns"]
        timing["server_inference_ns"] = timing["server_inference_end_ns"] - timing["server_inference_start_ns"]
        timing["server_postprocess_and_encode_ns"] = timing["server_send_ns"] - timing["server_inference_end_ns"]
        timing["estimated_download_ns"] = response_arrival_wall_ns - timing["server_send_ns"]

        status = str(response_metadata["status"])
        timing["status"] = status
        timing["error"] = str(response_metadata.get("error_message", ""))
        self.last_timing = timing

        if status != "ok":
            self.close()
            raise RemoteInferenceError(
                f"server rejected episode={self.episode_id!r} sequence={sequence}: {timing['error']}"
            )

        action = np.asarray(response_arrays["action"], dtype=np.float32)
        require(action.shape == (7,), f"response action has shape {action.shape}, expected (7,)")
        require(np.all(np.isfinite(action)), "response action contains a non-finite value")
        self.next_sequence += 1
        return action, timing

    def __call__(self, ob: Mapping[str, Any], goal: Optional[Mapping[str, Any]] = None) -> np.ndarray:
        require(goal is None, "Stage 08 Lift policy does not accept goal observations")
        wire_obs = processed_observation_to_wire(ob)
        action, _ = self.request_raw(wire_obs)
        return action


def write_json_exclusive(path: Path, data: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def temporary_artifact_path(path: Path) -> Path:
    """Return a sibling in-progress path while preserving the real suffix."""

    return path.with_name(f"{path.stem}.inprogress{path.suffix}")


def summarize_nanoseconds(values: list[int]) -> Dict[str, Any]:
    """Return stable millisecond summary fields for nanosecond durations."""

    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
        }
    array_ms = np.asarray(values, dtype=np.float64) / 1e6
    return {
        "count": int(array_ms.size),
        "mean_ms": float(np.mean(array_ms)),
        "p50_ms": float(np.percentile(array_ms, 50)),
        "p95_ms": float(np.percentile(array_ms, 95)),
        "p99_ms": float(np.percentile(array_ms, 99)),
        "max_ms": float(np.max(array_ms)),
    }


def write_jsonl_record(stream: Any, record: Mapping[str, Any]) -> None:
    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    stream.flush()


def client_self_test(config_path: Path) -> None:
    config = Protocol.load_config(config_path)
    raw_obs = fixed_raw_observation()
    input_names = tuple(entry["name"] for entry in config["request"]["arrays"])
    raw_hash = sha256_arrays(raw_obs, input_names)
    require(raw_hash == FIXED_WIRE_ARRAYS_SHA256, f"fixed wire arrays hash mismatch: {raw_hash}")

    for name in ("agentview_image", "robot0_eye_in_hand_image"):
        require(raw_obs[name].shape == (84, 84, 3), f"{name} shape mismatch")
        require(raw_obs[name].dtype == np.uint8, f"{name} dtype mismatch")
    for name in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
        require(raw_obs[name].dtype == np.float32, f"{name} dtype mismatch")

    client = RemotePolicyClient(config_path)
    first_episode = client.start_episode("self-test-episode-1")
    require(first_episode == "self-test-episode-1", "explicit episode ID was not retained")
    require(client.next_sequence == 0, "new episode did not reset sequence to zero")
    client.next_sequence = 19
    second_episode = client.start_episode("self-test-episode-2")
    require(second_episode == "self-test-episode-2", "second episode ID was not retained")
    require(client.next_sequence == 0, "second episode did not reset sequence to zero")

    left, right = socket.socketpair()
    left.settimeout(1.0)
    right.settimeout(1.0)
    client.sock = left
    client.connection_index = 1
    client.start_episode("self-test-socket-episode")
    server_failures = []
    received = {}

    def serve_one() -> None:
        try:
            payload = Protocol.recv_frame(right, config)
            _, metadata, arrays = Protocol.unpack_message(
                config,
                payload,
                expected_message_type=config["request"]["message_type"],
            )
            received["metadata"] = metadata
            received["arrays"] = arrays
            server_receive_ns = max(time.time_ns(), int(metadata["client_send_ns"]))
            inference_start_ns = max(time.time_ns(), server_receive_ns)
            inference_end_ns = max(time.time_ns(), inference_start_ns)
            server_send_ns = max(time.time_ns(), inference_end_ns)
            response_metadata = {
                "episode_id": metadata["episode_id"],
                "sequence_id": metadata["sequence_id"],
                "server_receive_ns": server_receive_ns,
                "inference_start_ns": inference_start_ns,
                "inference_end_ns": inference_end_ns,
                "server_send_ns": server_send_ns,
                "status": "ok",
                "error_message": "",
            }
            response_payload = Protocol.pack_message(
                config,
                config["response"]["message_type"],
                response_metadata,
                {"action": EXPECTED_FIXED_ACTION.copy()},
            )
            Protocol.send_frame(right, config, response_payload)
        except Exception as exc:  # pragma: no cover - reported in main test thread
            server_failures.append(exc)
        finally:
            right.close()

    server_thread = threading.Thread(target=serve_one, name="stage08-client-self-test", daemon=True)
    server_thread.start()
    try:
        action, timing = client.request_raw(raw_obs)
    finally:
        client.close()
    server_thread.join(timeout=1.0)
    require(not server_thread.is_alive(), "socket self-test server did not terminate")
    require(not server_failures, f"socket self-test server failed: {server_failures}")
    require(received["metadata"]["episode_id"] == "self-test-socket-episode", "request episode changed")
    require(received["metadata"]["sequence_id"] == 0, "request sequence did not start at zero")
    require(
        all(np.array_equal(received["arrays"][name], raw_obs[name]) for name in input_names),
        "request arrays changed during socket round trip",
    )
    require(np.array_equal(action, EXPECTED_FIXED_ACTION), "response action changed during socket round trip")
    require(timing["status"] == "ok", "socket response status changed")
    require(client.next_sequence == 1, "successful response did not advance sequence")

    print(f"FixedWireArraysSHA256= {raw_hash}")
    print("RawImageWireFormat= HWC uint8")
    print("LowDimWireFormat= float32")
    print("EpisodeSequenceReset=PASS")
    print("SocketRequestResponseRoundTrip=PASS")
    print("ResponseEpisodeAndSequenceMatch=PASS")
    print("Stage08LiftSimClientSelfTest=PASS")


def run_fixed_input_probe(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    config = Protocol.load_config(config_path)
    raw_obs = fixed_raw_observation()
    input_names = tuple(entry["name"] for entry in config["request"]["arrays"])
    raw_hash = sha256_arrays(raw_obs, input_names)
    require(raw_hash == FIXED_WIRE_ARRAYS_SHA256, f"fixed wire arrays hash mismatch: {raw_hash}")

    report_path = Path(args.report_path).resolve()
    require(not report_path.exists(), f"refusing to overwrite existing report: {report_path}")

    client = RemotePolicyClient(
        config_path,
        timeout_mode=args.timeout_mode,
        bind_expected_client=not args.no_bind_expected_client,
    )
    client.start_episode(args.episode_id)

    probe_start_ns = time.time_ns()
    try:
        action, timing = client.request_raw(raw_obs)
    finally:
        client.close()
    probe_end_ns = time.time_ns()

    action_limit = float(config["safety"]["action_absolute_limit"])
    max_abs_diff = float(np.max(np.abs(action - EXPECTED_FIXED_ACTION)))
    tolerance = float(args.expected_action_tolerance)
    action_finite = bool(np.all(np.isfinite(action)))
    action_within_limit = bool(np.max(np.abs(action)) <= action_limit)
    expected_action_match = bool(max_abs_diff <= tolerance)
    round_trip_50ms = bool(int(timing["round_trip_ns"]) <= int(config["evaluation"]["control_period_ns"]))

    require(action_finite, "network probe action is not finite")
    require(action_within_limit, "network probe action exceeds configured safety limit")
    require(expected_action_match, f"network action differs from fixed-input reference by {max_abs_diff}")

    report: Dict[str, Any] = {
        "artifact": "stage08_fixed_input_network_probe",
        "artifact_version": 1,
        "script_version": SCRIPT_VERSION,
        "created_unix_ns": probe_end_ns,
        "probe_start_unix_ns": probe_start_ns,
        "probe_end_unix_ns": probe_end_ns,
        "protocol_config": str(config_path),
        "protocol_config_sha256": sha256_file(config_path),
        "client_script": str(Path(__file__).resolve()),
        "client_script_sha256": sha256_file(Path(__file__).resolve()),
        "endpoint": {"host": client.endpoint[0], "port": client.endpoint[1]},
        "timeout_mode": args.timeout_mode,
        "episode_id": args.episode_id,
        "sequence": 0,
        "fixed_wire_arrays_sha256": raw_hash,
        "action": [float(value) for value in action],
        "expected_action": [float(value) for value in EXPECTED_FIXED_ACTION],
        "expected_action_tolerance": tolerance,
        "action_max_abs_diff": max_abs_diff,
        "action_finite": action_finite,
        "action_within_limit": action_within_limit,
        "expected_action_match": expected_action_match,
        "round_trip_50ms_budget": round_trip_50ms,
        "timing": timing,
        "status": "PASS",
    }
    write_json_exclusive(report_path, report)

    print("Stage08FixedInputNetworkProbeBegin=")
    print(f"Endpoint= {client.endpoint[0]}:{client.endpoint[1]}")
    print(f"EpisodeID= {args.episode_id}")
    print("Sequence= 0")
    print(f"RawInputSHA256= {raw_hash}")
    print("Action=", np.array2string(action, precision=9, separator=", "))
    print(f"ExpectedActionMaxAbsDiff= {max_abs_diff:.12g}")
    print(f"ExpectedActionTolerance= {tolerance:.12g}")
    print(f"ExpectedActionMatch= {expected_action_match}")
    print(f"RoundTripMs= {int(timing['round_trip_ns']) / 1e6:.3f}")
    print(f"ServerPreInferenceMs= {int(timing['server_pre_inference_ns']) / 1e6:.3f}")
    print(f"ServerInferenceMs= {int(timing['server_inference_ns']) / 1e6:.3f}")
    print(f"ServerPostprocessAndEncodeMs= {int(timing['server_postprocess_and_encode_ns']) / 1e6:.3f}")
    print(f"NetworkRoundTrip50msBudget= {'PASS' if round_trip_50ms else 'FAIL_INFORMATIONAL'}")
    print(f"ReportPath= {report_path}")
    print(f"ReportSHA256= {sha256_file(report_path)}")
    print("Stage08FixedInputNetworkProbe=PASS")
    print("Stage08FixedInputNetworkProbeEnd=")


def run_latency_benchmark(args: argparse.Namespace) -> None:
    """Measure the fixed-input request path on an explicit 20 Hz schedule."""

    config_path = Path(args.config).resolve()
    report_path = Path(args.benchmark_report_path).resolve()
    step_log_path = Path(args.benchmark_step_log_path).resolve()
    config = Protocol.load_config(config_path)

    require(args.benchmark_samples >= 1000, "benchmark_samples must be at least 1000")
    require(args.benchmark_warmup_requests >= 10, "benchmark_warmup_requests must be at least 10")
    require(not report_path.exists(), f"refusing to overwrite benchmark report: {report_path}")
    require(not step_log_path.exists(), f"refusing to overwrite benchmark step log: {step_log_path}")

    period_ns = int(config["evaluation"]["control_period_ns"])
    require(period_ns == 50_000_000, f"unexpected control period: {period_ns}")
    raw_obs = fixed_raw_observation()
    input_names = tuple(entry["name"] for entry in config["request"]["arrays"])
    raw_hash = sha256_arrays(raw_obs, input_names)
    require(raw_hash == FIXED_WIRE_ARRAYS_SHA256, f"fixed wire arrays hash mismatch: {raw_hash}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    step_log_path.parent.mkdir(parents=True, exist_ok=True)

    started_unix_ns = time.time_ns()
    client = RemotePolicyClient(
        config_path,
        timeout_mode=args.timeout_mode,
        bind_expected_client=not args.no_bind_expected_client,
    )
    step_stream: Any = None
    failure: Optional[str] = None
    failure_traceback: Optional[str] = None
    timeout_count = 0
    completed_warmups = 0
    completed_samples = 0
    schedule_origin_ns: Optional[int] = None
    benchmark_end_ns: Optional[int] = None

    action_latency_values: list[int] = []
    scheduled_action_latency_values: list[int] = []
    release_lateness_values: list[int] = []
    request_interval_values: list[int] = []
    serialize_values: list[int] = []
    round_trip_values: list[int] = []
    server_pre_inference_values: list[int] = []
    server_inference_values: list[int] = []
    server_post_inference_values: list[int] = []
    deadline_lateness_values: list[int] = []
    deadline_miss_count = 0
    last_actual_release_ns: Optional[int] = None

    try:
        step_stream = step_log_path.open("x", encoding="utf-8", buffering=1)

        client.start_episode(f"{args.benchmark_episode_prefix}-warmup")
        for _ in range(args.benchmark_warmup_requests):
            action, _ = client.request_raw(raw_obs)
            require(np.all(np.isfinite(action)), "warm-up action is not finite")
            completed_warmups += 1

        client.start_episode(f"{args.benchmark_episode_prefix}-measure")
        schedule_origin_ns = time.perf_counter_ns()

        for sample_index in range(args.benchmark_samples):
            scheduled_release_ns = schedule_origin_ns + sample_index * period_ns
            remaining_ns = scheduled_release_ns - time.perf_counter_ns()
            if remaining_ns > 0:
                time.sleep(remaining_ns / 1e9)

            actual_release_ns = time.perf_counter_ns()
            release_lateness_ns = max(actual_release_ns - scheduled_release_ns, 0)
            if last_actual_release_ns is not None:
                request_interval_values.append(actual_release_ns - last_actual_release_ns)
            last_actual_release_ns = actual_release_ns

            try:
                action, network_timing = client.request_raw(raw_obs)
            except (socket.timeout, TimeoutError):
                timeout_count += 1
                raise

            action_ready_ns = time.perf_counter_ns()
            action_latency_ns = action_ready_ns - actual_release_ns
            scheduled_action_latency_ns = action_ready_ns - scheduled_release_ns
            scheduled_deadline_ns = scheduled_release_ns + period_ns
            deadline_lateness_ns = max(action_ready_ns - scheduled_deadline_ns, 0)
            deadline_miss = deadline_lateness_ns > 0

            action_latency_values.append(action_latency_ns)
            scheduled_action_latency_values.append(scheduled_action_latency_ns)
            release_lateness_values.append(release_lateness_ns)
            serialize_values.append(int(network_timing["serialize_ns"]))
            round_trip_values.append(int(network_timing["round_trip_ns"]))
            server_pre_inference_values.append(int(network_timing["server_pre_inference_ns"]))
            server_inference_values.append(int(network_timing["server_inference_ns"]))
            server_post_inference_values.append(
                int(network_timing["server_postprocess_and_encode_ns"])
            )
            deadline_lateness_values.append(deadline_lateness_ns)
            deadline_miss_count += int(deadline_miss)
            completed_samples += 1

            write_jsonl_record(
                step_stream,
                {
                    "record_type": "latency_sample",
                    "sample_index": sample_index,
                    "scheduled_release_monotonic_ns": scheduled_release_ns,
                    "actual_release_monotonic_ns": actual_release_ns,
                    "action_ready_monotonic_ns": action_ready_ns,
                    "scheduled_deadline_monotonic_ns": scheduled_deadline_ns,
                    "release_lateness_ns": release_lateness_ns,
                    "action_latency_ns": action_latency_ns,
                    "scheduled_action_latency_ns": scheduled_action_latency_ns,
                    "deadline_lateness_ns": deadline_lateness_ns,
                    "deadline_miss": deadline_miss,
                    "action": [float(value) for value in action],
                    "network_timing": network_timing,
                },
            )

        benchmark_end_ns = time.perf_counter_ns()

    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        failure_traceback = traceback.format_exc()
        benchmark_end_ns = time.perf_counter_ns()
    finally:
        client.close()
        if step_stream is not None:
            step_stream.close()

    benchmark_completed = (
        failure is None
        and completed_warmups == args.benchmark_warmup_requests
        and completed_samples == args.benchmark_samples
        and timeout_count == 0
    )
    stable_20hz = benchmark_completed and deadline_miss_count == 0
    completed_unix_ns = time.time_ns()
    benchmark_duration_ns = (
        benchmark_end_ns - schedule_origin_ns
        if benchmark_end_ns is not None and schedule_origin_ns is not None
        else None
    )

    report: Dict[str, Any] = {
        "artifact": "stage08_paced_latency_benchmark",
        "artifact_version": 1,
        "script_version": SCRIPT_VERSION,
        "status": "PASS" if benchmark_completed else "FAIL",
        "stable_20hz": stable_20hz,
        "error": failure or "",
        "traceback": failure_traceback or "",
        "started_unix_ns": started_unix_ns,
        "completed_unix_ns": completed_unix_ns,
        "benchmark_duration_ns": benchmark_duration_ns,
        "protocol_config": str(config_path),
        "protocol_config_sha256": sha256_file(config_path),
        "client_script": str(Path(__file__).resolve()),
        "client_script_sha256": sha256_file(Path(__file__).resolve()),
        "endpoint": {"host": client.endpoint[0], "port": client.endpoint[1]},
        "timeout_mode": args.timeout_mode,
        "control_frequency_hz": int(config["evaluation"]["control_frequency_hz"]),
        "control_period_ns": period_ns,
        "fixed_wire_arrays_sha256": raw_hash,
        "requested_warmup_requests": args.benchmark_warmup_requests,
        "completed_warmup_requests": completed_warmups,
        "requested_samples": args.benchmark_samples,
        "completed_samples": completed_samples,
        "timeout_count": timeout_count,
        "deadline_miss_count": deadline_miss_count,
        "deadline_miss_rate": (
            deadline_miss_count / completed_samples if completed_samples else None
        ),
        "action_latency": summarize_nanoseconds(action_latency_values),
        "scheduled_action_latency": summarize_nanoseconds(scheduled_action_latency_values),
        "release_lateness": summarize_nanoseconds(release_lateness_values),
        "request_interval": summarize_nanoseconds(request_interval_values),
        "client_serialize": summarize_nanoseconds(serialize_values),
        "network_round_trip": summarize_nanoseconds(round_trip_values),
        "server_receive_decode_preprocess": summarize_nanoseconds(
            server_pre_inference_values
        ),
        "server_inference": summarize_nanoseconds(server_inference_values),
        "server_post_inference_before_send": summarize_nanoseconds(
            server_post_inference_values
        ),
        "deadline_lateness": summarize_nanoseconds(deadline_lateness_values),
        "step_log": str(step_log_path),
        "step_log_sha256": sha256_file(step_log_path) if step_log_path.exists() else None,
    }
    write_json_exclusive(report_path, report)

    if failure is not None:
        raise Stage08ClientError(failure)
    require(benchmark_completed, "latency benchmark did not complete")

    print("Stage08PacedLatencyBenchmarkBegin=")
    print(f"Endpoint= {client.endpoint[0]}:{client.endpoint[1]}")
    print(f"ControlFrequencyHz= {report['control_frequency_hz']}")
    print(f"WarmupRequests= {completed_warmups}")
    print(f"MeasuredSamples= {completed_samples}")
    print(f"ActionLatencyP50Ms= {report['action_latency']['p50_ms']:.3f}")
    print(f"ActionLatencyP95Ms= {report['action_latency']['p95_ms']:.3f}")
    print(f"ActionLatencyP99Ms= {report['action_latency']['p99_ms']:.3f}")
    print(f"ActionLatencyMaxMs= {report['action_latency']['max_ms']:.3f}")
    print(f"ReleaseLatenessP99Ms= {report['release_lateness']['p99_ms']:.3f}")
    print(f"RequestIntervalP50Ms= {report['request_interval']['p50_ms']:.3f}")
    print(f"RequestIntervalP99Ms= {report['request_interval']['p99_ms']:.3f}")
    print(f"DeadlineMissCount= {deadline_miss_count}")
    print(f"TimeoutCount= {timeout_count}")
    print(f"Stable20Hz= {stable_20hz}")
    print(f"StepLogPath= {step_log_path}")
    print(f"ReportPath= {report_path}")
    print(f"ReportSHA256= {sha256_file(report_path)}")
    print("Stage08PacedLatencyBenchmark=PASS")
    print("Stage08PacedLatencyBenchmarkEnd=")


def run_simulation_rollouts(args: argparse.Namespace) -> None:
    """Run Lift in MuJoCo while every policy action is produced by Jetson."""

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    summary_path = Path(args.summary_path).resolve()
    step_log_path = Path(args.step_log_path).resolve()
    trajectory_path = Path(args.trajectory_path).resolve() if args.trajectory_path else None
    video_path = Path(args.video_path).resolve() if args.video_path else None

    config = Protocol.load_config(config_path)
    require(checkpoint_path.is_file(), f"checkpoint does not exist: {checkpoint_path}")
    require(args.n_rollouts > 0, "n_rollouts must be positive")
    require(args.horizon > 0, "horizon must be positive")
    require(args.seed >= 0, "seed must be nonnegative")
    require(args.video_episodes >= 0, "video_episodes must be nonnegative")
    require(not summary_path.exists(), f"refusing to overwrite summary: {summary_path}")
    require(not step_log_path.exists(), f"refusing to overwrite step log: {step_log_path}")

    for artifact in (trajectory_path, video_path):
        if artifact is None:
            continue
        require(not artifact.exists(), f"refusing to overwrite artifact: {artifact}")
        require(
            not temporary_artifact_path(artifact).exists(),
            f"in-progress artifact already exists: {temporary_artifact_path(artifact)}",
        )

    checkpoint_sha256 = sha256_file(checkpoint_path)
    require(
        checkpoint_sha256 == config["evaluation"]["checkpoint_sha256"],
        f"checkpoint SHA256 mismatch: {checkpoint_sha256}",
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    step_log_path.parent.mkdir(parents=True, exist_ok=True)
    if trajectory_path is not None:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)

    started_unix_ns = time.time_ns()
    client: Optional[RemotePolicyClient] = None
    env: Any = None
    step_stream: Any = None
    trajectory_file: Any = None
    trajectory_data_group: Any = None
    video_writer: Any = None
    trajectory_temp = temporary_artifact_path(trajectory_path) if trajectory_path is not None else None
    video_temp = temporary_artifact_path(video_path) if video_path is not None else None
    failure: Optional[str] = None
    failure_traceback: Optional[str] = None
    acceptance_failure: Optional[str] = None
    episode_summaries: list[Dict[str, Any]] = []
    environment_info: Dict[str, Any] = {}
    total_transitions = 0
    video_frames = 0
    action_latency_values: list[int] = []
    round_trip_values: list[int] = []
    inference_values: list[int] = []
    control_iteration_values: list[int] = []
    action_deadline_misses = 0
    control_period_overruns = 0
    period_ns = int(config["evaluation"]["control_period_ns"])

    try:
        import h5py
        import imageio.v2 as imageio
        import torch
        import robomimic.utils.file_utils as FileUtils
        import robomimic.utils.obs_utils as ObsUtils

        checkpoint_load_start = time.perf_counter_ns()
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_load_ns = time.perf_counter_ns() - checkpoint_load_start
        restored_config, _ = FileUtils.config_from_checkpoint(
            ckpt_dict=checkpoint,
            verbose=False,
        )
        ObsUtils.initialize_obs_utils_with_config(restored_config)

        environment_create_start = time.perf_counter_ns()
        env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=checkpoint,
            render=False,
            render_offscreen=True,
            verbose=False,
        )
        environment_create_ns = time.perf_counter_ns() - environment_create_start
        del checkpoint

        require(env.name == "Lift", f"unexpected environment name: {env.name}")
        require(env.action_dimension == 7, f"unexpected action dimension: {env.action_dimension}")
        environment_info = {
            "name": env.name,
            "version": env.version,
            "action_dimension": int(env.action_dimension),
            "checkpoint_load_ms": checkpoint_load_ns / 1e6,
            "environment_create_ms": environment_create_ns / 1e6,
        }

        # Match the official Stage 7 independent evaluation: seed after the
        # environment has been constructed and before the first reset.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        step_stream = step_log_path.open("x", encoding="utf-8", buffering=1)
        if trajectory_temp is not None:
            trajectory_file = h5py.File(trajectory_temp, "x")
            trajectory_data_group = trajectory_file.create_group("data")
        if video_temp is not None and args.video_episodes > 0:
            video_writer = imageio.get_writer(str(video_temp), fps=20)

        client = RemotePolicyClient(
            config_path,
            timeout_mode=args.timeout_mode,
            bind_expected_client=not args.no_bind_expected_client,
        )

        for episode_index in range(args.n_rollouts):
            episode_id = f"{args.episode_prefix}-{episode_index:03d}"
            client.start_episode(episode_id)

            reset_start = time.perf_counter_ns()
            observation = env.reset()
            episode_initial_state_dict = env.get_state()
            current_state_dict = episode_initial_state_dict
            observation = env.reset_to(episode_initial_state_dict)
            reset_ns = time.perf_counter_ns() - reset_start
            require(observation is not None, "reset_to did not return an observation")

            actions = []
            rewards = []
            dones = []
            states = []
            episode_return = 0.0
            success = False
            done = False
            termination_reason = "horizon"
            first_wire_sha256: Optional[str] = None

            for step_index in range(args.horizon):
                iteration_start = time.perf_counter_ns()
                wire_start = time.perf_counter_ns()
                wire_observation = processed_observation_to_wire(observation)
                wire_end = time.perf_counter_ns()
                if first_wire_sha256 is None:
                    wire_names = tuple(entry["name"] for entry in config["request"]["arrays"])
                    first_wire_sha256 = sha256_arrays(wire_observation, wire_names)

                action, network_timing = client.request_raw(wire_observation)
                action_ready_ns = time.perf_counter_ns()
                next_observation, reward, done, _ = env.step(action)
                environment_step_end = time.perf_counter_ns()

                episode_return += float(reward)
                current_success = bool(env.is_success()["task"])
                success = success or current_success

                wire_conversion_ns = wire_end - wire_start
                action_latency_ns = action_ready_ns - iteration_start
                environment_step_ns = environment_step_end - action_ready_ns
                control_iteration_ns = environment_step_end - iteration_start
                action_deadline_miss = action_latency_ns > period_ns
                control_period_overrun = control_iteration_ns > period_ns

                action_latency_values.append(action_latency_ns)
                round_trip_values.append(int(network_timing["round_trip_ns"]))
                inference_values.append(int(network_timing["server_inference_ns"]))
                control_iteration_values.append(control_iteration_ns)
                action_deadline_misses += int(action_deadline_miss)
                control_period_overruns += int(control_period_overrun)

                record = {
                    "record_type": "inference_step",
                    "episode_index": episode_index,
                    "episode_id": episode_id,
                    "step_index": step_index,
                    "sequence": int(network_timing["sequence"]),
                    "reward": float(reward),
                    "done": bool(done),
                    "success": current_success,
                    "action": [float(value) for value in action],
                    "wire_conversion_ns": wire_conversion_ns,
                    "action_latency_ns": action_latency_ns,
                    "environment_step_ns": environment_step_ns,
                    "control_iteration_ns": control_iteration_ns,
                    "action_deadline_miss": action_deadline_miss,
                    "control_period_overrun": control_period_overrun,
                    "network_timing": network_timing,
                }
                write_jsonl_record(step_stream, record)

                actions.append(action.copy())
                rewards.append(float(reward))
                dones.append(bool(done))
                # Match run_trained_agent.py: store the simulator state that
                # produced this observation, immediately before applying the
                # corresponding action.
                states.append(np.asarray(current_state_dict["states"]).copy())

                if video_writer is not None and episode_index < args.video_episodes:
                    frame = np.concatenate(
                        (
                            wire_observation["agentview_image"],
                            wire_observation["robot0_eye_in_hand_image"],
                        ),
                        axis=1,
                    )
                    # 84x168 -> 336x672, both dimensions are codec-friendly.
                    frame = np.repeat(np.repeat(frame, 4, axis=0), 4, axis=1)
                    video_writer.append_data(frame)
                    video_frames += 1

                if done and success:
                    termination_reason = "done_and_success"
                    break
                if done:
                    termination_reason = "done"
                    break
                if args.terminate_on_success and success:
                    termination_reason = "success"
                    break

                observation = deepcopy(next_observation)
                current_state_dict = env.get_state()

            episode_horizon = len(actions)
            total_transitions += episode_horizon
            episode_summary = {
                "episode_index": episode_index,
                "episode_id": episode_id,
                "return": episode_return,
                "horizon": episode_horizon,
                "success": success,
                "termination_reason": termination_reason,
                "reset_ms": reset_ns / 1e6,
                "first_wire_observation_sha256": first_wire_sha256,
            }
            episode_summaries.append(episode_summary)
            write_jsonl_record(
                step_stream,
                {"record_type": "episode_summary", **episode_summary},
            )

            if trajectory_data_group is not None:
                episode_group = trajectory_data_group.create_group(f"demo_{episode_index}")
                episode_group.create_dataset("actions", data=np.asarray(actions, dtype=np.float32))
                episode_group.create_dataset("states", data=np.asarray(states))
                episode_group.create_dataset("rewards", data=np.asarray(rewards, dtype=np.float64))
                episode_group.create_dataset("dones", data=np.asarray(dones, dtype=np.bool_))
                if "model" in episode_initial_state_dict:
                    episode_group.attrs["model_file"] = episode_initial_state_dict["model"]
                if "ep_meta" in episode_initial_state_dict:
                    episode_group.attrs["ep_meta"] = episode_initial_state_dict["ep_meta"]
                episode_group.attrs["num_samples"] = episode_horizon
                trajectory_file.flush()

            print(
                f"NetworkRolloutEpisode=index:{episode_index};success:{int(success)};"
                f"horizon:{episode_horizon};return:{episode_return:.6f};"
                f"termination:{termination_reason}",
                flush=True,
            )

        success_count = sum(int(episode["success"]) for episode in episode_summaries)
        if args.require_all_success and success_count != args.n_rollouts:
            acceptance_failure = (
                f"required all rollouts to succeed, got {success_count}/{args.n_rollouts}"
            )

        if trajectory_data_group is not None:
            trajectory_data_group.attrs["total"] = total_transitions
            trajectory_data_group.attrs["env_args"] = json.dumps(env.serialize(), indent=4)
            trajectory_data_group.attrs["stage08_seed"] = args.seed
            trajectory_data_group.attrs["stage08_checkpoint_sha256"] = checkpoint_sha256

    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        failure_traceback = traceback.format_exc()
    finally:
        if client is not None:
            client.close()
        if video_writer is not None:
            try:
                video_writer.close()
            except Exception as exc:
                if failure is None:
                    failure = f"video close failed: {type(exc).__name__}: {exc}"
                    failure_traceback = traceback.format_exc()
        if trajectory_file is not None:
            try:
                trajectory_file.close()
            except Exception as exc:
                if failure is None:
                    failure = f"trajectory close failed: {type(exc).__name__}: {exc}"
                    failure_traceback = traceback.format_exc()
        if step_stream is not None:
            step_stream.close()
        if env is not None:
            try:
                env.env.close()
            except Exception:
                pass

        if failure is None:
            if trajectory_temp is not None:
                trajectory_temp.rename(trajectory_path)
            if video_temp is not None and video_temp.exists():
                video_temp.rename(video_path)

        success_count = sum(int(episode["success"]) for episode in episode_summaries)
        execution_status = "PASS" if failure is None else "FAIL"
        acceptance_status = (
            "PASS" if failure is None and acceptance_failure is None else "FAIL"
        )
        completed_unix_ns = time.time_ns()
        summary: Dict[str, Any] = {
            "artifact": "stage08_network_simulation_rollouts",
            "artifact_version": 1,
            "script_version": SCRIPT_VERSION,
            "status": acceptance_status,
            "execution_status": execution_status,
            "error": failure or acceptance_failure or "",
            "traceback": failure_traceback or "",
            "started_unix_ns": started_unix_ns,
            "completed_unix_ns": completed_unix_ns,
            "protocol_config": str(config_path),
            "protocol_config_sha256": sha256_file(config_path),
            "client_script": str(Path(__file__).resolve()),
            "client_script_sha256": sha256_file(Path(__file__).resolve()),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "endpoint": {
                "host": str(config["transport"]["server_host"]),
                "port": int(config["transport"]["server_port"]),
            },
            "timeout_mode": args.timeout_mode,
            "seed": args.seed,
            "requested_rollouts": args.n_rollouts,
            "completed_rollouts": len(episode_summaries),
            "horizon_limit": args.horizon,
            "terminate_on_success": args.terminate_on_success,
            "require_all_success": args.require_all_success,
            "success_count": success_count,
            "success_rate": (
                success_count / len(episode_summaries) if episode_summaries else 0.0
            ),
            "mean_horizon": (
                float(np.mean([episode["horizon"] for episode in episode_summaries]))
                if episode_summaries
                else None
            ),
            "mean_return": (
                float(np.mean([episode["return"] for episode in episode_summaries]))
                if episode_summaries
                else None
            ),
            "total_transitions": total_transitions,
            "control_period_ns": period_ns,
            "action_deadline_miss_count": action_deadline_misses,
            "control_period_overrun_count": control_period_overruns,
            "action_latency": summarize_nanoseconds(action_latency_values),
            "network_round_trip": summarize_nanoseconds(round_trip_values),
            "server_inference": summarize_nanoseconds(inference_values),
            "control_iteration": summarize_nanoseconds(control_iteration_values),
            "environment": environment_info,
            "episodes": episode_summaries,
            "step_log": str(step_log_path),
            "step_log_sha256": sha256_file(step_log_path) if step_log_path.exists() else None,
            "trajectory": str(trajectory_path) if trajectory_path is not None else None,
            "trajectory_sha256": (
                sha256_file(trajectory_path)
                if trajectory_path is not None and trajectory_path.exists()
                else None
            ),
            "video": str(video_path) if video_path is not None else None,
            "video_sha256": (
                sha256_file(video_path)
                if video_path is not None and video_path.exists()
                else None
            ),
            "video_frames": video_frames,
        }
        if not summary_path.exists():
            write_json_exclusive(summary_path, summary)

    if failure is not None:
        raise Stage08ClientError(failure)
    if acceptance_failure is not None:
        raise Stage08ClientError(acceptance_failure)

    report = json.loads(summary_path.read_text(encoding="utf-8"))
    print("Stage08NetworkSimulationRolloutBegin=")
    print(f"Environment= {report['environment'].get('name')}")
    print(f"Seed= {report['seed']}")
    print(f"Rollouts= {report['completed_rollouts']}")
    print(f"SuccessCount= {report['success_count']}")
    print(f"SuccessRate= {report['success_rate']:.6f}")
    print(f"MeanHorizon= {report['mean_horizon']:.6f}")
    print(f"MeanReturn= {report['mean_return']:.6f}")
    print(f"ActionLatencyP50Ms= {report['action_latency']['p50_ms']:.3f}")
    print(f"ActionLatencyP95Ms= {report['action_latency']['p95_ms']:.3f}")
    print(f"ActionLatencyMaxMs= {report['action_latency']['max_ms']:.3f}")
    print(f"ActionDeadlineMissCount= {report['action_deadline_miss_count']}")
    print(f"ControlPeriodOverrunCount= {report['control_period_overrun_count']}")
    print(f"StepLogPath= {step_log_path}")
    if trajectory_path is not None:
        print(f"TrajectoryPath= {trajectory_path}")
    if video_path is not None:
        print(f"VideoPath= {video_path}")
    print(f"SummaryPath= {summary_path}")
    print(f"SummarySHA256= {sha256_file(summary_path)}")
    print("Stage08NetworkSimulationRollout=PASS")
    print("Stage08NetworkSimulationRolloutEnd=")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Stage 08 protocol JSON")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true", help="run local deterministic checks")
    mode.add_argument("--fixed-input-probe", action="store_true", help="send one deterministic request")
    mode.add_argument(
        "--latency-benchmark",
        action="store_true",
        help="run a paced 20 Hz fixed-input latency benchmark",
    )
    mode.add_argument(
        "--simulation-rollouts",
        action="store_true",
        help="run Lift in MuJoCo with actions served by Jetson",
    )
    parser.add_argument("--timeout-mode", choices=("functional", "realtime"), default="functional")
    parser.add_argument("--episode-id", default="stage08-fixed-input-network-probe")
    parser.add_argument(
        "--report-path",
        default="/home/lx/robomimic/runs/lift_ph/hil/stage08-fixed-input-network-probe.json",
    )
    parser.add_argument("--expected-action-tolerance", type=float, default=1e-6)
    parser.add_argument("--no-bind-expected-client", action="store_true")
    parser.add_argument("--benchmark-samples", type=int, default=1000)
    parser.add_argument("--benchmark-warmup-requests", type=int, default=20)
    parser.add_argument("--benchmark-episode-prefix", default="stage08-paced-latency")
    parser.add_argument("--benchmark-report-path")
    parser.add_argument("--benchmark-step-log-path")
    parser.add_argument("--checkpoint", help="checkpoint used to reconstruct the Lift environment")
    parser.add_argument("--n-rollouts", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument(
        "--terminate-on-success",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--require-all-success", action="store_true")
    parser.add_argument("--episode-prefix", default="stage08-network-rollout")
    parser.add_argument("--summary-path")
    parser.add_argument("--step-log-path")
    parser.add_argument("--trajectory-path")
    parser.add_argument("--video-path")
    parser.add_argument("--video-episodes", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        client_self_test(Path(args.config).resolve())
    elif args.fixed_input_probe:
        run_fixed_input_probe(args)
    elif args.latency_benchmark:
        require(
            args.benchmark_report_path is not None,
            "--benchmark-report-path is required for the latency benchmark",
        )
        require(
            args.benchmark_step_log_path is not None,
            "--benchmark-step-log-path is required for the latency benchmark",
        )
        run_latency_benchmark(args)
    else:
        require(args.checkpoint is not None, "--checkpoint is required for simulation rollouts")
        require(args.summary_path is not None, "--summary-path is required for simulation rollouts")
        require(args.step_log_path is not None, "--step-log-path is required for simulation rollouts")
        run_simulation_rollouts(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Stage08LiftSimClientError={type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
