#!/usr/bin/env python3
"""Binary TCP protocol helpers for the Stage 8 Lift-PH Jetson HIL link."""

import argparse
import hashlib
import json
import math
import socket
import struct
import zlib
from pathlib import Path

import numpy as np


OUTER_HEADER = struct.Struct("!4sI")
INNER_HEADER_LENGTH = struct.Struct("!I")
DTYPES = {
    "uint8": np.dtype("uint8"),
    "float32": np.dtype("<f4"),
}


class ProtocolError(RuntimeError):
    """Raised when a frame or message violates the protocol contract."""


class ConnectionClosed(ProtocolError):
    """Raised when the peer closes a socket before a full frame is received."""


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def load_config(path):
    """Load and validate a Stage 8 protocol JSON file."""
    path = Path(path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot load protocol config {path}: {exc}") from exc
    validate_config(config)
    return config


def _expected_array_bytes(specifications):
    total = 0
    for spec in specifications:
        require(spec.get("dtype") in DTYPES, f"unsupported dtype: {spec.get('dtype')!r}")
        shape = spec.get("shape")
        require(
            isinstance(shape, list)
            and bool(shape)
            and all(type(value) is int and value > 0 for value in shape),
            f"invalid shape for {spec.get('name')!r}",
        )
        total += math.prod(shape) * DTYPES[spec["dtype"]].itemsize
    return total


def validate_config(config):
    """Reject configuration drift that would make the wire format ambiguous."""
    require(isinstance(config, dict), "config must be a JSON object")
    require(config.get("schema_version") == 1, "unsupported schema_version")
    require(config.get("stage") == 8, "config stage must be 8")
    require(config.get("protocol_name") == "robomimic-lift-ph-hil", "unexpected protocol_name")
    require(config.get("protocol_version") == 1, "unsupported protocol_version")

    transport = config.get("transport")
    encoding = config.get("encoding")
    sequence = config.get("sequence")
    safety = config.get("safety")
    evaluation = config.get("evaluation")
    require(all(isinstance(item, dict) for item in (transport, encoding, sequence, safety, evaluation)), "missing config section")

    magic = transport.get("magic_ascii")
    require(isinstance(magic, str) and len(magic.encode("ascii")) == 4, "magic_ascii must be four ASCII bytes")
    require(transport.get("type") == "tcp", "transport must be tcp")
    require(transport.get("connection_model") == "single_persistent_connection", "unexpected connection model")
    require(transport.get("outer_length_prefix_bytes") == 4, "outer length prefix must be four bytes")
    require(transport.get("outer_length_prefix_byte_order") == "big", "outer length prefix must be big-endian")
    require(type(transport.get("server_port")) is int and 1 <= transport["server_port"] <= 65535, "invalid server port")
    require(type(transport.get("max_payload_bytes")) is int and 1024 <= transport["max_payload_bytes"] <= 16 * 1024 * 1024, "invalid max payload")
    for key in (
        "connect_timeout_seconds",
        "functional_response_timeout_seconds",
        "realtime_response_timeout_seconds",
        "server_idle_timeout_seconds",
    ):
        require(type(transport.get(key)) in (int, float) and transport[key] > 0, f"invalid {key}")
    require(
        transport["realtime_response_timeout_seconds"]
        <= transport["functional_response_timeout_seconds"],
        "realtime timeout cannot exceed functional timeout",
    )

    require(encoding.get("header") == "utf8_json", "header encoding must be utf8_json")
    require(encoding.get("header_length_prefix_bytes") == 4, "header length prefix must be four bytes")
    require(encoding.get("header_length_prefix_byte_order") == "big", "header length prefix must be big-endian")
    require(encoding.get("arrays") == "uncompressed_contiguous_c_order", "unexpected array encoding")
    require(encoding.get("float_byte_order") == "little", "float byte order must be little-endian")
    require(encoding.get("array_checksum") == "crc32", "array checksum must be crc32")
    require(encoding.get("compression") == "none", "compression must be disabled")
    require(encoding.get("pickle_allowed") is False, "pickle must be disabled")

    expected_sections = {
        "request": {
            "message_type": "inference_request",
            "metadata": {"episode_id", "sequence_id", "client_send_ns"},
            "arrays": [
                ("agentview_image", "uint8", [84, 84, 3]),
                ("robot0_eye_in_hand_image", "uint8", [84, 84, 3]),
                ("robot0_eef_pos", "float32", [3]),
                ("robot0_eef_quat", "float32", [4]),
                ("robot0_gripper_qpos", "float32", [2]),
            ],
        },
        "response": {
            "message_type": "inference_response",
            "metadata": {
                "episode_id",
                "sequence_id",
                "server_receive_ns",
                "inference_start_ns",
                "inference_end_ns",
                "server_send_ns",
                "status",
                "error_message",
            },
            "arrays": [("action", "float32", [7])],
        },
    }

    for section_name, expected in expected_sections.items():
        section = config.get(section_name)
        require(isinstance(section, dict), f"missing {section_name} section")
        require(section.get("message_type") == expected["message_type"], f"invalid {section_name} message type")
        require(set(section.get("metadata", {})) == expected["metadata"], f"invalid {section_name} metadata fields")
        specifications = section.get("arrays")
        require(isinstance(specifications, list), f"invalid {section_name} arrays")
        actual = [(item.get("name"), item.get("dtype"), item.get("shape")) for item in specifications]
        require(actual == expected["arrays"], f"invalid {section_name} array contract")
        byte_count = _expected_array_bytes(specifications)
        require(byte_count == section.get("expected_array_payload_bytes"), f"invalid {section_name} byte count")
        require(byte_count < transport["max_payload_bytes"], f"{section_name} arrays exceed max payload")

    require(sequence.get("first_sequence_id") == 0, "first sequence must be zero")
    require(sequence.get("require_strictly_increasing") is True, "sequence IDs must increase")
    require(sequence.get("reject_duplicates") is True, "duplicate rejection must be enabled")
    require(sequence.get("reject_out_of_order") is True, "out-of-order rejection must be enabled")
    require(sequence.get("require_response_match") is True, "response matching must be enabled")

    require(type(safety.get("max_episode_id_utf8_bytes")) is int and safety["max_episode_id_utf8_bytes"] > 0, "invalid episode ID limit")
    require(type(safety.get("max_error_message_utf8_bytes")) is int and safety["max_error_message_utf8_bytes"] > 0, "invalid error message limit")
    require(type(safety.get("action_absolute_limit")) in (int, float) and safety["action_absolute_limit"] >= 1.0, "invalid action limit")
    for key in (
        "reject_unknown_message_type",
        "reject_unknown_arrays",
        "reject_missing_arrays",
        "reject_shape_mismatch",
        "reject_dtype_mismatch",
        "reject_nonfinite_floats",
        "zero_action_on_error",
    ):
        require(safety.get(key) is True, f"safety flag {key} must be true")

    frequency = evaluation.get("control_frequency_hz")
    period = evaluation.get("control_period_ns")
    require(type(frequency) is int and frequency > 0, "invalid control frequency")
    require(period == round(1_000_000_000 / frequency), "control period does not match frequency")
    checkpoint_sha = evaluation.get("checkpoint_sha256")
    require(
        isinstance(checkpoint_sha, str)
        and len(checkpoint_sha) == 64
        and all(character in "0123456789abcdef" for character in checkpoint_sha),
        "invalid checkpoint SHA256",
    )


def _section_for_message(config, message_type):
    for section_name in ("request", "response"):
        section = config[section_name]
        if section["message_type"] == message_type:
            return section_name, section
    raise ProtocolError(f"unknown message_type: {message_type!r}")


def _validate_nonnegative_integer(value, field):
    require(type(value) is int and value >= 0, f"{field} must be a nonnegative integer")


def _validate_metadata(config, section_name, metadata):
    require(isinstance(metadata, dict), "metadata must be an object")
    expected = set(config[section_name]["metadata"])
    require(set(metadata) == expected, f"metadata fields do not match {section_name} contract")

    episode_id = metadata["episode_id"]
    require(isinstance(episode_id, str) and bool(episode_id), "episode_id must be nonempty")
    require(
        len(episode_id.encode("utf-8")) <= config["safety"]["max_episode_id_utf8_bytes"],
        "episode_id is too long",
    )
    _validate_nonnegative_integer(metadata["sequence_id"], "sequence_id")

    if section_name == "request":
        _validate_nonnegative_integer(metadata["client_send_ns"], "client_send_ns")
        return

    timestamp_names = (
        "server_receive_ns",
        "inference_start_ns",
        "inference_end_ns",
        "server_send_ns",
    )
    for name in timestamp_names:
        _validate_nonnegative_integer(metadata[name], name)
    timestamps = [metadata[name] for name in timestamp_names]
    require(timestamps == sorted(timestamps), "server timestamps are out of order")

    status = metadata["status"]
    require(status in config["response"]["metadata"]["status"], "invalid response status")
    error_message = metadata["error_message"]
    require(isinstance(error_message, str), "error_message must be a string")
    require(
        len(error_message.encode("utf-8")) <= config["safety"]["max_error_message_utf8_bytes"],
        "error_message is too long",
    )
    require(status != "ok" or error_message == "", "successful response must have an empty error_message")
    require(status != "error" or bool(error_message), "error response must describe the error")


def _normalized_arrays(config, section_name, arrays):
    require(isinstance(arrays, dict), "arrays must be an object")
    specifications = config[section_name]["arrays"]
    expected_names = [spec["name"] for spec in specifications]
    require(set(arrays) == set(expected_names), f"array names do not match {section_name} contract")

    result = {}
    for spec in specifications:
        name = spec["name"]
        value = np.asarray(arrays[name])
        expected_dtype = DTYPES[spec["dtype"]]
        require(value.dtype == expected_dtype, f"{name} dtype must be {spec['dtype']}, got {value.dtype}")
        require(list(value.shape) == spec["shape"], f"{name} shape must be {spec['shape']}, got {list(value.shape)}")
        if np.issubdtype(value.dtype, np.floating) and config["safety"]["reject_nonfinite_floats"]:
            require(bool(np.isfinite(value).all()), f"{name} contains nonfinite values")
        if section_name == "response" and name == "action":
            limit = float(config["safety"]["action_absolute_limit"])
            require(bool((np.abs(value) <= limit).all()), f"action exceeds absolute limit {limit}")
        result[name] = np.ascontiguousarray(value, dtype=expected_dtype)
    return result


def validate_message(config, message_type, metadata, arrays):
    """Validate one logical request or response without encoding it."""
    section_name, _ = _section_for_message(config, message_type)
    _validate_metadata(config, section_name, metadata)
    return _normalized_arrays(config, section_name, arrays)


def pack_message(config, message_type, metadata, arrays):
    """Encode a logical message into an inner payload, without the TCP frame prefix."""
    normalized = validate_message(config, message_type, metadata, arrays)
    section_name, section = _section_for_message(config, message_type)
    descriptors = []
    chunks = []
    offset = 0

    for spec in section["arrays"]:
        name = spec["name"]
        raw = normalized[name].tobytes(order="C")
        descriptors.append({
            "name": name,
            "dtype": DTYPES[spec["dtype"]].str,
            "shape": spec["shape"],
            "offset": offset,
            "nbytes": len(raw),
            "crc32": f"{zlib.crc32(raw) & 0xffffffff:08x}",
        })
        chunks.append(raw)
        offset += len(raw)

    require(offset == section["expected_array_payload_bytes"], f"unexpected {section_name} array byte count")
    header = {
        "protocol_name": config["protocol_name"],
        "protocol_version": config["protocol_version"],
        "message_type": message_type,
        "metadata": metadata,
        "arrays": descriptors,
    }
    try:
        header_bytes = json.dumps(
            header,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ProtocolError(f"header is not JSON encodable: {exc}") from exc

    payload = INNER_HEADER_LENGTH.pack(len(header_bytes)) + header_bytes + b"".join(chunks)
    require(len(payload) <= config["transport"]["max_payload_bytes"], "encoded payload exceeds configured maximum")
    return payload


def unpack_message(config, payload, expected_message_type=None):
    """Decode and validate an inner payload returned by :func:`pack_message`."""
    require(isinstance(payload, (bytes, bytearray, memoryview)), "payload must be bytes-like")
    payload = bytes(payload)
    require(len(payload) <= config["transport"]["max_payload_bytes"], "payload exceeds configured maximum")
    require(len(payload) >= INNER_HEADER_LENGTH.size, "payload is shorter than header length prefix")

    header_length = INNER_HEADER_LENGTH.unpack(payload[: INNER_HEADER_LENGTH.size])[0]
    require(header_length > 0, "JSON header is empty")
    header_end = INNER_HEADER_LENGTH.size + header_length
    require(header_end <= len(payload), "JSON header length exceeds payload")

    try:
        header = json.loads(payload[INNER_HEADER_LENGTH.size : header_end].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid JSON header: {exc}") from exc
    require(isinstance(header, dict), "message header must be an object")
    require(
        set(header) == {"protocol_name", "protocol_version", "message_type", "metadata", "arrays"},
        "message header fields are invalid",
    )
    require(header["protocol_name"] == config["protocol_name"], "protocol_name mismatch")
    require(header["protocol_version"] == config["protocol_version"], "protocol_version mismatch")
    message_type = header["message_type"]
    if expected_message_type is not None:
        require(message_type == expected_message_type, f"expected {expected_message_type}, got {message_type}")
    section_name, section = _section_for_message(config, message_type)

    descriptors = header["arrays"]
    require(isinstance(descriptors, list), "array descriptors must be a list")
    require(len(descriptors) == len(section["arrays"]), "array descriptor count mismatch")
    array_blob = payload[header_end:]
    arrays = {}
    expected_offset = 0

    for spec, descriptor in zip(section["arrays"], descriptors):
        require(isinstance(descriptor, dict), "array descriptor must be an object")
        require(
            set(descriptor) == {"name", "dtype", "shape", "offset", "nbytes", "crc32"},
            "array descriptor fields are invalid",
        )
        name = spec["name"]
        dtype = DTYPES[spec["dtype"]]
        expected_nbytes = math.prod(spec["shape"]) * dtype.itemsize
        require(descriptor["name"] == name, f"array order or name mismatch for {name}")
        require(descriptor["dtype"] == dtype.str, f"wire dtype mismatch for {name}")
        require(descriptor["shape"] == spec["shape"], f"wire shape mismatch for {name}")
        require(descriptor["offset"] == expected_offset, f"noncontiguous array offset for {name}")
        require(descriptor["nbytes"] == expected_nbytes, f"wire byte count mismatch for {name}")
        end = expected_offset + expected_nbytes
        require(end <= len(array_blob), f"array bytes truncated for {name}")
        raw = array_blob[expected_offset:end]
        require(descriptor["crc32"] == f"{zlib.crc32(raw) & 0xffffffff:08x}", f"CRC32 mismatch for {name}")
        arrays[name] = np.frombuffer(raw, dtype=dtype).reshape(spec["shape"]).copy()
        expected_offset = end

    require(expected_offset == len(array_blob), "unexpected trailing bytes after arrays")
    normalized = validate_message(config, message_type, header["metadata"], arrays)
    return message_type, dict(header["metadata"]), normalized


def _recv_exact(sock, byte_count):
    chunks = []
    remaining = byte_count
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionClosed(f"peer closed with {remaining} bytes still expected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(sock, config, payload):
    """Send one payload with magic and a four-byte unsigned big-endian length."""
    require(isinstance(payload, (bytes, bytearray, memoryview)), "payload must be bytes-like")
    payload = bytes(payload)
    require(bool(payload), "cannot send an empty payload")
    require(len(payload) <= config["transport"]["max_payload_bytes"], "payload exceeds configured maximum")
    magic = config["transport"]["magic_ascii"].encode("ascii")
    sock.sendall(OUTER_HEADER.pack(magic, len(payload)) + payload)


def recv_frame(sock, config):
    """Receive one length-prefixed payload from a stream socket."""
    outer = _recv_exact(sock, OUTER_HEADER.size)
    magic, payload_length = OUTER_HEADER.unpack(outer)
    expected_magic = config["transport"]["magic_ascii"].encode("ascii")
    require(magic == expected_magic, f"frame magic mismatch: {magic!r}")
    require(payload_length > 0, "frame payload is empty")
    require(payload_length <= config["transport"]["max_payload_bytes"], "frame payload exceeds configured maximum")
    return _recv_exact(sock, payload_length)


def send_message(sock, config, message_type, metadata, arrays):
    payload = pack_message(config, message_type, metadata, arrays)
    send_frame(sock, config, payload)
    return len(payload)


def recv_message(sock, config, expected_message_type=None):
    payload = recv_frame(sock, config)
    return unpack_message(config, payload, expected_message_type=expected_message_type)


def _expect_protocol_error(label, function):
    try:
        function()
    except ProtocolError:
        print(f"{label}=PASS")
        return
    raise RuntimeError(f"{label} did not raise ProtocolError")


def run_self_test(config_path):
    """Exercise valid round trips and deliberate contract violations."""
    config_path = Path(config_path)
    config = load_config(config_path)
    pixels = np.arange(84 * 84 * 3, dtype=np.uint32).reshape(84, 84, 3)
    request_arrays = {
        "agentview_image": ((pixels * 17 + 13) % 256).astype(np.uint8),
        "robot0_eye_in_hand_image": ((pixels * 29 + 7) % 256).astype(np.uint8),
        "robot0_eef_pos": np.array([0.05, -0.10, 0.90], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.02, -0.02], dtype=np.float32),
    }
    request_metadata = {
        "episode_id": "stage08-protocol-self-test",
        "sequence_id": 0,
        "client_send_ns": 1_700_000_000_000_000_000,
    }
    request_payload = pack_message(
        config,
        config["request"]["message_type"],
        request_metadata,
        request_arrays,
    )
    request_type, decoded_request_metadata, decoded_request_arrays = unpack_message(
        config,
        request_payload,
        expected_message_type=config["request"]["message_type"],
    )
    require(request_type == config["request"]["message_type"], "request type changed")
    require(decoded_request_metadata == request_metadata, "request metadata changed")
    require(
        all(np.array_equal(decoded_request_arrays[name], value) for name, value in request_arrays.items()),
        "request arrays changed",
    )

    response_metadata = {
        "episode_id": request_metadata["episode_id"],
        "sequence_id": request_metadata["sequence_id"],
        "server_receive_ns": 1_700_000_000_001_000_000,
        "inference_start_ns": 1_700_000_000_001_100_000,
        "inference_end_ns": 1_700_000_000_030_100_000,
        "server_send_ns": 1_700_000_000_030_200_000,
        "status": "ok",
        "error_message": "",
    }
    response_arrays = {
        "action": np.array(
            [0.719678, -0.455591, -0.387794, -0.057099, -0.007840, 0.276168, -1.0],
            dtype=np.float32,
        )
    }
    response_payload = pack_message(
        config,
        config["response"]["message_type"],
        response_metadata,
        response_arrays,
    )
    _, decoded_response_metadata, decoded_response_arrays = unpack_message(
        config,
        response_payload,
        expected_message_type=config["response"]["message_type"],
    )
    require(decoded_response_metadata == response_metadata, "response metadata changed")
    require(np.array_equal(decoded_response_arrays["action"], response_arrays["action"]), "response action changed")

    left, right = socket.socketpair()
    try:
        left.settimeout(1.0)
        right.settimeout(1.0)
        send_frame(left, config, request_payload)
        framed_payload = recv_frame(right, config)
        require(framed_payload == request_payload, "socket frame round trip changed payload")
    finally:
        left.close()
        right.close()

    corrupted = bytearray(request_payload)
    corrupted[-1] ^= 0x01
    _expect_protocol_error("CRC32CorruptionRejection", lambda: unpack_message(config, corrupted))

    wrong_shape = dict(request_arrays)
    wrong_shape["robot0_eef_pos"] = np.zeros((4,), dtype=np.float32)
    _expect_protocol_error(
        "ShapeMismatchRejection",
        lambda: pack_message(config, config["request"]["message_type"], request_metadata, wrong_shape),
    )

    wrong_dtype = dict(request_arrays)
    wrong_dtype["robot0_eef_pos"] = request_arrays["robot0_eef_pos"].astype(np.float64)
    _expect_protocol_error(
        "DTypeMismatchRejection",
        lambda: pack_message(config, config["request"]["message_type"], request_metadata, wrong_dtype),
    )

    nonfinite = {"action": response_arrays["action"].copy()}
    nonfinite["action"][0] = np.nan
    _expect_protocol_error(
        "NonfiniteActionRejection",
        lambda: pack_message(config, config["response"]["message_type"], response_metadata, nonfinite),
    )

    excessive = {"action": response_arrays["action"].copy()}
    excessive["action"][0] = np.float32(1.01)
    _expect_protocol_error(
        "ActionLimitRejection",
        lambda: pack_message(config, config["response"]["message_type"], response_metadata, excessive),
    )

    negative_sequence = dict(request_metadata)
    negative_sequence["sequence_id"] = -1
    _expect_protocol_error(
        "NegativeSequenceRejection",
        lambda: pack_message(config, config["request"]["message_type"], negative_sequence, request_arrays),
    )

    error_metadata = dict(response_metadata)
    error_metadata["status"] = "error"
    error_metadata["error_message"] = "self-test error"
    error_payload = pack_message(
        config,
        config["response"]["message_type"],
        error_metadata,
        {"action": np.zeros((7,), dtype=np.float32)},
    )
    _, decoded_error_metadata, decoded_error_arrays = unpack_message(config, error_payload)
    require(decoded_error_metadata["status"] == "error", "error status changed")
    require(not decoded_error_arrays["action"].any(), "error action is not zero")

    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    print("ProtocolConfigSHA256=", config_sha)
    print("RequestPayloadBytes=", len(request_payload))
    print("RequestPayloadSHA256=", hashlib.sha256(request_payload).hexdigest())
    print("ResponsePayloadBytes=", len(response_payload))
    print("ResponsePayloadSHA256=", hashlib.sha256(response_payload).hexdigest())
    print("ErrorPayloadBytes=", len(error_payload))
    print("SocketFrameRoundTrip=PASS")
    print("RequestRoundTripExact=PASS")
    print("ResponseRoundTripExact=PASS")
    print("ErrorResponseRoundTrip=PASS")
    print("Stage08ProtocolSelfTest=PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("no action selected; pass --self-test")
    run_self_test(args.config)


if __name__ == "__main__":
    main()
