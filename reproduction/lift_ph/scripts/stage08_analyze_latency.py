#!/usr/bin/env python3
"""Cross-audit Stage 08 client, server, and Jetson telemetry evidence.

The analyzer treats the paced client report as the evaluation contract, joins
its measured episode to server JSONL records by episode and sequence, and
restricts tegrastats data to the actual inference-response window.  It writes
one detailed JSON artifact and one compact CSV suitable for later repository
export.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np


SCRIPT_VERSION = 1


class Stage08LatencyAnalysisError(RuntimeError):
    """Raised when source evidence is incomplete or internally inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage08LatencyAnalysisError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Stage08LatencyAnalysisError(f"could not read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except Exception as exc:
                raise Stage08LatencyAnalysisError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            require(
                isinstance(value, dict),
                f"JSONL record is not an object at {path}:{line_number}",
            )
            records.append(value)
    return records


def summarize(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    require(array.size > 0, "cannot summarize an empty value sequence")
    require(np.isfinite(array).all(), "summary input contains a non-finite value")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def summarize_ns(records: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    return summarize(float(record[key]) / 1e6 for record in records)


def parse_cpu_mean_percent(line: str) -> float | None:
    match = re.search(r"\bCPU \[([^]]+)\]", line)
    if match is None:
        return None
    values = []
    for field in match.group(1).split(","):
        utilization = re.search(r"(\d+(?:\.\d+)?)%@", field.strip())
        if utilization is not None:
            values.append(float(utilization.group(1)))
    return float(np.mean(values)) if values else None


def parse_tegrastats(
    path: Path,
    timezone_name: str,
) -> list[Dict[str, Any]]:
    timezone = ZoneInfo(timezone_name)
    patterns = {
        "ram_used_mb": r"\bRAM (\d+)/\d+MB",
        "swap_used_mb": r"\bSWAP (\d+)/\d+MB",
        "gr3d_frequency_percent": r"\bGR3D_FREQ (\d+(?:\.\d+)?)%",
        "cpu_temperature_c": r"\bcpu@(\d+(?:\.\d+)?)C",
        "gpu_temperature_c": r"\bgpu@(\d+(?:\.\d+)?)C",
        "junction_temperature_c": r"\btj@(\d+(?:\.\d+)?)C",
        "vdd_in_mw": r"\bVDD_IN (\d+)mW/\d+mW",
        "vdd_cpu_gpu_cv_mw": r"\bVDD_CPU_GPU_CV (\d+)mW/\d+mW",
        "vdd_soc_mw": r"\bVDD_SOC (\d+)mW/\d+mW",
    }

    samples: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                timestamp = datetime.strptime(line[:19], "%m-%d-%Y %H:%M:%S")
            except ValueError as exc:
                raise Stage08LatencyAnalysisError(
                    f"invalid tegrastats timestamp at {path}:{line_number}"
                ) from exc
            timestamp = timestamp.replace(tzinfo=timezone)
            sample: Dict[str, Any] = {
                "timestamp": timestamp.isoformat(),
                "timestamp_ns": int(timestamp.timestamp() * 1e9),
                "line_number": line_number,
            }
            for name, pattern in patterns.items():
                match = re.search(pattern, line)
                if match is not None:
                    sample[name] = float(match.group(1))
            cpu_mean = parse_cpu_mean_percent(line)
            if cpu_mean is not None:
                sample["cpu_mean_utilization_percent"] = cpu_mean
            samples.append(sample)
    return samples


def verify_client_summary(
    report: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    minimum_samples: int,
) -> tuple[str, int]:
    require(report.get("artifact") == "stage08_paced_latency_benchmark", "wrong client artifact type")
    require(report.get("status") == "PASS", "client benchmark status is not PASS")
    require(report.get("stable_20hz") is True, "client strict stable_20hz result is not true")
    require(int(report.get("control_frequency_hz", -1)) == 20, "client control frequency is not 20 Hz")
    require(int(report.get("control_period_ns", -1)) == 50_000_000, "client period is not 50 ms")

    completed_samples = int(report.get("completed_samples", -1))
    completed_warmups = int(report.get("completed_warmup_requests", -1))
    require(completed_samples >= minimum_samples, "too few completed measured samples")
    require(completed_samples == int(report.get("requested_samples", -2)), "client sample count is incomplete")
    require(completed_warmups == int(report.get("requested_warmup_requests", -2)), "client warmup count is incomplete")
    require(int(report.get("timeout_count", -1)) == 0, "client recorded a timeout")
    require(int(report.get("deadline_miss_count", -1)) == 0, "client recorded a deadline miss")
    require(len(records) == completed_samples, "client JSONL count does not match report")

    require(
        all(record.get("record_type") == "latency_sample" for record in records),
        "client JSONL contains an unexpected record type",
    )
    require(
        [int(record["sample_index"]) for record in records] == list(range(completed_samples)),
        "client sample indices are not contiguous",
    )
    require(
        [int(record["network_timing"]["sequence"]) for record in records]
        == list(range(completed_samples)),
        "client network sequence IDs are not contiguous",
    )
    episode_ids = {str(record["network_timing"]["episode_id"]) for record in records}
    require(len(episode_ids) == 1, "client measured samples span multiple episode IDs")
    measured_episode = next(iter(episode_ids))

    scheduled = np.asarray(
        [int(record["scheduled_release_monotonic_ns"]) for record in records],
        dtype=np.int64,
    )
    require(
        np.all(np.diff(scheduled) == 50_000_000),
        "client scheduled release spacing is not exactly 50 ms",
    )
    require(
        sum(bool(record["deadline_miss"]) for record in records) == 0,
        "client JSONL contains a deadline miss",
    )
    return measured_episode, completed_warmups


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_csv_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = ["category", "metric", "unit", "count", "mean", "p50", "p95", "p99", "max"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metric_row(category: str, metric: str, unit: str, summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "category": category,
        "metric": metric,
        "unit": unit,
        "count": summary["count"],
        "mean": summary["mean"],
        "p50": summary["p50"],
        "p95": summary["p95"],
        "p99": summary["p99"],
        "max": summary["max"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-report", type=Path, required=True)
    parser.add_argument("--client-step-log", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--tegrastats-log", type=Path, required=True)
    parser.add_argument("--system-log", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--window-padding-seconds", type=float, default=1.0)
    parser.add_argument("--minimum-samples", type=int, default=1000)
    args = parser.parse_args()

    inputs = {
        "client_report": args.client_report.resolve(),
        "client_step_log": args.client_step_log.resolve(),
        "server_log": args.server_log.resolve(),
        "tegrastats_log": args.tegrastats_log.resolve(),
        "system_log": args.system_log.resolve(),
    }
    output_json = args.output_json.resolve()
    output_csv = args.output_csv.resolve()
    require(args.window_padding_seconds >= 0.0, "window padding must be nonnegative")
    require(args.minimum_samples >= 1000, "minimum samples must be at least 1000")
    for name, path in inputs.items():
        require(path.is_file(), f"missing {name}: {path}")
    require(not output_json.exists(), f"refusing to overwrite: {output_json}")
    require(not output_csv.exists(), f"refusing to overwrite: {output_csv}")

    client_report = read_json(inputs["client_report"])
    client_records = read_jsonl(inputs["client_step_log"])
    measured_episode, warmup_count = verify_client_summary(
        client_report,
        client_records,
        args.minimum_samples,
    )

    server_records = read_jsonl(inputs["server_log"])
    measured_server = [
        record
        for record in server_records
        if record.get("event") == "inference_response"
        and record.get("episode_id") == measured_episode
    ]
    all_server_responses = [
        record for record in server_records if record.get("event") == "inference_response"
    ]
    error_events = {
        "connection_rejected",
        "frame_receive_error",
        "request_decode_error",
        "inference_error_response",
        "error_response_send_failure",
    }
    observed_errors = [record for record in server_records if record.get("event") in error_events]
    connection_records = [record for record in server_records if record.get("event") == "connection_accepted"]
    stop_records = [record for record in server_records if record.get("event") == "server_stop"]

    measured_count = int(client_report["completed_samples"])
    require(len(measured_server) == measured_count, "server measured response count does not match client")
    require(
        [int(record["sequence_id"]) for record in measured_server] == list(range(measured_count)),
        "server measured sequence IDs are not contiguous",
    )
    require(
        all(record.get("status") == "ok" for record in measured_server),
        "server measured records contain a non-ok status",
    )
    require(len(connection_records) == 1, "server did not accept exactly one connection")
    require(not observed_errors, "server log contains a protocol or inference error event")
    require(len(stop_records) == 1, "server log does not contain exactly one stop event")
    expected_total_responses = measured_count + warmup_count
    require(
        len(all_server_responses) == expected_total_responses,
        "server total response count does not equal warmup plus measured requests",
    )
    require(
        int(stop_records[0]["handled_requests"]) == expected_total_responses,
        "server stop request count is inconsistent",
    )

    server_metric_keys = (
        "receive_duration_ns",
        "decode_duration_ns",
        "preprocess_duration_ns",
        "inference_duration_ns",
        "postprocess_duration_ns",
        "encode_duration_ns",
        "send_duration_ns",
        "server_request_duration_ns",
    )
    server_latency = {
        key.removesuffix("_duration_ns"): summarize_ns(measured_server, key)
        for key in server_metric_keys
    }

    first_response_ns = min(int(record["log_time_ns"]) for record in all_server_responses)
    last_response_ns = max(int(record["log_time_ns"]) for record in all_server_responses)
    padding_ns = int(args.window_padding_seconds * 1e9)
    telemetry_window_start_ns = first_response_ns - padding_ns
    telemetry_window_end_ns = last_response_ns + padding_ns

    raw_telemetry = parse_tegrastats(inputs["tegrastats_log"], args.timezone)
    selected_telemetry = [
        sample
        for sample in raw_telemetry
        if telemetry_window_start_ns <= int(sample["timestamp_ns"]) <= telemetry_window_end_ns
    ]
    require(len(raw_telemetry) > 0, "tegrastats log contains no samples")
    require(len(selected_telemetry) >= 45, "too few tegrastats samples in the inference window")

    resource_units = {
        "ram_used_mb": "MB",
        "swap_used_mb": "MB",
        "gr3d_frequency_percent": "percent",
        "cpu_mean_utilization_percent": "percent",
        "cpu_temperature_c": "C",
        "gpu_temperature_c": "C",
        "junction_temperature_c": "C",
        "vdd_in_mw": "mW",
        "vdd_cpu_gpu_cv_mw": "mW",
        "vdd_soc_mw": "mW",
    }
    resource_summary: Dict[str, Dict[str, Any]] = {}
    for metric in resource_units:
        values = [sample[metric] for sample in selected_telemetry if metric in sample]
        require(len(values) == len(selected_telemetry), f"tegrastats metric is missing: {metric}")
        resource_summary[metric] = summarize(values)

    system_text = inputs["system_log"].read_text(encoding="utf-8")
    power_mode_match = re.search(r"NV Power Mode:\s*([^\r\n]+)", system_text)
    require(power_mode_match is not None, "could not parse Jetson power mode")
    power_mode = power_mode_match.group(1).strip()
    require(power_mode == "MAXN_SUPER", f"unexpected Jetson power mode: {power_mode}")

    client_latency_keys = (
        "action_latency",
        "scheduled_action_latency",
        "release_lateness",
        "request_interval",
        "client_serialize",
        "network_round_trip",
        "server_receive_decode_preprocess",
        "server_inference",
        "server_post_inference_before_send",
        "deadline_lateness",
    )
    client_latency: Dict[str, Dict[str, Any]] = {}
    for key in client_latency_keys:
        source = client_report[key]
        client_latency[key] = {
            "count": int(source["count"]),
            "mean": float(source["mean_ms"]),
            "p50": float(source["p50_ms"]),
            "p95": float(source["p95_ms"]),
            "p99": float(source["p99_ms"]),
            "max": float(source["max_ms"]),
        }

    provenance = {
        name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for name, path in inputs.items()
    }
    analysis: Dict[str, Any] = {
        "artifact": "stage08_paced_latency_cross_device_analysis",
        "artifact_version": 1,
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "strict_stable_20hz": True,
        "evaluation": {
            "control_frequency_hz": 20,
            "control_period_ns": 50_000_000,
            "warmup_requests": warmup_count,
            "measured_requests": measured_count,
            "server_total_requests": len(all_server_responses),
            "connections": len(connection_records),
            "timeout_count": int(client_report["timeout_count"]),
            "deadline_miss_count": int(client_report["deadline_miss_count"]),
            "measured_episode_id": measured_episode,
        },
        "client_latency_ms": client_latency,
        "server_stage_latency_ms": server_latency,
        "jetson": {
            "power_mode": power_mode,
            "raw_tegrastats_samples": len(raw_telemetry),
            "selected_tegrastats_samples": len(selected_telemetry),
            "telemetry_timezone": args.timezone,
            "telemetry_padding_seconds": args.window_padding_seconds,
            "window_start_unix_ns": telemetry_window_start_ns,
            "window_end_unix_ns": telemetry_window_end_ns,
            "window_first_sample": selected_telemetry[0]["timestamp"],
            "window_last_sample": selected_telemetry[-1]["timestamp"],
            "resources": resource_summary,
        },
        "validation": {
            "client_status_pass": True,
            "client_server_episode_match": True,
            "client_server_sequence_match": True,
            "server_request_accounting_match": True,
            "server_error_event_count": len(observed_errors),
            "fixed_20hz_schedule": True,
            "zero_timeouts": True,
            "zero_deadline_misses": True,
            "power_mode_maxn_super": True,
        },
        "provenance": provenance,
    }

    rows: list[Dict[str, Any]] = []
    for metric, summary in client_latency.items():
        rows.append(metric_row("client_latency", metric, "ms", summary))
    for metric, summary in server_latency.items():
        rows.append(metric_row("server_stage_latency", metric, "ms", summary))
    for metric, summary in resource_summary.items():
        rows.append(metric_row("jetson_resource", metric, resource_units[metric], summary))

    write_json_exclusive(output_json, analysis)
    write_csv_exclusive(output_csv, rows)

    print("Stage08LatencyCrossDeviceAnalysisBegin=")
    print(f"MeasuredEpisodeID= {measured_episode}")
    print(f"WarmupRequests= {warmup_count}")
    print(f"MeasuredRequests= {measured_count}")
    print(f"ServerTotalRequests= {len(all_server_responses)}")
    print(f"ServerErrorEvents= {len(observed_errors)}")
    print(f"TegrastatsRawSamples= {len(raw_telemetry)}")
    print(f"TegrastatsSelectedSamples= {len(selected_telemetry)}")
    print(f"PowerMode= {power_mode}")
    print(f"ActionLatencyP99Ms= {client_latency['action_latency']['p99']:.3f}")
    print(f"ActionLatencyMaxMs= {client_latency['action_latency']['max']:.3f}")
    print(f"ServerInferenceP99Ms= {server_latency['inference']['p99']:.3f}")
    print(f"GPUUtilizationP95Percent= {resource_summary['gr3d_frequency_percent']['p95']:.3f}")
    print(f"GPUTemperatureMaxC= {resource_summary['gpu_temperature_c']['max']:.3f}")
    print(f"JunctionTemperatureMaxC= {resource_summary['junction_temperature_c']['max']:.3f}")
    print(f"VDDInMeanMW= {resource_summary['vdd_in_mw']['mean']:.3f}")
    print(f"VDDInMaxMW= {resource_summary['vdd_in_mw']['max']:.3f}")
    print(f"RAMUsedMaxMB= {resource_summary['ram_used_mb']['max']:.3f}")
    print(f"OutputJSON= {output_json}")
    print(f"OutputJSONSHA256= {sha256_file(output_json)}")
    print(f"OutputCSV= {output_csv}")
    print(f"OutputCSVSHA256= {sha256_file(output_csv)}")
    print("StrictStable20HzAcceptance=PASS")
    print("Stage08LatencyCrossDeviceAnalysis=PASS")
    print("Stage08LatencyCrossDeviceAnalysisEnd=")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Stage08LatencyAnalysisError={type(exc).__name__}: {exc}")
        raise SystemExit(1)
