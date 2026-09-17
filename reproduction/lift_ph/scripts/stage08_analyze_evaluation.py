#!/usr/bin/env python3
"""Analyze paired Stage 7 local and Stage 8 Jetson HIL Lift evaluations.

The analysis is intentionally read-only with respect to its evidence inputs. It
compares every saved initial simulator state, derives outcome and horizon
statistics, and writes lightweight JSON and per-episode CSV artifacts without
re-running the policy or simulator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

if TYPE_CHECKING:
    import h5py
    import numpy as np


SCRIPT_VERSION = 1
Z_95 = 1.959963984540054


class AnalysisError(RuntimeError):
    """Raised when evidence is incomplete or internally inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def decode_hdf5_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_json_like(value: Any, *, missing: Any) -> Any:
    if value is None:
        return missing
    value = decode_hdf5_value(value)
    if value == "":
        return missing
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return missing if parsed is None else parsed
    return value


def normalized_attribute(group: h5py.Group, name: str, *, missing: Any) -> Any:
    if name not in group.attrs:
        return missing
    return normalize_json_like(group.attrs[name], missing=missing)


def normalized_env_args(group: h5py.Group) -> Any:
    return normalized_attribute(group, "env_args", missing={})


def ordered_demo_names(group: h5py.Group) -> list[str]:
    names = list(group.keys())
    try:
        names.sort(key=lambda name: int(name.rsplit("_", 1)[1]))
    except (IndexError, ValueError) as exc:
        raise AnalysisError("trajectory contains a nonstandard demo name") from exc
    expected = [f"demo_{index}" for index in range(len(names))]
    require(names == expected, "trajectory demo indices are not contiguous from zero")
    return names


def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    require(total > 0, "Wilson interval requires a positive sample count")
    require(0 <= successes <= total, "invalid success count")
    probability = successes / total
    denominator = 1.0 + (z * z) / total
    center = (probability + (z * z) / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + (z * z) / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def exact_mcnemar_pvalue(local_only: int, hil_only: int) -> float:
    """Return the two-sided exact McNemar p-value for discordant pairs."""
    require(local_only >= 0 and hil_only >= 0, "negative discordant count")
    discordant = local_only + hil_only
    if discordant == 0:
        return 1.0
    smaller = min(local_only, hil_only)
    lower_tail = sum(
        math.comb(discordant, index) for index in range(smaller + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * lower_tail)


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    require(math.isfinite(result), f"{label} is nonfinite")
    return result


def temporary_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.inprogress{path.suffix}")


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_csv_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    require(bool(rows), "refusing to write an empty episode CSV")
    fieldnames = list(rows[0].keys())
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_self_test() -> None:
    low_100, high_100 = wilson_interval(100, 100)
    low_99, high_99 = wilson_interval(99, 100)
    require(abs(low_100 - 0.9630065017930143) < 1e-12, "100/100 Wilson lower bound changed")
    require(high_100 == 1.0, "100/100 Wilson upper bound changed")
    require(abs(low_99 - 0.9455138038212946) < 1e-12, "99/100 Wilson lower bound changed")
    require(abs(high_99 - 0.9982325679358593) < 1e-12, "99/100 Wilson upper bound changed")
    require(exact_mcnemar_pvalue(1, 0) == 1.0, "one-discordant-pair McNemar result changed")
    require(exact_mcnemar_pvalue(0, 0) == 1.0, "zero-discordant-pair McNemar result changed")
    print("WilsonIntervalSelfTest=PASS")
    print("ExactMcNemarSelfTest=PASS")
    print("Stage08PairedEvaluationAnalyzerSelfTest=PASS")


def analyze(args: argparse.Namespace) -> None:
    global h5py, np
    import h5py
    import numpy as np

    stage07_path = args.stage07_trajectory.resolve()
    stage08_summary_path = args.stage08_summary.resolve()
    stage08_path = args.stage08_trajectory.resolve()
    output_json = args.output_json.resolve()
    output_csv = args.output_csv.resolve()

    for path, label in (
        (stage07_path, "Stage 7 trajectory"),
        (stage08_summary_path, "Stage 8 summary"),
        (stage08_path, "Stage 8 trajectory"),
    ):
        require(path.is_file(), f"{label} does not exist: {path}")

    require(output_json != output_csv, "JSON and CSV outputs resolve to the same path")
    for path in (output_json, output_csv):
        require(not path.exists(), f"refusing to overwrite output: {path}")
        require(not temporary_path(path).exists(), f"stale in-progress output exists: {temporary_path(path)}")
        path.parent.mkdir(parents=True, exist_ok=True)

    stage08_summary = json.loads(stage08_summary_path.read_text(encoding="utf-8"))
    require(stage08_summary.get("execution_status") == "PASS", "Stage 8 execution did not pass")
    require(stage08_summary.get("requested_rollouts") == 100, "Stage 8 did not request 100 rollouts")
    require(stage08_summary.get("completed_rollouts") == 100, "Stage 8 did not complete 100 rollouts")
    require(stage08_summary.get("seed") == 20260915, "unexpected Stage 8 evaluation seed")
    require(stage08_summary.get("horizon_limit") == 400, "unexpected Stage 8 horizon")
    require(stage08_summary.get("terminate_on_success") is True, "Stage 8 did not terminate on success")
    require(len(stage08_summary.get("episodes", [])) == 100, "Stage 8 summary lacks 100 episode records")

    episode_rows: list[dict[str, Any]] = []
    maximum_initial_state_difference = 0.0
    state_match_count = 0
    model_match_count = 0
    ep_meta_match_count = 0
    exact_pair_count = 0

    with h5py.File(stage07_path, "r") as stage07, h5py.File(stage08_path, "r") as stage08:
        require("data" in stage07 and "data" in stage08, "trajectory lacks a data group")
        data07 = stage07["data"]
        data08 = stage08["data"]
        names07 = ordered_demo_names(data07)
        names08 = ordered_demo_names(data08)
        require(len(names07) == 100 and len(names08) == 100, "expected 100 demos in each trajectory")

        env07 = normalized_env_args(data07)
        env08 = normalized_env_args(data08)
        environment_match = env07 == env08
        require(environment_match, "Stage 7 and Stage 8 environment metadata differ")

        for episode_index in range(100):
            demo07 = data07[f"demo_{episode_index}"]
            demo08 = data08[f"demo_{episode_index}"]
            for demo, label in ((demo07, "Stage 7"), (demo08, "Stage 8")):
                for dataset_name in ("states", "actions", "rewards"):
                    require(dataset_name in demo, f"{label} demo {episode_index} lacks {dataset_name}")

            state07 = np.asarray(demo07["states"][0])
            state08 = np.asarray(demo08["states"][0])
            state_same_shape = state07.shape == state08.shape
            state_exact = state_same_shape and np.array_equal(state07, state08)
            state_difference = (
                float(np.max(np.abs(state07 - state08))) if state_same_shape else float("inf")
            )
            require(math.isfinite(state_difference), f"episode {episode_index} initial states are incomparable")
            maximum_initial_state_difference = max(maximum_initial_state_difference, state_difference)

            model07 = normalized_attribute(demo07, "model_file", missing=None)
            model08 = normalized_attribute(demo08, "model_file", missing=None)
            model_exact = model07 == model08
            ep_meta07 = normalized_attribute(demo07, "ep_meta", missing={})
            ep_meta08 = normalized_attribute(demo08, "ep_meta", missing={})
            ep_meta_exact = ep_meta07 == ep_meta08
            exact_pair = state_exact and model_exact and ep_meta_exact

            state_match_count += int(state_exact)
            model_match_count += int(model_exact)
            ep_meta_match_count += int(ep_meta_exact)
            exact_pair_count += int(exact_pair)

            actions07 = np.asarray(demo07["actions"])
            actions08 = np.asarray(demo08["actions"])
            rewards07 = np.asarray(demo07["rewards"])
            rewards08 = np.asarray(demo08["rewards"])
            require(actions07.ndim == 2 and actions07.shape[1] == 7, f"invalid Stage 7 action shape in episode {episode_index}")
            require(actions08.ndim == 2 and actions08.shape[1] == 7, f"invalid Stage 8 action shape in episode {episode_index}")
            require(actions07.shape[0] == rewards07.shape[0], f"Stage 7 horizon mismatch in episode {episode_index}")
            require(actions08.shape[0] == rewards08.shape[0], f"Stage 8 horizon mismatch in episode {episode_index}")
            require(np.isfinite(actions07).all() and np.isfinite(actions08).all(), f"nonfinite action in episode {episode_index}")
            require(np.isfinite(rewards07).all() and np.isfinite(rewards08).all(), f"nonfinite reward in episode {episode_index}")

            horizon07 = int(actions07.shape[0])
            horizon08 = int(actions08.shape[0])
            return07 = finite_float(rewards07.sum(), f"Stage 7 return for episode {episode_index}")
            return08 = finite_float(rewards08.sum(), f"Stage 8 return for episode {episode_index}")
            success07 = return07 > 0.0
            success08 = return08 > 0.0

            summary_episode = stage08_summary["episodes"][episode_index]
            require(summary_episode["episode_index"] == episode_index, "Stage 8 episode ordering changed")
            require(bool(summary_episode["success"]) == success08, f"Stage 8 success mismatch in episode {episode_index}")
            require(int(summary_episode["horizon"]) == horizon08, f"Stage 8 horizon mismatch in episode {episode_index}")
            require(abs(float(summary_episode["return"]) - return08) < 1e-12, f"Stage 8 return mismatch in episode {episode_index}")

            episode_rows.append(
                {
                    "episode_index": episode_index,
                    "initial_state_sha256": sha256_array(state07),
                    "initial_state_exact": state_exact,
                    "model_exact": model_exact,
                    "episode_metadata_semantic_exact": ep_meta_exact,
                    "exact_initial_condition_pair": exact_pair,
                    "stage07_success": success07,
                    "stage08_success": success08,
                    "stage07_return": return07,
                    "stage08_return": return08,
                    "stage07_horizon": horizon07,
                    "stage08_horizon": horizon08,
                    "horizon_difference_stage08_minus_stage07": horizon08 - horizon07,
                    "outcome_pair": (
                        "both_success"
                        if success07 and success08
                        else "stage07_only"
                        if success07
                        else "stage08_only"
                        if success08
                        else "both_failure"
                    ),
                }
            )

        total07 = int(data07.attrs.get("total", sum(row["stage07_horizon"] for row in episode_rows)))
        total08 = int(data08.attrs.get("total", sum(row["stage08_horizon"] for row in episode_rows)))

    require(exact_pair_count == 100, "not all initial conditions are exact semantic pairs")
    require(maximum_initial_state_difference == 0.0, "paired initial states differ numerically")
    require(total07 == sum(row["stage07_horizon"] for row in episode_rows), "Stage 7 total transition count is inconsistent")
    require(total08 == sum(row["stage08_horizon"] for row in episode_rows), "Stage 8 total transition count is inconsistent")
    require(total08 == int(stage08_summary["total_transitions"]), "Stage 8 summary transition count is inconsistent")

    success07 = sum(int(row["stage07_success"]) for row in episode_rows)
    success08 = sum(int(row["stage08_success"]) for row in episode_rows)
    both_success = sum(row["outcome_pair"] == "both_success" for row in episode_rows)
    stage07_only = sum(row["outcome_pair"] == "stage07_only" for row in episode_rows)
    stage08_only = sum(row["outcome_pair"] == "stage08_only" for row in episode_rows)
    both_failure = sum(row["outcome_pair"] == "both_failure" for row in episode_rows)
    failed07 = [row["episode_index"] for row in episode_rows if not row["stage07_success"]]
    failed08 = [row["episode_index"] for row in episode_rows if not row["stage08_success"]]

    horizons07 = np.asarray([row["stage07_horizon"] for row in episode_rows], dtype=np.float64)
    horizons08 = np.asarray([row["stage08_horizon"] for row in episode_rows], dtype=np.float64)
    successful_horizons07 = np.asarray(
        [row["stage07_horizon"] for row in episode_rows if row["stage07_success"]], dtype=np.float64
    )
    successful_horizons08 = np.asarray(
        [row["stage08_horizon"] for row in episode_rows if row["stage08_success"]], dtype=np.float64
    )
    differences = horizons08 - horizons07
    ci07 = wilson_interval(success07, 100)
    ci08 = wilson_interval(success08, 100)
    mcnemar_pvalue = exact_mcnemar_pvalue(stage07_only, stage08_only)

    require(success07 == 100, f"unexpected Stage 7 success count: {success07}")
    require(success08 == int(stage08_summary["success_count"]), "Stage 8 success count is inconsistent")
    strict_all_success = success08 == 100

    analysis = {
        "artifact": "stage08_paired_independent_evaluation_analysis",
        "artifact_version": 1,
        "script_version": SCRIPT_VERSION,
        "analysis_status": "PASS",
        "evaluation_classification": (
            "PASS" if strict_all_success else "COMPLETED_WITH_RECORDED_DEVIATION"
        ),
        "strict_all_success_gate": "PASS" if strict_all_success else "FAIL",
        "inputs": {
            "stage07_trajectory": str(stage07_path),
            "stage07_trajectory_sha256": sha256_file(stage07_path),
            "stage08_summary": str(stage08_summary_path),
            "stage08_summary_sha256": sha256_file(stage08_summary_path),
            "stage08_trajectory": str(stage08_path),
            "stage08_trajectory_sha256": sha256_file(stage08_path),
        },
        "protocol": {
            "evaluation_seed": 20260915,
            "rollouts": 100,
            "horizon_limit": 400,
            "terminate_on_success": True,
            "checkpoint_sha256": stage08_summary.get("checkpoint_sha256"),
        },
        "pairing": {
            "environment_metadata_semantic_match": environment_match,
            "state_exact_match_count": state_match_count,
            "model_exact_match_count": model_match_count,
            "episode_metadata_semantic_match_count": ep_meta_match_count,
            "exact_initial_condition_pair_count": exact_pair_count,
            "maximum_initial_state_absolute_difference": maximum_initial_state_difference,
        },
        "stage07_local": {
            "success_count": success07,
            "failure_count": 100 - success07,
            "success_rate": success07 / 100.0,
            "success_rate_wilson_95": {"lower": ci07[0], "upper": ci07[1]},
            "failed_episode_indices": failed07,
            "total_transitions": total07,
            "mean_horizon_all": float(horizons07.mean()),
            "median_horizon_all": float(np.median(horizons07)),
            "mean_horizon_successful": float(successful_horizons07.mean()),
            "mean_return": float(np.mean([row["stage07_return"] for row in episode_rows])),
        },
        "stage08_jetson_hil": {
            "report_status": stage08_summary.get("status"),
            "execution_status": stage08_summary.get("execution_status"),
            "report_error": stage08_summary.get("error"),
            "success_count": success08,
            "failure_count": 100 - success08,
            "success_rate": success08 / 100.0,
            "success_rate_wilson_95": {"lower": ci08[0], "upper": ci08[1]},
            "failed_episode_indices": failed08,
            "total_transitions": total08,
            "mean_horizon_all": float(horizons08.mean()),
            "median_horizon_all": float(np.median(horizons08)),
            "mean_horizon_successful": float(successful_horizons08.mean()),
            "mean_return": float(np.mean([row["stage08_return"] for row in episode_rows])),
            "action_deadline_miss_count": int(stage08_summary["action_deadline_miss_count"]),
            "control_period_overrun_count": int(stage08_summary["control_period_overrun_count"]),
        },
        "paired_comparison": {
            "both_success_count": both_success,
            "stage07_only_success_count": stage07_only,
            "stage08_only_success_count": stage08_only,
            "both_failure_count": both_failure,
            "success_rate_difference_stage08_minus_stage07": (success08 - success07) / 100.0,
            "exact_mcnemar_two_sided_pvalue": mcnemar_pvalue,
            "mean_horizon_difference_stage08_minus_stage07": float(differences.mean()),
            "median_horizon_difference_stage08_minus_stage07": float(np.median(differences)),
            "minimum_horizon_difference_stage08_minus_stage07": int(differences.min()),
            "maximum_horizon_difference_stage08_minus_stage07": int(differences.max()),
            "equal_horizon_episode_count": int(np.sum(differences == 0)),
        },
        "episodes": episode_rows,
    }

    json_temp = temporary_path(output_json)
    csv_temp = temporary_path(output_csv)
    try:
        write_json_exclusive(json_temp, analysis)
        write_csv_exclusive(csv_temp, episode_rows)
        json_temp.rename(output_json)
        csv_temp.rename(output_csv)
    except Exception:
        for path in (json_temp, csv_temp):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise

    print("Stage08PairedEvaluationAnalysisBegin=")
    print(f"ExactInitialConditionPairs= {exact_pair_count}")
    print(f"MaximumInitialStateAbsDiff= {maximum_initial_state_difference}")
    print(f"Stage07Success= {success07}/100")
    print(f"Stage08Success= {success08}/100")
    print(f"Stage07MeanHorizon= {horizons07.mean():.6f}")
    print(f"Stage08MeanHorizon= {horizons08.mean():.6f}")
    print(f"Stage08SuccessfulMeanHorizon= {successful_horizons08.mean():.6f}")
    print(f"BothSuccessCount= {both_success}")
    print(f"Stage07OnlySuccessCount= {stage07_only}")
    print(f"Stage08OnlySuccessCount= {stage08_only}")
    print(f"BothFailureCount= {both_failure}")
    print(f"ExactMcNemarTwoSidedPValue= {mcnemar_pvalue:.12g}")
    print(f"Stage07Wilson95= [{ci07[0]:.9f}, {ci07[1]:.9f}]")
    print(f"Stage08Wilson95= [{ci08[0]:.9f}, {ci08[1]:.9f}]")
    print(f"FailedStage08Episodes= {failed08}")
    print(f"StrictAllSuccessGate= {'PASS' if strict_all_success else 'FAIL'}")
    print(f"EvaluationClassification= {analysis['evaluation_classification']}")
    print(f"OutputJSON= {output_json}")
    print(f"OutputJSONSHA256= {sha256_file(output_json)}")
    print(f"OutputCSV= {output_csv}")
    print(f"OutputCSVSHA256= {sha256_file(output_csv)}")
    print("Stage08PairedEvaluationAnalysis=PASS")
    print("Stage08PairedEvaluationAnalysisEnd=")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage07-trajectory", type=Path)
    parser.add_argument("--stage08-summary", type=Path)
    parser.add_argument("--stage08-trajectory", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        for name in (
            "stage07_trajectory",
            "stage08_summary",
            "stage08_trajectory",
            "output_json",
            "output_csv",
        ):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required unless --self-test is used")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
    else:
        analyze(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as exc:
        print(f"Stage08PairedEvaluationAnalysisError={type(exc).__name__}: {exc}")
        raise SystemExit(1)
