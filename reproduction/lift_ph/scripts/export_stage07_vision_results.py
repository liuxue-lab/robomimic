#!/usr/bin/env python3
"""Export accepted Stage 7 records using the Python standard library only."""
import argparse
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path

RUN = "core_bc_lift_ph_image_no_video_20260916-155536_ebb628cb_workers0"
LAUNCH = Path("runs/lift_ph/vision_full_train") / RUN
EVAL = Path("runs/lift_ph/evaluation/vision_bc_epoch80_seed20260915_n100_eg6v6_dh")
OFFICIAL = Path("runs/lift_ph/vision_config_audit/generated_DrZ9PC/configs/core/lift/ph/image/bc.json")
OFFICIAL_SHA = "01928b478a329021a631fae6569c659a8fc72c9cb8f514955900babbe4ea86c1"
RUNTIME_SHA = "0f66dd225bf4d56c58a35b54f692e2db0d67b76ad8915f55abf47bc9fd0b6f31"
SELECTED_SHA = "4a2ebaa236f2621d0ac8ecc54e625f82ce2a374e2b908d330db92e0312314cec"
FINAL_SHA = "66317268b0909c02f8214d2be53a7d34e5bbce25d4e9dc116f0e42559ab9127e"
SOURCE_SHA = "2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540"
IMAGE_SHA = "1708dbdd6087073a3665444bb4a4dd65e60564d632e2c712e109b2a0ced81632"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def differences(before, after, prefix=""):
    if isinstance(before, dict) and isinstance(after, dict):
        require(set(before) == set(after), "配置字段集合发生额外变化：" + prefix)
        result = []
        for key in sorted(before):
            result.extend(differences(before[key], after[key], f"{prefix}.{key}".lstrip(".")))
        return result
    return [] if before == after else [{"field": prefix, "before": before, "after": after}]


def csv_bytes(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def build_outputs(repo):
    """Read source evidence and return five deterministic archive file payloads."""
    provenance = {}

    def read(relative, as_json=True):
        path = repo / relative
        raw = path.read_bytes()
        require(bool(raw), f"空文件：{path}")
        provenance[str(relative)] = {"sha256": sha256(raw), "bytes": len(raw)}
        return json.loads(raw) if as_json else raw

    official_bytes = read(OFFICIAL, False)
    runtime_bytes = read(LAUNCH / "config.snapshot.json", False)
    require(sha256(official_bytes) == OFFICIAL_SHA, "官方配置哈希不一致")
    require(sha256(runtime_bytes) == RUNTIME_SHA, "成功运行配置哈希不一致")
    official, runtime = json.loads(official_bytes), json.loads(runtime_bytes)
    diff = differences(official, runtime)
    require({row["field"] for row in diff} == {
        "experiment.name", "experiment.render_video", "train.data", "train.num_data_workers"
    }, "官方配置与运行配置的差异字段不符合已验收记录")
    require(runtime["train"]["num_data_workers"] == 0, "workers 应为 0")
    require(runtime["train"]["num_epochs"] == 600, "训练 epoch 数不一致")
    require(runtime["train"]["seed"] == 1, "训练 seed 不一致")
    require(runtime["experiment"]["epoch_every_n_steps"] == 500, "每 epoch 更新次数不一致")
    require(runtime["experiment"]["name"] == RUN, "实验名不一致")

    training_execution = read(LAUNCH / "execution.json")
    require(training_execution["status"] == "COMPLETED", "正式训练未完成")
    audit = read(LAUNCH / "audit_73_crdba/training_audit.json")
    summary = audit.get("summary", audit)
    require(summary["status"] == "PASS", "训练审计未通过")
    for key in ("train_epochs", "validation_epochs", "completed_epochs"):
        require(summary[key] == 600, key + " 不等于 600")
    require(summary["updates_from_epoch_logs"] == 300000, "训练更新总数不一致")
    require(summary["config_sha256"] == RUNTIME_SHA, "审计中的配置哈希不一致")
    require(summary["selected_epoch"] == 80, "选定 epoch 不一致")
    require(summary["selected_checkpoint_sha256"] == SELECTED_SHA, "选定 checkpoint 记录不一致")
    require(summary["final_checkpoint_sha256"] == FINAL_SHA, "最终 checkpoint 记录不一致")
    require(summary["selection_rule"] == "max_rollout_success_then_earliest_epoch", "选择规则不一致")

    log = read(LAUNCH / "console.log", False).decode("utf-8", errors="replace")
    pattern = re.compile(r"(?m)^Epoch\s+(\d+)\s+Rollouts took[^\n]*\n\s*Env:\s*Lift\s*\n")
    rollout_rows = []
    for match in pattern.finditer(log):
        metrics, _ = json.JSONDecoder().raw_decode(log[match.end():].lstrip())
        epoch = int(match.group(1))
        rate = float(metrics["Success_Rate"])
        horizon = float(metrics["Horizon"])
        mean_return = float(metrics["Return"])
        require(all(math.isfinite(v) for v in (rate, horizon, mean_return)), "rollout 指标非有限")
        require(0 <= rate <= 1 and 1 <= horizon <= 400, "rollout 指标超出范围")
        require(math.isclose(rate * 50, round(rate * 50), abs_tol=1e-8), "成功率与 50 次评估不一致")
        rollout_rows.append({
            "method": "Vision BC", "train_seed": 1, "epoch": epoch,
            "updates": epoch * 500, "evaluation_episodes": 50,
            "success_count": round(rate * 50), "success_rate": rate,
            "mean_horizon": horizon, "mean_return": mean_return,
        })
    require([r["epoch"] for r in rollout_rows] == list(range(20, 601, 20)), "训练期评估应为 30 个有序记录")
    selected = min(rollout_rows, key=lambda r: (-r["success_rate"], r["epoch"]))
    require(selected["epoch"] == 80 and selected["success_count"] == 50, "固定选择规则的结果不一致")
    require(math.isclose(selected["mean_horizon"], 44.06), "选定点 Horizon 不一致")

    dataset = read(Path("runs/lift_ph/vision_data/extract_SklstU/audit_2gy0n38h/dataset_audit.json"))
    require(dataset["status"] == "PASS", "数据审计未通过")
    require(dataset["source_sha256"] == SOURCE_SHA and dataset["image_sha256"] == IMAGE_SHA, "数据审计哈希记录不一致")
    require(dataset["demo_count"] == 200 and dataset["total_steps"] == 9666, "数据规模不一致")
    failed = read(Path("runs/lift_ph/vision_full_train") / RUN.removesuffix("_workers0") / "execution.json")
    require(failed["status"] == "FAILED", "首次运行记录不一致")
    protocol = read(EVAL / "protocol.json")
    execution = read(EVAL / "execution.json")
    require(execution["returncode"] == 0 and execution["checkpoint_unchanged"] is True, "独立评估执行记录未通过")
    video = read(EVAL / "video5_ngahmsp3/evaluation_and_video_report.json")
    require(video["evaluation_status"] == "PASS" and video["visual_inspection"] == "PASS", "评估或视觉验收未通过")
    require(video["video_returncode"] == 0 and video["trajectory_unchanged"] is True, "视频回放未通过")
    require(video["video_source_demos"] == [f"demo_{i}" for i in range(5)], "视觉验收范围不一致")
    require(video["evaluation_seed"] == 20260915 and video["rollout_count"] == 100, "独立评估协议不一致")
    require(video["checkpoint_sha256"] == SELECTED_SHA, "独立评估 checkpoint 不一致")
    stats = video["statistics"]
    for key, value in {"Return": 1.0, "Horizon": 46.47, "Success_Rate": 1.0, "Num_Success": 100.0}.items():
        require(math.isclose(float(stats[key]), value), f"独立评估 {key} 不一致")
    episodes = video["episodes"]
    require(len(episodes) == 100 and {r["demo"] for r in episodes} == {f"demo_{i}" for i in range(100)}, "保存的回合列表不一致")
    require(math.isclose(sum(r["horizon"] for r in episodes) / 100, 46.47), "保存的平均 Horizon 不一致")

    independent = [{
        "method": "Vision BC", "algorithm": "BC_GMM", "train_seed": 1,
        "completed_epochs": 600, "completed_updates": 300000,
        "selected_epoch": 80, "selected_checkpoint_updates": 40000,
        "evaluation_seed": 20260915, "evaluation_episodes": 100,
        "horizon_limit": 400, "terminate_on_success": True,
        "success_count": 100, "success_rate": stats["Success_Rate"],
        "mean_horizon_all_episodes": stats["Horizon"], "mean_return": stats["Return"],
        "checkpoint_sha256": SELECTED_SHA, "visual_inspection_episodes": 5,
    }]
    evidence = {
        "schema_version": 1, "stage": 7, "scope": "Vision BC extension to completed stages 0-6",
        "status": "PASS", "git_publication": "Not established by experiment evidence",
        "verification_scope": {
            "copied_configuration_bytes": "SHA256 checked during export",
            "dataset_and_checkpoint_hashes": "Recorded values from prior accepted audits; binaries not rehashed during export",
            "training_rollouts": "Parsed from existing console.log; no new training or evaluation",
            "visual_inspection": "User confirmation of saved evaluation demos 0 through 4 only",
        },
        "configuration_differences": diff,
        "dataset_audit": dataset,
        "failed_workers2_execution": failed,
        "successful_workers0_execution": training_execution,
        "training_audit_summary_as_recorded": summary,
        "independent_evaluation_protocol": protocol,
        "independent_evaluation_execution": execution,
        "evaluation_and_video_report": video,
        "historical_record_note": "Any PENDING independent_evaluation field in the earlier training audit is preserved as historical; the later evaluation/video report records completion.",
        "source_files": provenance,
    }
    return {
        "configs/vision_bc_official.json": official_bytes,
        "configs/vision_bc_runtime_workers0.json": runtime_bytes,
        "results/stage07-training-rollouts.csv": csv_bytes(rollout_rows),
        "results/stage07-independent-evaluation.csv": csv_bytes(independent),
        "results/stage07-vision-bc-evidence.json": json_bytes(evidence),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/lx/robomimic"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    output = output.resolve()
    target = (repo / "reproduction/lift_ph").resolve()
    require(output != target and target not in output.parents, "预览请写到 runs/ 下的独立目录")
    outputs = build_outputs(repo)
    for name, content in outputs.items():
        dest = output / name
        require(not dest.exists() or dest.read_bytes() == content, f"已有不同内容：{dest}")
    for name, content in outputs.items():
        dest = output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            with dest.open("xb") as stream:
                stream.write(content)
        print("Exported=", dest)
    print("VisionResultsExport=PASS")


if __name__ == "__main__":
    main()
