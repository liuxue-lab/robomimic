#!/usr/bin/env python3
"""Audit the official robomimic Lift-PH low-dimensional dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


# Official SHA256 for v1.5/lift/ph/low_dim_v15.hdf5.
EXPECTED_SHA256 = "2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540"

# Observation keys selected by robomimic/exps/templates/bc.json.
STATE_BC_OBS_DIMS = {
    "robot0_eef_pos": 3,
    "robot0_eef_quat": 4,
    "robot0_gripper_qpos": 2,
    "object": 10,
}

EXPECTED_MASK_SIZES = {
    "train": 180,
    "valid": 20,
    "20_percent": 40,
    "20_percent_train": 36,
    "20_percent_valid": 4,
    "50_percent": 100,
    "50_percent_train": 90,
    "50_percent_valid": 10,
}


def require(condition: bool, message: str) -> None:
    """Raise a clear error when an audit condition is false."""
    if not condition:
        raise AssertionError(message)


def calculate_sha256(path: Path) -> str:
    """Calculate the file hash in chunks instead of loading it all into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def demo_sort_key(name: str) -> int:
    """Sort demo_2 before demo_10 by using the numeric suffix."""
    return int(name.rsplit("_", maxsplit=1)[1])


def decode_demo_names(dataset: h5py.Dataset) -> list[str]:
    """Convert HDF5 byte strings such as b'demo_0' into Python strings."""
    names = []
    for value in dataset[:]:
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        else:
            names.append(str(value))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args()

    dataset_path = args.dataset.expanduser().resolve()
    require(dataset_path.is_file(), f"Dataset does not exist: {dataset_path}")

    actual_sha256 = calculate_sha256(dataset_path)
    require(
        actual_sha256 == EXPECTED_SHA256,
        f"SHA256 mismatch: expected {EXPECTED_SHA256}, got {actual_sha256}",
    )

    with h5py.File(dataset_path, "r") as hdf5_file:
        # A robomimic dataset should contain demonstrations and split masks.
        require("data" in hdf5_file, "Missing root group: data")
        require("mask" in hdf5_file, "Missing root group: mask")

        data_group = hdf5_file["data"]
        mask_group = hdf5_file["mask"]
        demo_names = sorted(data_group.keys(), key=demo_sort_key)
        all_demos = set(demo_names)

        require(len(demo_names) == 200, f"Expected 200 demos, got {len(demo_names)}")

        lengths = []
        action_min = np.full(7, np.inf)
        action_max = np.full(7, -np.inf)
        reward_values: set[float] = set()
        done_values: set[int] = set()
        reference_obs_keys: set[str] | None = None

        required_demo_keys = {"actions", "dones", "next_obs", "obs", "rewards", "states"}

        for demo_name in demo_names:
            demo = data_group[demo_name]
            missing_demo_keys = required_demo_keys - set(demo.keys())
            require(not missing_demo_keys, f"{demo_name} missing keys: {sorted(missing_demo_keys)}")

            actions = demo["actions"][:]
            rewards = demo["rewards"][:]
            dones = demo["dones"][:]
            states = demo["states"][:]
            transition_count = actions.shape[0]
            lengths.append(transition_count)

            # Every transition-level array must share the same time dimension T.
            require(actions.ndim == 2, f"{demo_name} actions must be 2-D")
            require(actions.shape[1] == 7, f"{demo_name} action dimension is not 7")
            require(rewards.shape == (transition_count,), f"{demo_name} reward shape mismatch")
            require(dones.shape == (transition_count,), f"{demo_name} done shape mismatch")
            require(states.shape[0] == transition_count, f"{demo_name} state length mismatch")

            declared_count = int(demo.attrs.get("num_samples", transition_count))
            require(
                declared_count == transition_count,
                f"{demo_name} num_samples attribute does not match actions",
            )

            # Reject NaN and infinity before these values reach model training.
            for key, array in {
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "states": states,
            }.items():
                require(np.isfinite(array).all(), f"{demo_name}/{key} contains NaN or Inf")

            obs_group = demo["obs"]
            next_obs_group = demo["next_obs"]
            obs_keys = set(obs_group.keys())
            next_obs_keys = set(next_obs_group.keys())
            require(obs_keys == next_obs_keys, f"{demo_name} obs and next_obs keys differ")

            if reference_obs_keys is None:
                reference_obs_keys = obs_keys
            require(obs_keys == reference_obs_keys, f"{demo_name} observation keys are inconsistent")

            for obs_key in sorted(obs_keys):
                obs = obs_group[obs_key][:]
                next_obs = next_obs_group[obs_key][:]
                require(obs.shape[0] == transition_count, f"{demo_name}/obs/{obs_key} length mismatch")
                require(
                    next_obs.shape[0] == transition_count,
                    f"{demo_name}/next_obs/{obs_key} length mismatch",
                )
                require(obs.shape[1:] == next_obs.shape[1:], f"{demo_name}/{obs_key} feature shape mismatch")
                require(np.isfinite(obs).all(), f"{demo_name}/obs/{obs_key} contains NaN or Inf")
                require(
                    np.isfinite(next_obs).all(),
                    f"{demo_name}/next_obs/{obs_key} contains NaN or Inf",
                )

            # Confirm the exact four fields and dimensions used by State BC.
            for obs_key, expected_dim in STATE_BC_OBS_DIMS.items():
                require(obs_key in obs_group, f"{demo_name} missing State BC key: {obs_key}")
                require(
                    obs_group[obs_key].shape[1:] == (expected_dim,),
                    f"{demo_name}/{obs_key} expected dim {expected_dim}, "
                    f"got {obs_group[obs_key].shape[1:]}",
                )

            # The saved policy actions are normalized controller commands.
            require(
                np.all(actions >= -1.0 - 1e-6) and np.all(actions <= 1.0 + 1e-6),
                f"{demo_name} contains actions outside [-1, 1]",
            )

            action_min = np.minimum(action_min, actions.min(axis=0))
            action_max = np.maximum(action_max, actions.max(axis=0))
            reward_values.update(float(value) for value in np.unique(rewards))
            done_values.update(int(value) for value in np.unique(dones))

        total_transitions = int(sum(lengths))
        declared_total = int(data_group.attrs.get("total", total_transitions))
        require(declared_total == total_transitions, "data.total does not match summed demo lengths")
        require(total_transitions == 9666, f"Expected 9666 transitions, got {total_transitions}")

        # Verify that the embedded metadata can reconstruct the correct environment.
        require("env_args" in data_group.attrs, "Missing data.env_args metadata")
        env_args = json.loads(data_group.attrs["env_args"])
        require(env_args["env_name"] == "Lift", "Environment is not Lift")
        require(env_args["env_version"] == "1.5.1", "Environment version is not 1.5.1")
        require(env_args["env_kwargs"]["robots"] == ["Panda"], "Robot is not Panda")
        require(env_args["env_kwargs"]["use_camera_obs"] is False, "Dataset unexpectedly uses camera obs")
        require(env_args["env_kwargs"]["control_freq"] == 20, "Control frequency is not 20 Hz")

        # Decode masks and ensure every referenced demo actually exists.
        mask_sets: dict[str, set[str]] = {}
        for mask_name in mask_group.keys():
            mask_demo_names = decode_demo_names(mask_group[mask_name])
            require(
                len(mask_demo_names) == len(set(mask_demo_names)),
                f"Mask {mask_name} contains duplicate demo names",
            )
            mask_sets[mask_name] = set(mask_demo_names)
            unknown_demos = mask_sets[mask_name] - all_demos
            require(not unknown_demos, f"Mask {mask_name} references unknown demos: {sorted(unknown_demos)}")

        for mask_name, expected_size in EXPECTED_MASK_SIZES.items():
            require(mask_name in mask_sets, f"Missing mask: {mask_name}")
            require(
                len(mask_sets[mask_name]) == expected_size,
                f"Mask {mask_name} expected {expected_size} demos, got {len(mask_sets[mask_name])}",
            )

        # Full split: no trajectory may appear in both train and validation.
        train_demos = mask_sets["train"]
        valid_demos = mask_sets["valid"]
        require(train_demos.isdisjoint(valid_demos), "train and valid masks overlap")
        require(train_demos | valid_demos == all_demos, "train and valid do not cover all demos")

        # Reduced-data splits must also be internally disjoint and complete.
        for prefix in ("20_percent", "50_percent"):
            subset = mask_sets[prefix]
            subset_train = mask_sets[f"{prefix}_train"]
            subset_valid = mask_sets[f"{prefix}_valid"]
            require(subset_train.isdisjoint(subset_valid), f"{prefix} train and valid overlap")
            require(
                subset_train | subset_valid == subset,
                f"{prefix} train and valid do not reconstruct the subset",
            )

        state_bc_input_dim = sum(STATE_BC_OBS_DIMS.values())

        print("DatasetPath =", dataset_path)
        print("SHA256 =", actual_sha256)
        print("RootKeys =", sorted(hdf5_file.keys()))
        print("TrajectoryCount =", len(demo_names))
        print("TransitionCount =", total_transitions)
        print("TrajectoryLengthMin =", min(lengths))
        print("TrajectoryLengthMean =", float(np.mean(lengths)))
        print("TrajectoryLengthMax =", max(lengths))
        print("ObservationKeys =", sorted(reference_obs_keys or set()))
        print("StateBCInputDim =", state_bc_input_dim)
        print("ActionDim =", 7)
        print("ActionMinPerDim =", np.array2string(action_min, precision=6))
        print("ActionMaxPerDim =", np.array2string(action_max, precision=6))
        print("RewardValues =", sorted(reward_values))
        print("DoneValues =", sorted(done_values))
        print("TrainDemoCount =", len(train_demos))
        print("ValidDemoCount =", len(valid_demos))
        print("TrainValidOverlap =", len(train_demos & valid_demos))
        print("Environment =", env_args["env_name"])
        print("EnvironmentVersion =", env_args["env_version"])
        print("Robot =", env_args["env_kwargs"]["robots"][0])
        print("ControlFrequencyHz =", env_args["env_kwargs"]["control_freq"])
        print("AuditStatus = PASS")


if __name__ == "__main__":
    main()
