#!/usr/bin/env python3
"""Describe actions, terminal signals, and split sizes in Lift-PH low-dim data.

This script complements ``audit_lift_ph_dataset.py``:

* the audit script checks whether required invariants hold and fails on errors;
* this script reports distributions that help us understand the demonstrations.

The dataset is opened read-only. No HDF5 content is changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


# robosuite's 7-D OSC_POSE action convention for this dataset.
ACTION_NAMES = (
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_rx",
    "delta_ry",
    "delta_rz",
    "gripper",
)


def demo_sort_key(name: str) -> int:
    """Sort demo_2 before demo_10 by comparing their numeric suffixes."""
    return int(name.rsplit("_", maxsplit=1)[1])


def decode_demo_names(dataset: h5py.Dataset) -> list[str]:
    """Decode byte-string demo names stored in an HDF5 mask dataset."""
    names: list[str] = []
    for value in dataset[:]:
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        else:
            names.append(str(value))
    return names


def count_mask_transitions(
    data_group: h5py.Group,
    demo_names: list[str],
) -> int:
    """Sum transition counts for every complete trajectory in one mask."""
    return sum(int(data_group[name]["actions"].shape[0]) for name in demo_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to low_dim_v15.hdf5",
    )
    args = parser.parse_args()

    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

    # "r" means read-only: this analysis cannot accidentally modify the dataset.
    with h5py.File(dataset_path, "r") as hdf5_file:
        data_group = hdf5_file["data"]
        mask_group = hdf5_file["mask"]
        demo_names = sorted(data_group.keys(), key=demo_sort_key)

        # Each row is one 7-D expert action a_t. Concatenating trajectories here
        # is appropriate for marginal signal statistics, but not for train/valid
        # splitting (splits must remain trajectory-level).
        all_actions = np.concatenate(
            [data_group[name]["actions"][:] for name in demo_names],
            axis=0,
        )

        reward_one_count = 0
        done_one_count = 0
        demos_with_reward_one = 0
        demos_with_done_one = 0
        demos_ending_with_reward_one = 0
        demos_ending_with_done_one = 0
        nonterminal_done_count = 0
        reward_without_done_count = 0
        done_without_reward_count = 0

        for demo_name in demo_names:
            rewards = data_group[demo_name]["rewards"][:]
            dones = data_group[demo_name]["dones"][:]
            reward_is_one = np.isclose(rewards, 1.0)
            done_is_one = dones == 1

            reward_one_count += int(reward_is_one.sum())
            done_one_count += int(done_is_one.sum())
            demos_with_reward_one += int(reward_is_one.any())
            demos_with_done_one += int(done_is_one.any())
            demos_ending_with_reward_one += int(reward_is_one[-1])
            demos_ending_with_done_one += int(done_is_one[-1])

            # A done flag before the final stored row is called nonterminal here.
            # It is not necessarily corrupt: this dataset can retain later rows
            # after the Lift success condition first becomes true.
            nonterminal_done_count += int(done_is_one[:-1].sum())
            reward_without_done_count += int((reward_is_one & ~done_is_one).sum())
            done_without_reward_count += int((done_is_one & ~reward_is_one).sum())

        print("SignalStatisticsErrorFound=False")
        print("ActionStatisticsBegin")

        for index, action_name in enumerate(ACTION_NAMES):
            values = all_actions[:, index]
            print(
                f"Action[{index}]={action_name} "
                f"min={values.min():.6f} "
                f"p01={np.percentile(values, 1):.6f} "
                f"mean={values.mean():.6f} "
                f"std={values.std():.6f} "
                f"median={np.median(values):.6f} "
                f"p99={np.percentile(values, 99):.6f} "
                f"max={values.max():.6f} "
                f"at_minus_one={np.count_nonzero(np.isclose(values, -1.0))} "
                f"at_plus_one={np.count_nonzero(np.isclose(values, 1.0))}"
            )

        print("ActionStatisticsEnd")

        gripper_values, gripper_counts = np.unique(
            all_actions[:, 6],
            return_counts=True,
        )
        print("GripperUniqueValueCount =", len(gripper_values))
        for value, count in zip(gripper_values, gripper_counts, strict=True):
            print(f"GripperValue={value:.6f} Count={count}")

        print("RewardOneTransitionCount =", reward_one_count)
        print("DoneOneTransitionCount =", done_one_count)
        print("DemosWithRewardOne =", demos_with_reward_one)
        print("DemosWithDoneOne =", demos_with_done_one)
        print("DemosEndingWithRewardOne =", demos_ending_with_reward_one)
        print("DemosEndingWithDoneOne =", demos_ending_with_done_one)
        print("NonterminalDoneCount =", nonterminal_done_count)
        print("RewardWithoutDoneCount =", reward_without_done_count)
        print("DoneWithoutRewardCount =", done_without_reward_count)

        for mask_name in sorted(mask_group.keys()):
            mask_demo_names = decode_demo_names(mask_group[mask_name])
            transition_count = count_mask_transitions(data_group, mask_demo_names)
            print(
                f"Mask={mask_name} "
                f"DemoCount={len(mask_demo_names)} "
                f"TransitionCount={transition_count}"
            )

        print("SignalStatisticsStatus = PASS")


if __name__ == "__main__":
    main()
