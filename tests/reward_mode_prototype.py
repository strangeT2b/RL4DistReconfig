"""Prototype reward modes for review before touching RL/reward.py.

Run from repo root:
  python tests/reward_mode_prototype.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from RL.reward import (
    compute_reward_iou_only,
    compute_reward_valid_and_iou,
    compute_reward_valid_only,
)


def main() -> None:
    prompt = "Power Distribution Network: Busses=4, Lines=[(1, 2), (2, 3), (3, 1), (3, 4)]"
    valid_response = "Output: Open Lines=[(3, 1)], Node Voltages=[1.0], System Loss=0.12"
    invalid_response = "Output: Open Lines=[(9, 10)], Node Voltages=[1.0], System Loss=0.12"
    gt_response = "Output: Open Lines=[(9, 10)], Node Voltages=[1.0], System Loss=0.12"
    valid_gt_response = "Output: Open Lines=[(3, 1)], Node Voltages=[1.0], System Loss=0.12"

    cases = [
        ("valid_only / valid", compute_reward_valid_only(prompt, valid_response)),
        ("valid_only / invalid", compute_reward_valid_only(prompt, invalid_response)),
        ("iou_only / invalid_but_gt_match", compute_reward_iou_only(prompt, invalid_response, gt_response)),
        (
            "valid_and_iou / valid_gt_match",
            compute_reward_valid_and_iou(prompt, valid_response, valid_gt_response),
        ),
        (
            "valid_and_iou / invalid_gt_match",
            compute_reward_valid_and_iou(prompt, invalid_response, gt_response),
        ),
    ]
    for name, (reward, parts) in cases:
        print(name)
        print("  reward:", reward)
        print("  parts:", parts)

    assert compute_reward_valid_only(prompt, valid_response)[0] == 1.0
    assert compute_reward_valid_only(prompt, invalid_response)[0] < 0.0
    assert compute_reward_iou_only(prompt, invalid_response, gt_response)[0] == 10.0
    assert compute_reward_valid_and_iou(prompt, valid_response, valid_gt_response)[0] == 11.0
    assert compute_reward_valid_and_iou(prompt, invalid_response, gt_response)[0] < 0.0


if __name__ == "__main__":
    main()
