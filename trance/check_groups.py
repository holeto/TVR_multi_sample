# -*- coding: utf-8 -*-
"""
Check generated samples for multi-sample consistency (plain Python, no bpy).

    python check_groups.py <output_data_dir>/<problem>/info [--n_per_trans N]

For every transformation (samples sharing `trans_id`) it checks that:
- all members have the same sequence of pairs, move types and target keys;
- each step's `obj_key` matches the stage-0 object at `obj_idx`;
- target keys are unique among the stage-0 objects of every member;
- targets are not occluded in the initial cameras at stage 0 and in the final
  cameras at the final stage.
It also prints how target slots (`obj_idx`) are spread over replayed members.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import group  # noqa: E402


def load_infos(info_dir):
    infos = []
    for name in sorted(os.listdir(info_dir)):
        if name.endswith(".json"):
            with open(os.path.join(info_dir, name), "r") as f:
                infos.append(json.load(f))
    return infos


def signature(info):
    return [
        (t["pair"], t.get("type"), t["obj_key"])
        for t in info["transformations"]
    ]


def check_member(info):
    errors = []
    objects = info["states"][0]["objects"]
    for i, trans in enumerate(info["transformations"]):
        keep_size = "size" in trans["obj_desc"]
        key = group.obj_key(objects[trans["obj_idx"]], keep_size)
        if key != trans["obj_key"]:
            errors.append(
                "step {}: obj_key {} but stage-0 object is {}".format(
                    i, trans["obj_key"], key
                )
            )
        unique = group.unique_key_indices(objects, keep_size)
        if trans["obj_idx"] not in unique:
            errors.append("step {}: target key {} not unique".format(i, key))
    initial_cameras = info["states"][0]["images"].keys()
    final_cameras = info["states"][-1]["images"].keys()
    for idx, stage, cam in group.occluded_targets(
        info, initial_cameras, final_cameras
    ):
        errors.append(
            "target {} occluded at stage {} in {}".format(idx, stage, cam)
        )
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("info_dir")
    parser.add_argument("--n_per_trans", type=int, default=None)
    args = parser.parse_args()

    groups = defaultdict(list)
    for info in load_infos(args.info_dir):
        groups[info["trans_id"]].append(info)

    n_errors = 0
    replay_slots = Counter()
    for trans_id, members in sorted(groups.items()):
        members.sort(key=lambda info: info["member"])
        errors = []
        if args.n_per_trans and len(members) != args.n_per_trans:
            errors.append(
                "{} members, expected {}".format(
                    len(members), args.n_per_trans
                )
            )
        anchor_sig = signature(members[0])
        for info in members:
            if signature(info) != anchor_sig:
                errors.append(
                    "member {}: transformation differs from anchor".format(
                        info["member"]
                    )
                )
            for error in check_member(info):
                errors.append("member {}: {}".format(info["member"], error))
            if info["member"] > 0:
                for trans in info["transformations"]:
                    replay_slots[trans["obj_idx"]] += 1
        for error in errors:
            print("[trans {}] {}".format(trans_id, error))
        n_errors += len(errors)

    print(
        "{} transformations, {} samples, {} errors".format(
            len(groups), sum(len(m) for m in groups.values()), n_errors
        )
    )
    print(
        "Target slots in replayed members: {}".format(
            sorted(replay_slots.items())
        )
    )
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
