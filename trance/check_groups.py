# -*- coding: utf-8 -*-
"""
Check generated samples for multi-sample consistency (plain Python, no bpy).

    python check_groups.py <output_data_dir>/<problem>/info [--n_per_trans N]

For every transformation (samples sharing `trans_id`) it checks that:
- all members have the same sequence of pairs, move types and target keys;
- each step's `obj_key` matches the stage-0 object at `obj_idx`;
- target keys are unique among the stage-0 objects of every member;
- targets are not occluded in the initial cameras at stage 0 and in the final
  cameras at the final stage;
- no attribute other than position is edited twice on the same object;
- moves follow the rules of the grammar: nothing happens to an object after it
  is moved out, an object moved in is not moved again, every object is moved
  at most once (unless --allow_repeated_moves) and never back to its start;
- no later move or growth of another object overlaps a candidate cell of an
  in-move;
- the steps are in canonical order;
- shared objects (`shared_objs`) match a non-target object of the anchor;
- core distractors are the same in every replayed member and visible (not
  occluded) at stage 0 in the initial cameras, in the anchor and in each member;
- invisible decoys copy a unique non-target of the anchor, are invisible, and
  their recorded in-move enters the view onto a free cell in their member and
  in the anchor; core decoys and their moves are the same in every replayed
  member.
It also prints how target slots (`obj_idx`) are spread over replayed members,
how many objects they share with the anchor and how many core distractors each
transformation has.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

TRANCE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(TRANCE_DIR)
import group  # noqa: E402

from option import Option  # noqa: E402

with open(os.path.join(TRANCE_DIR, "resource", "properties.json")) as f:
    PROPERTIES = json.load(f)
OPTION = Option(
    os.path.join(TRANCE_DIR, "resource", "properties.json"),
    os.path.join(TRANCE_DIR, "resource", "values.json"),
)


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


def check_shared(info, anchor):
    errors = []
    objects = info["states"][0]["objects"]
    anchor_objects = anchor["states"][0]["objects"]
    anchor_targets = group.target_indices(anchor)
    for shared in info.get("shared_objs", []):
        obj = objects[shared["obj_idx"]]
        anchor_obj = anchor_objects[shared["anchor_idx"]]
        if group.obj_key(obj) != group.obj_key(anchor_obj):
            errors.append(
                "shared object {} is {}, anchor object {} is {}".format(
                    shared["obj_idx"],
                    group.obj_key(obj),
                    shared["anchor_idx"],
                    group.obj_key(anchor_obj),
                )
            )
        if shared["anchor_idx"] in anchor_targets:
            errors.append(
                "shared object {} is a target".format(shared["anchor_idx"])
            )
        if shared.get("core"):
            sides = [
                (info, shared["obj_idx"]),
                (anchor, shared["anchor_idx"]),
            ]
            for member, idx in sides:
                state = member["states"][0]
                name = state["objects"][idx]["name"]
                for cam in sorted(state["images"]):
                    status = state["status"][cam][name]
                    if status != "visible":
                        errors.append(
                            "core distractor {} of member {}: {} in {}".format(
                                idx, member["member"], status, cam
                            )
                        )
    return errors


def in_moves(info, idx):
    return group.in_move_options(
        info["states"][0]["objects"],
        idx,
        OPTION.move_options,
        OPTION.is_position_valid,
        OPTION.is_visible,
        OPTION.original_coordinate,
        OPTION.size,
    )


def check_decoys(info, anchor):
    errors = []
    objects = info["states"][0]["objects"]
    anchor_objects = anchor["states"][0]["objects"]
    targets = group.target_indices(anchor)
    for decoy in info.get("decoy_objs", []):
        idx, anchor_idx = decoy["obj_idx"], decoy["anchor_idx"]
        move = tuple(decoy["move"])
        name = "decoy {}".format(idx)
        if group.obj_key(objects[idx]) != group.obj_key(
            anchor_objects[anchor_idx]
        ):
            errors.append("{} does not copy anchor {}".format(name, anchor_idx))
        if anchor_idx in targets:
            errors.append("{} copies a target".format(name))
        if OPTION.is_visible(*objects[idx]["position"]):
            errors.append("{} is visible".format(name))
        for member, i in [(info, idx), (anchor, anchor_idx)]:
            member_objects = member["states"][0]["objects"]
            if i not in group.unique_key_indices(member_objects):
                errors.append(
                    "{} not unique in member {}".format(name, member["member"])
                )
            if move not in in_moves(member, i):
                errors.append(
                    "{} can not move {} in member {}".format(
                        name, move, member["member"]
                    )
                )
    return errors


def core_decoys(members):
    """(anchor index, move) of the core decoys of each replayed member."""
    return [
        frozenset(
            (decoy["anchor_idx"], tuple(decoy["move"]))
            for decoy in info.get("decoy_objs", [])
            if decoy["core"]
        )
        for info in members[1:]
    ]


def core_distractors(members):
    """Anchor indices of the core distractors of each replayed member."""
    return [
        frozenset(
            shared["anchor_idx"]
            for shared in info.get("shared_objs", [])
            if shared.get("core")
        )
        for info in members[1:]
    ]


def is_canonical(info):
    keep_size = "size" in info["transformations"][0]["obj_desc"]
    plan = group.extract_plan(
        info, lambda x, y: True, ["small", "medium", "large"], keep_size
    )
    return group.canonical_steps(plan) == plan["steps"]


def check_member(info, allow_repeated_moves=False):
    errors = []
    for step, idx, attr in group.repeated_attr_steps(info):
        errors.append(
            "step {}: {} of object {} edited again".format(step, attr, idx)
        )
    for step, idx, reason in group.move_violations(
        info, allow_repeated_moves
    ):
        errors.append("step {}: object {}: {}".format(step, idx, reason))
    for step, idx, idx_in in group.candidate_conflicts(
        info, PROPERTIES["coordinate"], PROPERTIES["size"]
    ):
        errors.append(
            "step {}: object {} takes a candidate of moved-in {}".format(
                step, idx, idx_in
            )
        )
    objects = info["states"][0]["objects"]
    for i, trans in enumerate(info["transformations"]):
        if trans.get("type") == "in" and "candidates" not in trans:
            errors.append("step {}: in-move without candidates".format(i))
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
    parser.add_argument("--allow_repeated_moves", action="store_true")
    args = parser.parse_args()

    groups = defaultdict(list)
    for info in load_infos(args.info_dir):
        groups[info["trans_id"]].append(info)

    n_errors = 0
    replay_slots = Counter()
    n_shared = Counter()
    n_core = Counter()
    n_decoys = Counter()
    n_in_capable = Counter()
    n_canonical = 0
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
            for error in check_member(
                info, args.allow_repeated_moves
            ) + check_shared(info, members[0]) + check_decoys(
                info, members[0]
            ):
                errors.append("member {}: {}".format(info["member"], error))
            if info["member"] > 0:
                for trans in info["transformations"]:
                    replay_slots[trans["obj_idx"]] += 1
                n_shared[len(info.get("shared_objs", []))] += 1
        cores = core_distractors(members)
        if len(set(cores)) > 1:
            errors.append("core distractors differ between members")
        n_core[min(len(core) for core in cores) if cores else 0] += 1
        decoys = core_decoys(members)
        if len(set(decoys)) > 1:
            errors.append("core decoys differ between members")
        n_decoys[min(len(core) for core in decoys) if decoys else 0] += 1
        in_targets = set(
            t["obj_idx"]
            for t in members[0]["transformations"]
            if t.get("type") == "in"
        )
        n_core_decoys = min(len(core) for core in decoys) if decoys else 0
        n_in_capable[(n_core_decoys + len(in_targets), bool(in_targets))] += 1
        if is_canonical(members[0]):
            n_canonical += 1
        else:
            errors.append("steps are not in canonical order")
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
    print(
        "Shared objects in replayed members: {}".format(
            sorted(n_shared.items())
        )
    )
    print(
        "Core distractors per transformation: {}".format(
            sorted(n_core.items())
        )
    )
    print(
        "Core decoys per transformation: {}".format(sorted(n_decoys.items()))
    )
    print(
        "Core decoys + in-targets (count, has in-move): {}".format(
            sorted(n_in_capable.items())
        )
    )
    print(
        "Transformations in canonical order: {}/{}".format(
            n_canonical, len(groups)
        )
    )
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
