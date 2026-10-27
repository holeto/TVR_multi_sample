"""Transformations of generated TRANCE data, as the grammar sees them.

A group is one transformation: its program [(descriptor, action)] and, per
member, the init_desc the decoder gets (stage-0 objects with size, color,
material, shape and grid position; everything else is stripped).
"""
import json
import os
from collections import defaultdict

from .executor import DESC_ATTRS

INIT_DESC_KEYS = DESC_ATTRS + ["position"]


def init_desc(info):
    """Stage-0 objects of a generator info file, stripped."""
    return [
        {
            k: (list(obj[k]) if k == "position" else obj[k])
            for k in INIT_DESC_KEYS
        }
        for obj in info["states"][0]["objects"]
    ]


def abstract_action(trans):
    """Action of a recorded transformation as the grammar writes it."""
    if trans.get("type") in ("in", "out"):
        return "position." + trans["type"]
    return trans["pair"]


def program(info):
    """Program [(descriptor, action)] of a generator info file."""
    objects = info["states"][0]["objects"]
    return [
        (
            tuple(objects[t["obj_idx"]][attr] for attr in DESC_ATTRS),
            abstract_action(t),
        )
        for t in info["transformations"]
    ]


def load_groups(info_dir):
    """Groups of an info dir: [{trans_id, programs, members}], where
    programs and members are per member, anchor first."""
    by_id = defaultdict(list)
    for name in sorted(os.listdir(info_dir)):
        if name.endswith(".json"):
            with open(os.path.join(info_dir, name)) as f:
                info = json.load(f)
            by_id[info.get("trans_id", info["idx"])].append(info)
    groups = []
    for trans_id, infos in sorted(by_id.items()):
        infos.sort(key=lambda info: info.get("member", 0))
        groups.append(
            {
                "trans_id": trans_id,
                "programs": [program(info) for info in infos],
                "members": [init_desc(info) for info in infos],
            }
        )
    return groups


def _commute(step_a, step_b):
    """Steps commute unless they are of one object and one is its in/out."""
    (desc_a, action_a), (desc_b, action_b) = step_a, step_b
    moves = ("position.in", "position.out")
    return desc_a != desc_b or (action_a not in moves and action_b not in moves)


def canonical_order(program):
    """Canonical order of a program: commuting steps sorted by (descriptor
    text, action text), others in their order (as group.canonical_steps)."""
    remaining = list(range(len(program)))
    result = []
    while remaining:
        ready = [
            i
            for i in remaining
            if all(_commute(program[j], program[i]) for j in remaining if j < i)
        ]
        best = min(
            ready, key=lambda i: ("_".join(program[i][0]), program[i][1], i)
        )
        remaining.remove(best)
        result.append(program[best])
    return result
