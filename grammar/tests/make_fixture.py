"""Write a test fixture from a generator info dir.

    python -m grammar.tests.make_fixture <data_dir>/<problem>/info out.json

Per transformation it keeps the anchor's program, every member's init_desc
(what the decoder gets) and the number of core invisible decoys.
"""
import argparse
import json
import os

from ..data import init_desc, program


def make_fixture(info_dir):
    groups = {}
    for name in sorted(os.listdir(info_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(info_dir, name)) as f:
            info = json.load(f)
        trans_id = info.get("trans_id", info["idx"])
        entry = groups.setdefault(
            trans_id,
            {
                "trans_id": trans_id,
                "programs": [],
                "members": [],
                "n_core_decoys": 0,
            },
        )
        entry["programs"].append([[list(d), a] for d, a in program(info)])
        entry["members"].append(init_desc(info))
        if info.get("member", 0) == 1:
            entry["n_core_decoys"] = sum(
                1 for d in info.get("decoy_objs", []) if d["core"]
            )
    return [groups[k] for k in sorted(groups)]


def load_fixture(path):
    with open(path) as f:
        groups = json.load(f)
    for group in groups:
        group["programs"] = [
            [(tuple(d), a) for d, a in prog] for prog in group["programs"]
        ]
    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("info_dir")
    parser.add_argument("out")
    args = parser.parse_args()
    with open(args.out, "w") as f:
        json.dump(make_fixture(args.info_dir), f)
