"""Convert raw generator output (trance/render.py) into the format used for
training: <dataroot>/data.jsonl + <dataroot>/image/ (as in the released
TRANCE), optionally followed by scripts/preprocess.py to build data.h5.

Generator output has one JSON per sample in <gen_dir>/info/ with all stages and
extra bookkeeping. Samples generated with n_per_trans > 1 share a `trans_id`.
Here every sample becomes one line with a unique `idx`, and splits are made per
transformation, so that samples of one transformation never end up in
different splits.

Several generator directories can be given, e.g. the parts of a run split
over parallel processes (disjoint index ranges, `--start`/`--n_sample`); they
are merged into one dataset.

Example:
    python scripts/convert_generated.py /data/trance_gen/standard \
        /data/trance_multi --preprocess
    python scripts/convert_generated.py /data/gen/parts/*/center_1step \
        /data/trance_center_1step --split-ratio 125000 2500 5000 --preprocess
"""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import jsonlines

sys.path.append(str(Path(__file__).resolve().parent))
from preprocess import copy_resources, preprocess  # noqa: E402

OBJECT_KEYS = ["color", "material", "shape", "size", "position"]


def convert_object(obj):
    obj = {k: obj[k] for k in OBJECT_KEYS}
    obj["position"] = list(obj["position"])
    return obj


def convert_state(state):
    return {
        "objects": [convert_object(obj) for obj in state["objects"]],
        "images": dict(state["images"]),
    }


def convert_trans(trans):
    if trans["attr"] == "position":
        val = list(trans["target"])
        options = [list(option) for option in trans["options"]]
    else:
        val = trans["target"]
        options = [val]
    result = {
        "obj_idx": trans["obj_idx"],
        "attr": trans["attr"],
        "val": val,
        "options": options,
    }
    if trans["attr"] == "position":
        result["type"] = trans["type"]
    # Identity of the object across samples of one transformation.
    if "obj_key" in trans:
        result["obj_key"] = trans["obj_key"]
    return result


def convert_sample(info, idx):
    """One generator sample -> one line of data.jsonl.

    Only the initial and the final state are kept, like in the released data.
    """
    return {
        "idx": idx,
        "trans_id": info.get("trans_id", info["idx"]),
        "member": info.get("member", 0),
        "states": [
            convert_state(info["states"][0]),
            convert_state(info["states"][-1]),
        ],
        "transformations": [
            convert_trans(trans) for trans in info["transformations"]
        ],
    }


def load_groups(gen_dirs):
    """Samples of the generator directories, grouped by transformation.

    Returns (groups, sources): groups maps trans_id to its samples sorted by
    member, sources maps trans_id to the directory it was read from.
    """
    groups = defaultdict(list)
    sources = {}
    seen = set()
    for gen_dir in gen_dirs:
        for path in sorted(Path(gen_dir, "info").glob("*.json")):
            with open(path, "r") as f:
                info = json.load(f)
            trans_id = info.get("trans_id", info["idx"])
            member = info.get("member", 0)
            # Overlapping index ranges, or a directory given twice, would
            # duplicate samples. Images are read from one directory per
            # transformation, so it must not be split over directories.
            if (trans_id, member) in seen:
                raise ValueError(
                    f"sample {trans_id}.{member} found twice ({gen_dir})"
                )
            if sources.setdefault(trans_id, gen_dir) != gen_dir:
                raise ValueError(
                    f"transformation {trans_id} is in both "
                    f"{sources[trans_id]} and {gen_dir}"
                )
            seen.add((trans_id, member))
            groups[trans_id].append(info)
    for members in groups.values():
        members.sort(key=lambda info: info.get("member", 0))
    return groups, sources


def split_transformations(trans_ids, ratios):
    """Consecutive split of the transformations by the given ratios.

    Every split with a positive ratio gets at least one transformation when
    there are enough of them; the rounding remainder goes to the first split.
    """
    n = len(trans_ids)
    counts = [int(n * r / sum(ratios)) for r in ratios]
    for i, r in enumerate(ratios):
        if r > 0 and counts[i] == 0 and n >= len(ratios):
            counts[i] = 1
    counts[0] += n - sum(counts)
    assert counts[0] >= 0, "too few transformations for the split"
    result = []
    start = 0
    for count in counts:
        result.append(trans_ids[start : start + count])
        start += count
    return result


def convert(gen_dirs, dataroot, ratios, split_names, force=False):
    gen_dirs = [Path(gen_dir) for gen_dir in gen_dirs]
    dataroot = Path(dataroot)
    data_path = dataroot / "data.jsonl"
    image_dir = dataroot / "image"
    if data_path.exists() and not force:
        raise FileExistsError(f"{data_path} exists, use --force.")
    image_dir.mkdir(parents=True, exist_ok=True)

    groups, sources = load_groups(gen_dirs)
    sizes = sorted(set(len(members) for members in groups.values()))
    print(f"Transformations: {len(groups)}, samples per one: {sizes}")

    splits = split_transformations(sorted(groups), ratios)
    samples = []
    n_samples = []
    for name, trans_ids in zip(split_names, splits):
        n_before = len(samples)
        for trans_id in trans_ids:
            for info in groups[trans_id]:
                samples.append(convert_sample(info, len(samples)))
        n_samples.append(len(samples) - n_before)
        print(
            f"- {name}: {len(trans_ids)} transformations, "
            f"{n_samples[-1]} samples"
        )

    for sample in samples:
        gen_dir = sources[sample["trans_id"]]
        for state in (sample["states"][0], sample["states"][-1]):
            for image in state["images"].values():
                src = gen_dir / "image" / image
                if not src.is_file():
                    raise FileNotFoundError(f"{src} (render it first)")
                shutil.copyfile(src, image_dir / image)

    with jsonlines.open(data_path, "w") as f:
        f.write_all(samples)
    with open(dataroot / "split.json", "w") as f:
        json.dump(
            {
                "split_name": split_names,
                "n_samples": n_samples,
                "n_transformations": [len(s) for s in splits],
            },
            f,
            indent=2,
        )
    return n_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "gen_dirs", nargs="+", help="one or more <output_data_dir>/<problem>"
    )
    parser.add_argument("dataroot")
    parser.add_argument(
        "--split-name", nargs="+", default=["train", "val", "test"]
    )
    parser.add_argument(
        "--split-ratio", nargs="+", type=float, default=[0.8, 0.1, 0.1]
    )
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    assert len(args.split_name) == len(args.split_ratio)

    n_samples = convert(
        args.gen_dirs,
        args.dataroot,
        args.split_ratio,
        args.split_name,
        args.force,
    )
    if args.preprocess:
        copy_resources(args.dataroot)
        preprocess(
            args.dataroot,
            args.force,
            n_samples,
            args.split_name,
            args.width,
            args.height,
        )
