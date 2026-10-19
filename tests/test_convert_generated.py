import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))
from convert_generated import (  # noqa: E402
    convert_sample,
    split_transformations,
)


def make_obj(name, position, color="red"):
    return {
        "name": name,
        "shape": "cube",
        "size": "large",
        "material": "metal",
        "color": color,
        "rotation": 1.0,
        "position": position,
        "3d_coords": (0.0, 0.0, 0.0),
    }


def make_info():
    init = [make_obj("Object_0", (0, 0)), make_obj("Object_1", (10, 10))]
    mid = [make_obj("Object_0", (0, 0), "blue"), init[1]]
    fin = [mid[0], make_obj("Object_1", (30, 10))]
    images_init = {"Camera_Center": "img_000007_01-initial.C.png"}
    images_fin = {
        c: "img_000007_01-final.{}.png".format(c[7])
        for c in ["Camera_Left", "Camera_Center", "Camera_Right"]
    }
    return {
        "idx": 7,
        "trans_id": 7,
        "member": 1,
        "states": [
            {"objects": init, "images": images_init, "status": {}},
            {"objects": mid, "images": {}, "status": {}},
            {"objects": fin, "images": images_fin, "status": {}},
        ],
        "transformations": [
            {
                "obj_idx": 0,
                "attr": "color",
                "pair": "color.blue",
                "target": "blue",
                "obj_key": "large_red_metal_cube",
            },
            {
                "obj_idx": 1,
                "attr": "position",
                "pair": "position.behind.2",
                "target": ("behind", 2),
                "type": "out",
                "options": [("behind", 2), ("behind-left", 2)],
                "obj_key": "large_red_metal_cube",
            },
        ],
    }


def test_convert_sample():
    sample = convert_sample(make_info(), 42)
    assert sample["idx"] == 42
    assert (sample["trans_id"], sample["member"]) == (7, 1)
    # Only the initial and final states, with the released object keys.
    assert len(sample["states"]) == 2
    assert sample["states"][0]["objects"][0] == {
        "color": "red",
        "material": "metal",
        "shape": "cube",
        "size": "large",
        "position": [0, 0],
    }
    assert sample["states"][1]["objects"][1]["position"] == [30, 10]
    assert list(sample["states"][1]["images"]) == [
        "Camera_Left",
        "Camera_Center",
        "Camera_Right",
    ]
    color, position = sample["transformations"]
    assert color == {
        "obj_idx": 0,
        "attr": "color",
        "val": "blue",
        "options": ["blue"],
        "obj_key": "large_red_metal_cube",
    }
    assert position["val"] == ["behind", 2]
    assert position["options"] == [["behind", 2], ["behind-left", 2]]
    assert position["type"] == "out"


def test_split_transformations():
    ids = list(range(25))
    train, val, test = split_transformations(ids, [0.8, 0.1, 0.1])
    assert (len(train), len(val), len(test)) == (21, 2, 2)
    assert train + val + test == ids
    # Small sets still get one transformation in each split.
    splits = split_transformations([0, 1, 2], [0.8, 0.1, 0.1])
    assert [len(s) for s in splits] == [1, 1, 1]
