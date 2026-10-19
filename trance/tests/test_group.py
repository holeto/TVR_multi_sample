# -*- coding: utf-8 -*-
"""
Tests of the bpy-free parts of multi-sample generation.

Run from the repository root: python -m pytest trance/tests
"""
import os
import random
import sys

TRANCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(TRANCE_DIR)

import group  # noqa: E402
from option import Option  # noqa: E402

RESOURCE = os.path.join(TRANCE_DIR, "resource")


def get_option():
    return Option(
        os.path.join(RESOURCE, "properties.json"),
        os.path.join(RESOURCE, "values.json"),
    )


def make_obj(name, position, color="red", shape="cube", size="large"):
    return {
        "name": name,
        "shape": shape,
        "size": size,
        "material": "metal",
        "color": color,
        "position": position,
    }


def make_info():
    """Object_0 moves right 1 (inner), Object_1 moves in, then changes size."""
    stages = [
        [make_obj("Object_0", (0, 0)), make_obj("Object_1", (-30, 0), "blue")],
        [make_obj("Object_0", (0, 10)), make_obj("Object_1", (-30, 0), "blue")],
        [
            make_obj("Object_0", (0, 10)),
            make_obj("Object_1", (-10, 0), "blue"),
        ],
        [
            make_obj("Object_0", (0, 10)),
            make_obj("Object_1", (-10, 0), "blue", size="small"),
        ],
    ]
    status_init = {"Object_0": "visible", "Object_1": "invisible"}
    status_fin = {"Object_0": "visible", "Object_1": "visible"}
    states = []
    for i, objects in enumerate(stages):
        status = status_init if i == 0 else status_fin
        states.append(
            {
                "objects": objects,
                "status": {cam: dict(status) for cam in ["C", "L"]},
            }
        )
    transformations = [
        {
            "obj_idx": 0,
            "attr": "position",
            "pair": "position.right.1",
            "type": "inner",
        },
        {
            "obj_idx": 1,
            "attr": "position",
            "pair": "position.behind.2",
            "type": "in",
        },
        {"obj_idx": 1, "attr": "size", "pair": "size.small"},
    ]
    return {"states": states, "transformations": transformations}


def test_obj_key():
    obj = make_obj("Object_0", (0, 0))
    assert group.obj_key(obj) == "large_red_metal_cube"
    assert group.obj_key(obj, keep_size=False) == "red_metal_cube"
    assert group.obj_desc(obj, keep_size=False) == {
        "color": "red",
        "material": "metal",
        "shape": "cube",
    }


def test_unique_key_indices():
    objects = [
        make_obj("Object_0", (0, 0)),
        make_obj("Object_1", (10, 0)),
        make_obj("Object_2", (20, 0), size="small"),
        make_obj("Object_3", (30, 0), color="blue"),
    ]
    assert group.unique_key_indices(objects) == [2, 3]
    assert group.unique_key_indices(objects, keep_size=False) == [3]


def test_extract_plan():
    option = get_option()
    sizes = option.size.keys()
    plan = group.extract_plan(make_info(), option.is_visible, sizes)
    assert plan["steps"] == [
        (0, "position.right.1", "inner"),
        (1, "position.behind.2", "in"),
        (1, "size.small", None),
    ]
    role_0, role_1 = plan["roles"]
    assert role_0["key"] == "large_red_metal_cube"
    assert role_0["sizes"] == ["large"]
    assert role_0["constraints"] == [
        (0, 0, True),
        (0, 10, True),
        (0, 10, True),
        (0, 10, True),
    ]
    assert role_1["constraints"] == [
        (0, 0, False),
        (0, 0, False),
        (20, 0, True),
        (20, 0, True),
    ]

    plan = group.extract_plan(
        make_info(), option.is_visible, sizes, keep_size=False
    )
    assert plan["roles"][0]["sizes"] == ["large", "medium", "small"]
    # A size step to small fails on a small object.
    assert plan["roles"][1]["sizes"] == ["large", "medium"]


def test_trajectory_mask_matches_brute_force():
    option = get_option()
    pm = option.pm
    random.seed(0)
    # Occupy some space so that the free-space part is exercised too.
    for _ in range(5):
        x, y = pm.random_position(6)
        option.record_obj_addition(x, y, 6)
    constraints = [(0, 0, True), (20, 0, False), (20, -10, False)]
    radius = 4
    mask = pm.trajectory_mask(constraints, radius)
    free = pm._erosion(pm.pos, radius)

    n_valid = 0
    for idx_x in range(pm.range[0]):
        for idx_y in range(pm.range[1]):
            x, y = pm._idx2pos(idx_x, idx_y)
            expected = bool(free[idx_x, idx_y])
            for dx, dy, vis in constraints:
                if not option.is_position_valid(x + dx, y + dy):
                    expected = False
                elif option.is_visible(x + dx, y + dy) != vis:
                    expected = False
            assert bool(mask[idx_x, idx_y]) == expected, (x, y)
            n_valid += expected
    assert n_valid > 0

    x, y = option.random_trajectory_start(radius, constraints)
    assert mask[pm._pos2idx(x, y)]


def test_occluded_targets():
    info = make_info()
    assert group.occluded_targets(info, ["C"], ["C", "L"]) == []
    info["states"][-1]["status"]["L"]["Object_1"] = "occluded"
    # Object_2 is not a target, so its occlusion does not matter.
    info["states"][0]["status"]["C"]["Object_2"] = "occluded"
    assert group.occluded_targets(info, ["C"], ["C", "L"]) == [(1, 3, "L")]
