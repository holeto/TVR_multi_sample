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


def make_plan(keys, steps):
    return {"roles": [{"key": key} for key in keys], "steps": steps}


def test_canonical_steps_sorts_commuting_steps():
    # Recolors of different objects come out in key order.
    plan = make_plan(
        ["red_cube", "blue_cube"],
        [(0, "color.green", None), (1, "color.yellow", None)],
    )
    assert group.canonical_steps(plan) == [
        (1, "color.yellow", None),
        (0, "color.green", None),
    ]
    # Edits of different attributes of one object are sorted by pair.
    plan = make_plan(
        ["red_cube"], [(0, "shape.sphere", None), (0, "color.blue", None)]
    )
    assert group.canonical_steps(plan) == [
        (0, "color.blue", None),
        (0, "shape.sphere", None),
    ]


def test_canonical_steps_keeps_dependent_steps():
    # Edits of an object moved in stay after the move, even if they sort
    # first.
    steps = [(0, "position.right.1", "in"), (0, "color.blue", None)]
    plan = make_plan(["red_cube"], steps)
    assert group.canonical_steps(plan) == steps

    # Steps of another object may be interleaved with the chain.
    steps = [
        (0, "position.right.1", "in"),
        (0, "color.blue", None),
        (1, "color.green", None),
    ]
    plan = make_plan(["red_cube", "blue_cube"], steps)
    assert group.canonical_steps(plan) == [
        (1, "color.green", None),
        (0, "position.right.1", "in"),
        (0, "color.blue", None),
    ]


def test_canonical_steps_sorts_steps_of_one_object():
    # An edit and an inner move of one object give the same images in either
    # order, and so do two inner moves.
    for steps in [
        [(0, "position.right.1", "inner"), (0, "color.blue", None)],
        [(0, "color.blue", None), (0, "position.right.1", "inner")],
    ]:
        assert group.canonical_steps(make_plan(["red_cube"], steps)) == [
            (0, "color.blue", None),
            (0, "position.right.1", "inner"),
        ]
    steps = [
        (0, "position.right.1", "inner"),
        (0, "position.front.1", "inner"),
    ]
    assert group.canonical_steps(make_plan(["red_cube"], steps)) == [
        (0, "position.front.1", "inner"),
        (0, "position.right.1", "inner"),
    ]


def test_canonical_steps_sorts_by_abstract_pair():
    # The direction of an out-move is not part of the canonical form.
    keys = ["a", "b"]
    for direction in ["behind", "front"]:
        steps = [
            (1, "color.blue", None),
            (0, "position.{}.2".format(direction), "out"),
        ]
        assert group.canonical_steps(make_plan(keys, steps)) == [
            steps[1],
            steps[0],
        ]


def test_abstract_transformation():
    assert group.abstract_pair("position.left.2", "out") == "position.out"
    assert group.abstract_pair("position.left.2", "in") == "position.in"
    assert group.abstract_pair("position.left.2", "inner") == (
        "position.left.2"
    )
    assert group.abstract_pair("color.red", None) == "color.red"

    info = make_info()
    info["idx"] = info["trans_id"] = 7
    for trans in info["transformations"]:
        obj = info["states"][0]["objects"][trans["obj_idx"]]
        trans["obj_key"] = group.obj_key(obj)
        trans["obj_desc"] = group.obj_desc(obj)
    abstract = group.abstract_transformation(info)
    assert abstract["trans_id"] == 7
    assert [s["pair"] for s in abstract["steps"]] == [
        "position.right.1",
        "position.in",
        "size.small",
    ]
    assert abstract["steps"][1]["obj_key"] == "large_blue_metal_cube"
    assert abstract["steps"][1]["obj_desc"]["color"] == "blue"


def add_step(info, trans, objects):
    info["transformations"].append(trans)
    info["states"].append({"objects": objects})


def test_move_violations():
    info = make_info()
    assert group.move_violations(info) == []
    stage = info["states"][-1]["objects"]

    # Object_1 was moved in, Object_0 was moved before.
    moved_back = [make_obj("Object_0", (0, 0)), stage[1]]
    add_step(
        info,
        {"obj_idx": 0, "attr": "position", "type": "inner"},
        moved_back,
    )
    assert group.move_violations(info) == [(3, 0, "second move")]
    assert group.move_violations(info, allow_repeated_moves=True) == [
        (3, 0, "move back to the start")
    ]
    add_step(
        info,
        {"obj_idx": 1, "attr": "position", "type": "inner"},
        [moved_back[0], make_obj("Object_1", (0, 10), "blue", size="small")],
    )
    assert group.move_violations(info, allow_repeated_moves=True)[1:] == [
        (4, 1, "move after moving in")
    ]

    info = make_info()
    info["transformations"][0]["type"] = "out"
    info["states"][1]["objects"][0]["position"] = (0, 30)
    add_step(
        info,
        {"obj_idx": 0, "attr": "position", "type": "in"},
        [make_obj("Object_0", (0, 10)), stage[1]],
    )
    assert group.move_violations(info) == [(3, 0, "step after moving out")]


def test_footprint_overlaps():
    option = get_option()
    coordinate, sizes = option.original_coordinate, option.size
    large = group.footprint(make_obj("A", (0, 0)), coordinate, sizes)
    assert large == ((0.0, 0.0), 0.9)
    # Large + small + gap = 1.35 m = 9 grid units.
    for x, expected in [(8, True), (10, False)]:
        small = group.footprint(
            make_obj("B", (x, 0), size="small"), coordinate, sizes
        )
        assert group.overlaps(large, small, coordinate["min_gap"]) == expected


def test_candidate_conflicts():
    option = get_option()
    coordinate, sizes = option.original_coordinate, option.size
    info = make_info()
    # Object_1 (small after step 2) could have moved in to (-10, 0) or
    # (-10, 20).
    info["transformations"][1]["candidates"] = [(-10, 0), (-10, 20)]
    assert group.candidate_conflicts(info, coordinate, sizes) == []

    # Object_0 then moves next to the unused candidate (-10, 20).
    stage = info["states"][-1]["objects"]
    add_step(
        info,
        {"obj_idx": 0, "attr": "position", "type": "inner"},
        [make_obj("Object_0", (-10, 28)), stage[1]],
    )
    assert group.candidate_conflicts(info, coordinate, sizes) == [(3, 0, 1)]

    # Growing next to a candidate is a conflict too, shrinking is not.
    for old, new, expected in [
        ("small", "medium", [(3, 0, 1)]),
        ("large", "medium", []),
    ]:
        info = make_info()
        info["transformations"][1]["candidates"] = [(-10, 0), (-10, 20)]
        stage = info["states"][-1]["objects"]
        stage[0] = make_obj("Object_0", (-10, 24), size=old)
        add_step(
            info,
            {"obj_idx": 0, "attr": "size"},
            [make_obj("Object_0", (-10, 24), size=new), stage[1]],
        )
        assert group.candidate_conflicts(info, coordinate, sizes) == expected


def test_canonical_steps_is_order_independent():
    steps = [
        (0, "position.right.1", "inner"),
        (1, "size.small", None),
        (2, "material.rubber", None),
        (1, "color.blue", None),
        (0, "position.front.2", "out"),
    ]
    plan = make_plan(["b", "a", "c"], steps)
    result = group.canonical_steps(plan)
    assert sorted(result) == sorted(steps)
    # Any valid reordering of the same steps gives the same canonical order.
    shuffled = [steps[i] for i in [2, 3, 0, 1, 4]]
    assert group.canonical_steps(make_plan(["b", "a", "c"], shuffled)) == (
        result
    )


def test_shared_candidates():
    option = get_option()
    info = make_info()
    for state in info["states"]:
        state["objects"].append(make_obj("Object_2", (0, -20), "green"))
        state["objects"].append(make_obj("Object_3", (-30, 20), "gray"))
    assert not option.is_visible(-30, 20)
    # Object_0 and Object_1 are targets, Object_3 is invisible.
    assert group.shared_candidates(info, option.is_visible) == [2]


def test_unoccluded_candidates():
    option = get_option()
    info = make_info()
    info["states"][0]["objects"].append(make_obj("Object_2", (0, -20), "gray"))
    info["states"][0]["objects"].append(make_obj("Object_3", (10, 20), "cyan"))
    for cam in ["C", "L"]:
        info["states"][0]["status"][cam]["Object_2"] = "visible"
        info["states"][0]["status"][cam]["Object_3"] = "visible"
    info["states"][0]["status"]["L"]["Object_3"] = "occluded"
    assert group.unoccluded_candidates(info, option.is_visible, ["C"]) == [
        2,
        3,
    ]
    assert group.unoccluded_candidates(
        info, option.is_visible, ["C", "L"]
    ) == [2]


def test_in_move_options():
    option = get_option()

    def moves(objects, idx):
        return group.in_move_options(
            objects,
            idx,
            option.move_options,
            option.is_position_valid,
            option.is_visible,
            option.original_coordinate,
            option.size,
        )

    # From (-35, 0) only moves of 2 steps towards x > 0 enter the view.
    objects = [make_obj("Object_0", (-35, 0), size="small")]
    assert moves(objects, 0) == [
        ("behind", 2),
        ("behind-left", 2),
        ("behind-right", 2),
    ]
    # An object next to the destination of behind.2 blocks it.
    objects.append(make_obj("Object_1", (-15, 4), size="small"))
    assert ("behind", 2) not in moves(objects, 0)
    # A visible object has no in-move.
    assert moves([make_obj("Object_0", (0, 0))], 0) == []


def test_repeated_attr_steps():
    info = make_info()
    assert group.repeated_attr_steps(info) == []
    info["transformations"].append(
        {"obj_idx": 0, "attr": "position", "pair": "position.left.1"}
    )
    assert group.repeated_attr_steps(info) == []
    info["transformations"].append(
        {"obj_idx": 1, "attr": "size", "pair": "size.large"}
    )
    assert group.repeated_attr_steps(info) == [(4, 1, "size")]
