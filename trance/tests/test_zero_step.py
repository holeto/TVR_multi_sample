# -*- coding: utf-8 -*-
"""
Tests of 0-step transformations in the bpy-free parts of the generator.

Run from the repository root: python -m pytest trance/tests
"""
import importlib
import importlib.util
import os
import sys
import types
from collections import Counter
from unittest import mock

import numpy as np

TRANCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(TRANCE_DIR)

import check_groups  # noqa: E402
import group  # noqa: E402


def load_record():
    # utils/__init__.py needs the renderer's config packages; record.py only
    # needs utils.sample, so that module alone stands in for the package.
    package = types.ModuleType("utils")
    package.__path__ = []
    spec = importlib.util.spec_from_file_location(
        "utils.sample", os.path.join(TRANCE_DIR, "utils", "sample.py")
    )
    sample = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sample)
    with mock.patch.dict(
        sys.modules, {"utils": package, "utils.sample": sample}
    ):
        sys.modules.pop("record", None)
        return importlib.import_module("record")


def make_manager(record):
    return record.SampleManager(
        valid_init_vis=range(1, 11),
        total_obj=10,
        valid_steps=range(0, 3),
        valid_pairs=["color.red"],
        valid_move_type=["inner", "out", "in"],
    )


def zero_step_info():
    obj = {
        "size": "large",
        "color": "red",
        "material": "metal",
        "shape": "cube",
        "position": [0, 0],
    }
    state = {
        "objects": [obj],
        "status": {"Camera_Center": {}},
        "images": {"Camera_Center": "a.png"},
    }
    return {
        "idx": 0,
        "trans_id": 0,
        "member": 0,
        "states": [state],
        "transformations": [],
    }


def test_failed_initial_state_does_not_count_a_step():
    manager = make_manager(load_record())
    manager.random_init()
    manager.sample_fail(keep_init_vis=True)
    assert manager.n_step == {0: 0, 1: 0, 2: 0}


def test_failed_sample_returns_its_drawn_step():
    manager = make_manager(load_record())
    manager.random_init()
    step = manager.random_step()
    manager.sample_fail()
    assert manager.n_step[step] == 0


def test_step_counts_stay_balanced_after_failures():
    np.random.seed(0)
    manager = make_manager(load_record())
    for _ in range(50):
        manager.random_init()
        manager.sample_fail(keep_init_vis=True)
    counts = Counter(int(manager.random_step()) for _ in range(600))
    assert counts == {0: 200, 1: 200, 2: 200}


def test_plan_of_zero_steps_is_empty():
    info = zero_step_info()
    plan = group.extract_plan(info, lambda x, y: True, ["large"])
    assert plan == {"roles": [], "steps": []}
    assert group.canonical_steps(plan) == []
    assert group.occluded_targets(info, ["Camera_Center"], ["Camera_Center"]) == []


def test_check_groups_accepts_zero_steps():
    info = zero_step_info()
    assert check_groups.is_canonical(info)
    assert check_groups.signature(info) == []
    assert check_groups.check_member(info) == []
