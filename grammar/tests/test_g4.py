"""Parse trees and masks of the transformations of the g4 test run.

fixtures/g4.json comes from TRANCE_dataset/trance_gen_test/g4 (4 members per
transformation, generated on 2026-10-19, before the grammar rules existed):
- 0: recolor the small yellow glass cube, move it behind 2, recolor the
  large blue glass cube (not in canonical order);
- 1: recolor the medium yellow metal cube, move the medium gray rubber sphere
  out, move the cube front-right 1, recolor the medium green glass sphere
  (not in canonical order);
- 2: recolor the large yellow glass cube twice (now forbidden).
"""
import os

import pytest

from grammar import Geometry, accepts, cfg
from grammar.data import canonical_order
from grammar.tests.make_fixture import load_fixture

GROUPS = load_fixture(
    os.path.join(os.path.dirname(__file__), "fixtures", "g4.json")
)
GEOMETRY = Geometry()


def rules_of(program):
    return cfg.derivation(cfg.parse(cfg.program_tokens(program)))


def test_fixture():
    assert [g["trans_id"] for g in GROUPS] == [0, 1, 2]
    assert all(len(g["members"]) == 4 for g in GROUPS)


@pytest.mark.parametrize(
    "group, n_rules", [(GROUPS[0], 29), (GROUPS[1], 37), (GROUPS[2], 19)]
)
def test_parse_trees(group, n_rules):
    trees = [cfg.parse(cfg.program_tokens(p)) for p in group["programs"]]
    # Every member has the same transformation, so the same tree.
    assert all(tree == trees[0] for tree in trees)
    rules = cfg.derivation(trees[0])
    assert len(rules) == n_rules
    assert cfg.tokens_program(cfg.derive(rules)) == group["programs"][0]


def test_step_structure():
    tree = cfg.parse(cfg.program_tokens(GROUPS[1]["programs"][0]))
    actions = []
    node = tree
    while node.rule != cfg.RULES_OF["C"][0]:
        s, node = node.children
        a = s.children[1]
        actions.append(cfg.RULES[a.rule].rhs)
    assert actions == [
        ("'color'", "COLOR"),
        ("'position'", "'out'"),
        ("'position'", "DIR", "N"),
        ("'color'", "COLOR"),
    ]


@pytest.mark.parametrize("group", GROUPS[:2])
def test_recorded_order_rejected_canonical_accepted(group):
    program = group["programs"][0]
    ok, message = accepts(group["members"], rules_of(program), GEOMETRY)
    assert not ok and "not allowed" in message
    canonical = canonical_order(program)
    assert canonical != program
    assert accepts(group["members"], rules_of(canonical), GEOMETRY) == (
        True,
        None,
    )


def test_repeated_edit_rejected():
    program = GROUPS[2]["programs"][0]
    assert canonical_order(program) == program
    ok, message = accepts(GROUPS[2]["members"], rules_of(program), GEOMETRY)
    assert not ok
    # Refused at the second color edit, after the first one was applied.
    assert "color.blue" in message and "('A'" in message
