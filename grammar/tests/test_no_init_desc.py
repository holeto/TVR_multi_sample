"""The mask without init_desc (use_init_desc=False).

It only sees the program decoded so far, so it must allow everything the mask
with init_desc allows, keep the constraints that need no scene, and lose the
ones that do.
"""
import os
import random

import pytest

from grammar import Geometry, MaskedDecoder, accepts, cfg
from grammar.data import canonical_order
from grammar.tests.make_fixture import load_fixture
from grammar.tests.test_mask import T, U, V, W, in_scene, obj, size_scene

GEOMETRY = Geometry()
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
GENERATED = load_fixture(os.path.join(FIXTURES, "generated.json"))
G4 = load_fixture(os.path.join(FIXTURES, "g4.json"))


def rules_of(program):
    return cfg.derivation(cfg.parse(cfg.program_tokens(program)))


def free_decoder(program=(), allow_repeated_moves=False):
    """Decoder without init_desc after the steps of program."""
    d = MaskedDecoder(
        geometry=GEOMETRY,
        allow_repeated_moves=allow_repeated_moves,
        use_init_desc=False,
    )
    if program:
        d.decode(rules_of(list(program))[:-1])
    return d


def actions(d, desc):
    return d.valid_map.get(desc, set())


def free_accepts(program, allow_repeated_moves=False):
    return accepts(
        None,
        rules_of(program),
        GEOMETRY,
        allow_repeated_moves,
        use_init_desc=False,
    )[0]


def test_init_desc_is_used_by_default():
    with pytest.raises(ValueError):
        MaskedDecoder()
    assert MaskedDecoder([[obj(T, [0, 0])]]).use_init_desc


@pytest.mark.parametrize("group", GENERATED, ids=lambda g: str(g["trans_id"]))
def test_allows_everything_the_scene_mask_allows(group):
    # At every step of a generated program, V with init_desc is a subset of
    # V without it.
    program = group["programs"][0]
    rules = rules_of(program)
    with_desc = MaskedDecoder(group["members"], GEOMETRY)
    without = free_decoder()
    for rule in rules:
        if with_desc.stack and with_desc.stack[-1][0] == "S":
            assert with_desc.valid <= without.valid
        with_desc.apply(rule)
        without.apply(rule)
    assert without.done


def test_g4_verdicts_need_no_scene():
    # The g4 verdicts come from the order rule and the one-edit rule, which
    # need no init_desc.
    for group in G4[:2]:
        program = group["programs"][0]
        assert not free_accepts(program)
        assert free_accepts(canonical_order(program))
    assert not free_accepts(G4[2]["programs"][0])


# --- what is kept ---------------------------------------------------------


def test_any_descriptor_can_start():
    d = free_decoder()
    assert len(d.valid_map) == 3 * 8 * 3 * 3
    d.apply(cfg.RULES_OF["T"][0])
    d.apply(cfg.RULES_OF["S"][0])
    d.apply(cfg.RULES_OF["R"][0])
    assert all(d.mask()[i] for i in cfg.RULES_OF["SIZE"])


def test_edit_value_must_differ_from_the_descriptor():
    # T is a large red metal cube.
    t = actions(free_decoder(), T)
    assert "color.red" not in t and "size.large" not in t
    assert "color.blue" in t and "size.small" in t


def test_one_edit_per_attribute():
    t = actions(free_decoder([(T, "color.blue")]), T)
    assert not any(a.startswith("color") for a in t)
    assert "shape.sphere" in t


def test_in_and_out_only_for_untouched_objects():
    t = actions(free_decoder([(T, "color.blue")]), T)
    assert "position.in" not in t and "position.out" not in t
    # Moving in, then editing, is fine; moving again is not.
    t = actions(free_decoder([(T, "position.in")]), T)
    assert "color.blue" in t
    assert not any(a.startswith("position") for a in t)


def test_nothing_after_out():
    assert actions(free_decoder([(T, "position.out")]), T) == set()


def test_one_move_per_object():
    t = actions(free_decoder([(T, "position.left.1")]), T)
    assert not any(a.startswith("position") for a in t)


def test_repeated_moves_never_cancel():
    d = free_decoder([(T, "position.left.1")], allow_repeated_moves=True)
    t = actions(d, T)
    assert "position.right.1" not in t  # back to the start
    assert "position.right.2" in t
    # front-left 1 and behind-right 1 cancel too.
    d = free_decoder(
        [(T, "position.behind-right.1")], allow_repeated_moves=True
    )
    assert "position.front-left.1" not in actions(d, T)


def test_order_rule():
    # large_red_metal_cube sorts before small_green_glass_cylinder.
    assert actions(free_decoder([(U, "color.red")]), T) == set()
    assert "color.red" in actions(free_decoder([(T, "color.blue")]), U)


def test_at_most_four_steps():
    program = [
        (T, "color.blue"),
        (T, "material.glass"),
        (T, "shape.sphere"),
        (T, "size.small"),
    ]
    stop, more = cfg.RULES_OF["C"]
    allowed = free_decoder(program).mask()
    assert allowed[stop] and not allowed[more]


# --- what is lost ---------------------------------------------------------


def scene_accepts(members, program):
    return accepts(members, rules_of(program), GEOMETRY)[0]


@pytest.mark.parametrize(
    "members, program",
    [
        # A visible object can not move in.
        (in_scene(), [(U, "position.in")]),
        # An invisible object can not be edited before it moves in.
        (in_scene(), [(T, "color.blue")]),
        # W is in no member: R must name an object of the scene.
        (in_scene(), [(W, "color.red")]),
        # W can not grow to large next to V.
        (size_scene(), [(W, "size.large")]),
        # No in-move brings T in in both members of this scene.
        (
            [
                [obj(T, [-35, 0]), obj(U, [5, 0])],
                [obj(T, [35, 0]), obj(U, [5, 12])],
            ],
            [(T, "position.in")],
        ),
        # A move off the grid.
        ([[obj(V, [35, 35])]], [(V, "position.behind.2")]),
    ],
)
def test_lost_constraints(members, program):
    assert not scene_accepts(members, program)
    assert free_accepts(program)


# --- random programs ------------------------------------------------------


@pytest.mark.parametrize("allow_repeated_moves", [False, True])
def test_random_programs(allow_repeated_moves):
    rng = random.Random(2)
    n_steps = set()
    for _ in range(100):
        d = free_decoder(allow_repeated_moves=allow_repeated_moves)
        rules = d.sample(rng)
        program = d.executor.steps
        n_steps.add(len(program))
        assert 1 <= len(program) <= 4
        assert cfg.tokens_program(cfg.derive(rules)) == program
        assert canonical_order(program) == program
        assert free_accepts(program, allow_repeated_moves)
    assert n_steps == {1, 2, 3, 4}
