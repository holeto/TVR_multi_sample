"""The decoder's mask on hand-made scenes covering edge cases.

Grid units: the visible square is |x|, |y| <= 20, one move step is 10 units,
and two objects overlap when their centres are closer than r_a + r_b + gap
(large 6, medium 4, small 2 units of radius, gap 1).
"""
import os
import random
import sys

import pytest

from grammar import Geometry, MaskedDecoder, cfg
from grammar.data import canonical_order

GEOMETRY = Geometry()

T = ("large", "red", "metal", "cube")
U = ("small", "green", "glass", "cylinder")
V = ("medium", "blue", "rubber", "sphere")
W = ("small", "yellow", "metal", "sphere")


def obj(desc, position):
    attrs = ["size", "color", "material", "shape"]
    return dict(zip(attrs, desc), position=position)


def decoder(members, program=(), allow_repeated_moves=False):
    """Decoder after the steps of program, with C on top of the stack."""
    d = MaskedDecoder(members, GEOMETRY, allow_repeated_moves)
    if program:
        rules = cfg.derivation(cfg.parse(cfg.program_tokens(list(program))))
        d.decode(rules[:-1])
    return d


def actions(d, desc):
    return set(a for x, a in d.valid if x == desc)


def rule_allowed(d, step):
    """Whether the mask allows every rule of one more step."""
    rules = cfg.derivation(cfg.parse(cfg.program_tokens([step])))
    # Skip T and the final C -> eps; expand C -> S C instead of T -> S C.
    rules = [cfg.RULES_OF["C"][1]] + rules[1:-1]
    if not d.executor.steps:
        rules[0] = cfg.RULES_OF["T"][0]
    try:
        d.decode(rules)
    except Exception:
        return False
    return True


# --- in-moves -------------------------------------------------------------


def in_scene():
    # T waits outside on the left in both members; behind-2 (+x) brings it
    # in everywhere. U is visible, V invisible on the far side.
    return [
        [obj(T, [-35, 0]), obj(U, [5, 0]), obj(V, [0, 35])],
        [obj(T, [-35, 10]), obj(U, [5, 12]), obj(V, [30, -30])],
    ]


def test_in_needs_a_common_move():
    d = decoder(in_scene())
    assert "position.in" in actions(d, T)
    # Visible objects can not move in.
    assert "position.in" not in actions(d, U)
    # On opposite sides, no move brings T in in both members.
    members = in_scene()
    members[1][0] = obj(T, [35, 0])
    assert "position.in" not in actions(decoder(members), T)


def test_in_candidates_and_later_steps():
    d = decoder(in_scene(), [(T, "position.in")])
    m0 = d.executor.members[0]
    # Every in-move from (-35, 0) that lands in view.
    assert m0.objects[0].cells == [(-15, -20), (-15, 0), (-15, 20)]
    assert m0.reserved == {0: m0.objects[0].cells}
    # A moved-in object can be edited but not moved again.
    after = actions(d, T)
    assert "color.blue" in after
    assert not any(a.startswith("position") for a in after)
    # Others must avoid every candidate: U at (5, 0) can not move front 2
    # onto (-15, 0), but can move behind.
    u = actions(d, U)
    assert "position.front.2" not in u
    assert "position.behind.1" in u


def test_in_first_then_own_edit_is_in_order():
    # The order rule lets an object's edits follow its own in-move although
    # their text sorts before "position.in".
    d = decoder(in_scene(), [(T, "position.in")])
    assert rule_allowed(d, (T, "color.blue"))


# --- out-moves ------------------------------------------------------------


def out_scene():
    # T near the right edge (y = 15): right.1 takes it out in both members.
    return [
        [obj(T, [0, 15]), obj(U, [0, -10])],
        [obj(T, [5, 18]), obj(U, [-5, -12])],
    ]


def test_out_needs_untouched_visible_object():
    d = decoder(out_scene())
    assert "position.out" in actions(d, T)
    # After an edit of T, T can not move out any more.
    d = decoder(out_scene(), [(T, "color.blue")])
    assert "position.out" not in actions(d, T)


def test_nothing_happens_after_out():
    d = decoder(out_scene(), [(T, "position.out")])
    assert actions(d, T) == set()
    assert d.executor.members[0].objects[0].cells  # possible destinations
    assert all(
        not GEOMETRY.visible(c) for c in d.executor.members[0].objects[0].cells
    )


def test_out_needs_a_common_move():
    members = out_scene()
    members[1][0] = obj(T, [0, -15])  # near the opposite edge
    assert "position.out" not in actions(decoder(members), T)


# --- size changes ---------------------------------------------------------


def size_scene():
    # W (small) with V (medium) 10 units away: W may grow to medium (needs
    # more than 4 + 4 + 1 = 9) but not to large (needs more than 6 + 4 + 1).
    return [
        [obj(W, [0, 0]), obj(V, [0, 10])],
        [obj(W, [10, 0]), obj(V, [10, 10])],
    ]


def test_size_growth_checks_overlap():
    w = actions(decoder(size_scene()), W)
    assert "size.medium" in w
    assert "size.large" not in w
    assert "size.small" not in w  # no change


def test_shrinking_needs_no_space():
    members = [[obj(T, [0, 0]), obj(V, [0, 11])]]
    assert "size.small" in actions(decoder(members), T)


def test_one_size_edit_per_object():
    d = decoder(size_scene(), [(W, "size.medium")])
    assert not any(a.startswith("size") for a in actions(d, W))


def test_moved_in_object_grows_where_some_candidate_fits():
    # W moves in from (-35, 0) to one of (-15, -20), (-15, 0), (-15, 20).
    # A large V at (-15, 10) blocks a large W at the two upper cells only.
    members = [[obj(W, [-35, 0]), obj(("large",) + V[1:], [-15, 10])]]
    d = decoder(members, [(W, "position.in")])
    assert len(d.executor.members[0].objects[0].cells) == 3
    # Its own candidate cells do not count as reserved for it.
    assert "size.large" in actions(d, W)
    # A second large object at (-15, -30) blocks the last cell too.
    members[0].append(obj(("large", "gray", "glass", "cube"), [-15, -30]))
    d = decoder(members, [(W, "position.in")])
    assert "size.large" not in actions(d, W)


# --- repeated moves -------------------------------------------------------


def move_scene():
    return [
        [obj(T, [0, 0]), obj(U, [15, 15])],
        [obj(T, [5, 5]), obj(U, [-15, 15])],
    ]


def test_second_move_masked_by_default():
    d = decoder(move_scene(), [(T, "position.left.1")])
    assert not any(a.startswith("position") for a in actions(d, T))


def test_repeated_moves_allowed_but_not_back_to_start():
    d = decoder(
        move_scene(), [(T, "position.left.1")], allow_repeated_moves=True
    )
    t = actions(d, T)
    assert "position.right.1" not in t  # back to the start
    assert "position.right.2" in t
    assert "position.left.1" in t  # same move again
    # The order rule still holds: "position.front.1" sorts before left.1.
    assert "position.front.1" not in t


def test_repeated_moves_flag_rejects_and_accepts_program():
    program = [(T, "position.left.1"), (T, "position.right.2")]
    rules = cfg.derivation(cfg.parse(cfg.program_tokens(program)))
    with pytest.raises(Exception):
        MaskedDecoder(move_scene(), GEOMETRY).decode(rules)
    d = MaskedDecoder(move_scene(), GEOMETRY, allow_repeated_moves=True)
    assert d.decode(rules).done


# --- order, uniqueness, length --------------------------------------------


def test_order_rule():
    members = [[obj(T, [0, 0]), obj(U, [10, -10])]]
    # large_red_metal_cube < small_green_glass_cylinder
    d = decoder(members, [(U, "color.red")])
    assert actions(d, T) == set()
    d = decoder(members, [(T, "color.blue")])
    assert "color.red" in actions(d, U)
    # Same object: later steps need a larger action text.
    assert "color.gray" not in actions(d, T)  # color edited already
    assert "shape.sphere" in actions(d, T)


def test_descriptor_must_be_unique_in_every_member():
    members = [
        [obj(T, [0, 0]), obj(U, [10, 10])],
        [obj(T, [0, 0]), obj(U, [10, 10]), obj(U, [-10, -10])],
    ]
    d = decoder(members)
    assert d.executor.descriptors() == [T]
    assert actions(d, U) == set()
    # The mask follows: SIZE may only start T's descriptor.
    d.apply(cfg.RULES_OF["T"][0])
    d.apply(cfg.RULES_OF["S"][0])
    d.apply(cfg.RULES_OF["R"][0])
    allowed = [cfg.RULES[i].rhs[0] for i, ok in enumerate(d.mask()) if ok]
    assert allowed == ["'large'"]


def test_at_most_four_steps():
    members = [[obj(T, [0, 0])]]
    program = [
        (T, "color.blue"),
        (T, "material.glass"),
        (T, "shape.sphere"),
        (T, "size.small"),
    ]
    d = decoder(members, program)
    stop, more = cfg.RULES_OF["C"]
    assert d.mask()[stop] and not d.mask()[more]


def test_no_step_left_only_allows_stopping():
    # After its only out-move, T can do nothing more and nothing else exists.
    d = decoder([[obj(T, [0, 15])]], [(T, "position.out")])
    stop, more = cfg.RULES_OF["C"]
    assert d.mask()[stop] and not d.mask()[more]


# --- random programs ------------------------------------------------------


def rich_scene(rng):
    descs = [
        (s, c, m, sh)
        for s in cfg.SIZES
        for c in cfg.COLORS[:3]
        for m in cfg.MATERIALS[:2]
        for sh in cfg.SHAPES[:2]
    ]
    chosen = rng.sample(descs, 6)
    visible = [(x, y) for x in range(-20, 21, 10) for y in range(-20, 21, 10)]
    hidden = [(x, y) for x in (-35, 35) for y in range(-30, 31, 15)]
    members = []
    for _ in range(3):
        cells = rng.sample(visible, 4) + rng.sample(hidden, 2)
        members.append([obj(d, list(c)) for d, c in zip(chosen, cells)])
    return members


def trance_canonical(program):
    sys.path.append(
        os.path.join(os.path.dirname(__file__), "..", "..", "trance")
    )
    import group

    roles, steps = [], []
    for desc, action in program:
        key = "_".join(desc)
        if key not in roles:
            roles.append(key)
        move = action.split(".")[1] if action.startswith("position") else None
        move = move if move in ("in", "out") else ("inner" if move else None)
        pair = action
        steps.append((roles.index(key), pair, move))
    plan = {"roles": [{"key": k} for k in roles], "steps": steps}
    return [
        (program[[i for i, s in enumerate(steps) if s == step][0]][0], step[1])
        for step in group.canonical_steps(plan)
    ]


@pytest.mark.parametrize("allow_repeated_moves", [False, True])
def test_random_programs(allow_repeated_moves):
    rng = random.Random(0)
    n_steps = []
    for _ in range(150):
        members = rich_scene(rng)
        d = MaskedDecoder(members, GEOMETRY, allow_repeated_moves)
        rules = d.sample(rng)
        program = d.executor.steps
        n_steps.append(len(program))
        assert 1 <= len(program) <= 4
        # The rules spell the program, and parsing gives the rules back.
        tokens = cfg.derive(rules)
        assert cfg.tokens_program(tokens) == program
        assert cfg.derivation(cfg.parse(tokens)) == rules
        # Decoded programs are in canonical order.
        assert canonical_order(program) == program
        # Decoding the same rules again is accepted.
        again = MaskedDecoder(members, GEOMETRY, allow_repeated_moves)
        assert again.decode(rules).done
    assert set(n_steps) == {1, 2, 3, 4}


def test_canonical_order_matches_generator():
    rng = random.Random(1)
    n_compared = 0
    for _ in range(300):
        d = MaskedDecoder(rich_scene(rng), GEOMETRY)
        d.sample(rng, stop_prob=0.1)
        program = d.executor.steps
        shuffled = program[:]
        rng.shuffle(shuffled)
        # Shuffles that keep each in/out before its object's other steps
        # are valid sampled orders; both orders give the same canonical one.
        if canonical_order(shuffled) != program:
            continue
        assert trance_canonical(shuffled) == program
        n_compared += 1
    assert n_compared >= 250
