"""The decoder's mask on hand-made scenes covering edge cases.

Grid units: the visible square is |x|, |y| <= 20, one move step is 10 units,
and two objects overlap when their centres are closer than r_a + r_b + gap
(large 6, medium 4, small 2 units of radius, gap 1).
"""
import json
import os
import random
import sys

import pytest

from grammar import Geometry, MaskedDecoder, cfg
from grammar.data import canonical_order
from grammar.geometry import PROPERTIES_JSON

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


# --- exact contact ---------------------------------------------------------

# Blender lets two objects stand at exactly r_a + r_b + gap, but the float
# test of Geometry.overlaps may call that an overlap (a 3-4-5 offset of small
# objects gives 0.7499999999999998 m against the limit 0.75 m). The checks that
# follow the generator's Blender test must allow contact, including where a
# moved-in object may land; the candidates P(o) and reservation of 'in' must
# not.


def collides_by(geometry, a, size_a, b, size_b):
    dx, dy = a[0] - b[0], a[1] - b[1]
    limit = geometry.cell_sizes[size_a] + geometry.cell_sizes[size_b] + 1
    return dx * dx + dy * dy, limit * limit


def test_collides_differs_from_overlaps_only_at_contact():
    anchors = [(0, 0), (19, -14), (-40, -13), (37, 22), (-3, 40)]
    n_contact = n_float_overlap_at_contact = 0
    for size_a in GEOMETRY.sizes:
        for size_b in GEOMETRY.sizes:
            for a in anchors:
                for dx in range(-30, 31):
                    for dy in range(-30, 31):
                        b = (a[0] + dx, a[1] + dy)
                        d2, limit2 = collides_by(GEOMETRY, a, size_a, b, size_b)
                        exact = GEOMETRY.collides(a, size_a, b, size_b)
                        assert exact == (d2 < limit2)
                        float_test = GEOMETRY.overlaps(a, size_a, b, size_b)
                        if d2 == limit2:
                            n_contact += 1
                            n_float_overlap_at_contact += float_test
                        else:
                            assert exact == float_test
    # Contact cells exist, and the float test does misjudge some of them.
    assert n_contact > 0
    assert n_float_overlap_at_contact > 0


def test_geometry_off_the_cell_grid_falls_back_to_floats():
    with open(PROPERTIES_JSON) as f:
        properties = json.load(f)
    properties["size"]["small"] = 0.31
    with pytest.warns(UserWarning, match="whole grid cells"):
        geometry = Geometry(properties)
    a, b = (19, -14), (16, -18)
    assert geometry.collides(a, "small", b, "small") == geometry.overlaps(
        a, "small", b, "small"
    )


def contact_inner_scene():
    # W moves left.1 from (19, -4) to (19, -14); U (small) at (16, -18) is
    # at the minimum distance 5 (offset 3, 4) from there.
    return [[obj(W, [19, -4]), obj(U, [16, -18])]]


def test_inner_move_allowed_at_exact_contact():
    assert GEOMETRY.overlaps((19, -14), "small", (16, -18), "small")
    assert "position.left.1" in actions(decoder(contact_inner_scene()), W)
    # One cell closer is a real overlap.
    members = contact_inner_scene()
    members[0][1] = obj(U, [17, -18])
    assert "position.left.1" not in actions(decoder(members), W)


def contact_in_scene():
    # T (large) waits at (-40, -13); only behind.2 -> (-20, -13) and
    # behind-right.2 -> (-20, 7) bring it in. In member 1, V (medium) at
    # (-20, 18) touches (-20, 7) (distance 11 = 6 + 4 + 1) and W (medium too)
    # really overlaps (-20, -13), so behind-right.2 is the only common move
    # and it is lost to the float test.
    medium_w = ("medium",) + W[1:]
    return [
        [obj(T, [-40, -13]), obj(V, [0, 0])],
        [obj(T, [-40, -13]), obj(V, [-20, 18]), obj(medium_w, [-20, -9])],
    ]


def test_in_move_with_common_move_at_exact_contact():
    assert GEOMETRY.overlaps((-20, 7), "large", (-20, 18), "medium")
    d = decoder(contact_in_scene())
    assert "position.in" in actions(d, T)
    m1 = d.executor.members[1]
    assert m1.moves_for(0, "in") == {("behind-right", "2")}
    # A real overlap instead of the contact loses the move.
    members = contact_in_scene()
    members[1][1] = obj(V, [-20, 17])
    assert "position.in" not in actions(decoder(members), T)


def test_in_candidates_ignore_exact_contact():
    # The generator's in_candidates use the float test, so the contact cell
    # (-20, 7) is not a candidate and is not reserved. T may still land there,
    # since the move itself passes Blender's test.
    members = [[obj(T, [-40, -13]), obj(V, [-20, 18])]]
    d = decoder(members, [(T, "position.in")])
    m0 = d.executor.members[0]
    assert m0.objects[0].cells == [(-20, -13), (-20, 7)]
    assert m0.objects[0].float_cells == [(-20, -13)]
    assert m0.reserved == {0: [(-20, -13)]}


def test_in_move_lands_at_exact_contact():
    # In member 1 of contact_in_scene, T can only land at the contact cell
    # (-20, 7). T is there afterwards, but P(o) is empty, so nothing is
    # reserved.
    d = decoder(contact_in_scene(), [(T, "position.in")])
    m1 = d.executor.members[1]
    t = m1.objects[0]
    assert t.cells == [(-20, 7)]
    assert t.float_cells == []
    assert m1.reserved == {0: []}
    # T blocks the others there and only there: V may move behind.1 to
    # (-10, 18) but not behind-left.1 to (-10, 8), and W may grow.
    assert m1.valid(1, "position.behind.1")
    assert not m1.valid(1, "position.behind-left.1")
    assert m1.valid(2, "size.large")
    assert ("front", "1") in m1.moves_for(1, "out")


def test_reservation_is_free_at_exact_contact():
    # U (small) may come in at (-20, 0), (-20, 20) or (-20, -20), all reserved.
    # W (small) at (-17, 14) moving right.1 to (-17, 24) would touch the
    # reserved (-20, 20) (offset 3, 4). The float test calls that blocked
    # here, but the generator's Python may round it the other way.
    members = [[obj(U, [-40, 0]), obj(W, [-17, 14])]]
    d = decoder(members, [(U, "position.in")])
    m0 = d.executor.members[0]
    assert (-20, 20) in m0.reserved[0]
    assert GEOMETRY.overlaps((-17, 24), "small", (-20, 20), "small")
    moves = m0.moves_for(1, "out")
    assert ("right", "1") in moves
    assert ("right", "2") in moves


def test_candidate_at_exact_contact_is_not_reserved():
    # trans_id 9902 of center_2step: the cylinder may come in at (-6, -3), at
    # exactly the minimum distance from the cube at (-3, 1). The float test
    # here calls that cell free, the generator's called it taken and let the
    # cube grow; the grown cube would overlap the cell.
    cylinder = ("small", "brown", "glass", "cylinder")
    cube = ("small", "purple", "rubber", "cube")
    members = [[obj(cylinder, [-6, -23]), obj(cube, [-3, 1])]]
    d = decoder(members, [(cylinder, "position.in")])
    m0 = d.executor.members[0]
    assert (-6, -3) in m0.objects[0].float_cells
    assert GEOMETRY.touches((-6, -3), "small", (-3, 1), "small")
    assert (-6, -3) not in m0.reserved[0]
    assert (-6, -13) in m0.reserved[0]
    assert "size.large" in actions(d, cube)
    # A candidate that is not at contact still blocks the growth.
    members = [[obj(cylinder, [-6, -23]), obj(cube, [-6, 4])]]
    d = decoder(members, [(cylinder, "position.in")])
    assert (-6, -3) in d.executor.members[0].reserved[0]
    assert "size.large" not in actions(d, cube)


def test_size_growth_allowed_at_exact_contact():
    # W (small) at (0, 0) grows to large; the large V at (12, 5) is at
    # 13 = 6 + 6 + 1 then.
    large_v = ("large",) + V[1:]
    assert GEOMETRY.overlaps((0, 0), "large", (12, 5), "large")
    members = [[obj(W, [0, 0]), obj(large_v, [12, 5])]]
    assert "size.large" in actions(decoder(members), W)
    members = [[obj(W, [0, 0]), obj(large_v, [11, 5])]]
    assert "size.large" not in actions(decoder(members), W)


def test_out_move_at_exact_contact_keeps_float_cells():
    # W (small) at (19, 1): right.2 leaves the view at (19, 21), which U
    # (small) at (22, 17) touches. The move is free, so the out is valid and
    # the destination is among the possible cells, but the cells that 'in'
    # candidates see (float_cells) are the float test's, as before.
    members = [[obj(W, [19, 1]), obj(U, [22, 17])]]
    d = decoder(members)
    m0 = d.executor.members[0]
    assert ("right", "2") in m0.moves_for(0, "out")
    assert "position.out" in actions(d, W)
    d = decoder(members, [(W, "position.out")])
    w = d.executor.members[0].objects[0]
    assert (19, 21) in w.cells
    assert (19, 21) not in w.float_cells
    assert set(w.float_cells) < set(w.cells)


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
