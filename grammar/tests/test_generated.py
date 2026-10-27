"""The mask on transformations of the current generator.

fixtures/generated.json comes from a run of trance/render.py on 2026-10-26
(n_per_trans 4, 30 transformations, --no_render): 2-4 visible shared
distractors (min 2), and per member 2-3 objects that can move in: in-targets
plus core decoys making 2 in total (re-placed in the anchor, directions drawn
independently of the targets'), plus extras. Rebuild it
with `python -m grammar.tests.make_fixture <info dir> <out>`.
"""
import os

import pytest

from grammar import Geometry, MaskedDecoder, accepts, cfg
from grammar.data import canonical_order
from grammar.tests.make_fixture import load_fixture

GROUPS = load_fixture(
    os.path.join(os.path.dirname(__file__), "fixtures", "generated.json")
)
GEOMETRY = Geometry()


def ids(group):
    return "trans{}".format(group["trans_id"])


def rules_of(program):
    return cfg.derivation(cfg.parse(cfg.program_tokens(program)))


def in_capable(decoder):
    """Descriptors the mask allows to move in at the next step."""
    return set(d for d, a in decoder.valid if a == "position.in")


@pytest.mark.parametrize("group", GROUPS, ids=ids)
def test_generated_program_is_accepted(group):
    programs = group["programs"]
    assert all(p == programs[0] for p in programs)
    assert canonical_order(programs[0]) == programs[0]
    assert accepts(group["members"], rules_of(programs[0]), GEOMETRY) == (
        True,
        None,
    )


N_IN = 2  # min_invisible_distractors of the run: in-targets + core decoys


def in_targets(program):
    return set(desc for desc, action in program if action == "position.in")


@pytest.mark.parametrize("group", GROUPS, ids=ids)
def test_in_capable_total_is_fixed(group):
    # Core decoys fill up the in-targets to the same total everywhere.
    program = group["programs"][0]
    assert group["n_core_decoys"] + len(in_targets(program)) == N_IN
    assert group["n_core_decoys"] >= 1


def test_count_does_not_tell_whether_an_in_move_exists():
    # Without an in-move, exactly the N_IN core decoys can move in before the
    # first step. With one, the count is N_IN too, except when an earlier step
    # frees the target's landing cell (an out-move next to it): then only the
    # decoys can move in before that step.
    for group in GROUPS:
        program = group["programs"][0]
        count = len(in_capable(MaskedDecoder(group["members"], GEOMETRY)))
        if not in_targets(program):
            assert count == N_IN
        else:
            assert count in (N_IN, N_IN - len(in_targets(program)))


def could_move_in(decoder):
    """Descriptors whose in-move is valid now, whatever the order rule says.

    The order rule only fixes where a step goes: a decoy that sorts before the
    previous step could have moved in earlier, in an equivalent program.
    """
    executor = decoder.executor
    return set(
        desc
        for desc in executor.descriptors()
        if executor.step_valid(desc, "position.in")
    )


def test_decoys_hide_the_in_target():
    # At the step of an in-move, N_IN objects could move in: the target and
    # its decoys.
    n_in = 0
    for group in GROUPS:
        program = group["programs"][0]
        rules = rules_of(program)
        decoder = MaskedDecoder(group["members"], GEOMETRY)
        for rule in rules[:-1]:
            if decoder.stack and decoder.stack[-1][0] == "S":
                desc, action = program[len(decoder.executor.steps)]
                if action == "position.in":
                    n_in += 1
                    capable = could_move_in(decoder)
                    assert desc in capable
                    assert len(capable) == N_IN
            decoder.apply(rule)
    assert n_in > 0
