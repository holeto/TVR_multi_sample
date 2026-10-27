"""The grammar: size, derivation lengths and parse trees."""
import pytest

from grammar import cfg
from grammar.executor import ACTIONS

DESC = ("large", "red", "metal", "cube")


def rule_count(program):
    return len(cfg.derivation(cfg.parse(cfg.program_tokens(program))))


def test_size():
    assert len(cfg.NONTERMINALS) == 11
    assert len(cfg.RULES) == 39
    assert cfg.K == 40 and cfg.NOOP == 39


def test_derivation_lengths():
    # Each step adds its own rules plus one C rule; T and the final C -> eps
    # add one each.
    assert rule_count([(DESC, "color.blue")]) == 1 + 8 + 1
    assert rule_count([(DESC, "position.front-left.2")]) == 1 + 9 + 1
    assert rule_count([(DESC, "position.in")]) == 1 + 7 + 1 == 9
    longest = [(DESC, "position.front.1")] * 4
    assert rule_count(longest) == 1 + 4 * 9 + 3 + 1 == 41


@pytest.mark.parametrize("action", ACTIONS)
def test_every_action_round_trips(action):
    program = [(DESC, action), (("small", "cyan", "glass", "sphere"), action)]
    tokens = cfg.program_tokens(program)
    tree = cfg.parse(tokens)
    assert tree.symbol == "T"
    assert cfg.tree_tokens(tree) == tokens
    rules = cfg.derivation(tree)
    assert cfg.derive(rules) == tokens
    assert cfg.tokens_program(tokens) == program


def test_parse_tree_shape():
    tree = cfg.parse(cfg.program_tokens([(DESC, "position.in")]))
    s, c = tree.children
    assert (s.symbol, c.symbol) == ("S", "C")
    assert c.rule == cfg.RULES_OF["C"][0]  # C -> eps
    r, a = s.children
    assert [child.symbol for child in r.children] == cfg.DESC_NT
    assert [child.symbol for child in a.children] == ["'position'", "'in'"]


@pytest.mark.parametrize(
    "tokens",
    [
        [],
        ["large", "red", "metal"],
        ["large", "red", "metal", "cube", "color"],
        ["large", "red", "metal", "cube", "position", "up", "1"],
        ["large", "red", "metal", "cube", "position", "front", "3"],
        ["huge", "red", "metal", "cube", "color", "red"],
        ["large", "red", "metal", "cube", "color", "red", "extra"],
    ],
)
def test_parse_errors(tokens):
    with pytest.raises(cfg.ParseError):
        cfg.parse(tokens)


def test_derive_rejects_wrong_rules():
    with pytest.raises(cfg.ParseError):
        cfg.derive([cfg.RULES_OF["S"][0]])  # S does not expand T
    with pytest.raises(cfg.ParseError):
        cfg.derive([cfg.RULES_OF["T"][0]])  # incomplete
