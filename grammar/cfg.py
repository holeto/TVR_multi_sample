"""The context-free grammar of TRANCE transformations.

T     -> S C
C     -> eps | S C
S     -> R A
R     -> SIZE COLOR MAT SHAPE
A     -> 'color' COLOR | 'shape' SHAPE | 'size' SIZE | 'material' MAT
       | 'position' DIR N | 'position' 'out' | 'position' 'in'

A step S reads "the object whose stage-0 descriptor is R undergoes A". A
program is a list of steps, each a pair (descriptor, action), e.g.
(("large", "red", "metal", "cube"), "position.front-left.1").
"""
from collections import namedtuple

SIZES = ["large", "medium", "small"]
COLORS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
MATERIALS = ["rubber", "metal", "glass"]
SHAPES = ["cube", "sphere", "cylinder"]
DIRECTIONS = [
    "front",
    "behind",
    "left",
    "right",
    "front-left",
    "front-right",
    "behind-left",
    "behind-right",
]
STEPS = ["1", "2"]

NONTERMINALS = [
    "T",
    "C",
    "S",
    "R",
    "A",
    "SIZE",
    "COLOR",
    "MAT",
    "SHAPE",
    "DIR",
    "N",
]
# Attribute name of an edit -> the non-terminal of its value.
ATTR_NT = {
    "color": "COLOR",
    "shape": "SHAPE",
    "size": "SIZE",
    "material": "MAT",
}
# Order of the descriptor's attributes in R.
DESC_NT = ["SIZE", "COLOR", "MAT", "SHAPE"]

Rule = namedtuple("Rule", ["lhs", "rhs"])


def _build_rules():
    rules = [
        Rule("T", ("S", "C")),
        Rule("C", ()),
        Rule("C", ("S", "C")),
        Rule("S", ("R", "A")),
        Rule("R", ("SIZE", "COLOR", "MAT", "SHAPE")),
        Rule("A", ("'color'", "COLOR")),
        Rule("A", ("'shape'", "SHAPE")),
        Rule("A", ("'size'", "SIZE")),
        Rule("A", ("'material'", "MAT")),
        Rule("A", ("'position'", "DIR", "N")),
        Rule("A", ("'position'", "'out'")),
        Rule("A", ("'position'", "'in'")),
    ]
    for lhs, values in [
        ("SIZE", SIZES),
        ("COLOR", COLORS),
        ("MAT", MATERIALS),
        ("SHAPE", SHAPES),
        ("DIR", DIRECTIONS),
        ("N", STEPS),
    ]:
        rules.extend(Rule(lhs, ("'{}'".format(v),)) for v in values)
    return rules


RULES = _build_rules()
# GVAE pads every derivation to the same length with a no-op rule.
NOOP = len(RULES)
K = NOOP + 1

RULES_OF = {
    nt: [i for i, r in enumerate(RULES) if r.lhs == nt] for nt in NONTERMINALS
}


def is_terminal(symbol):
    return symbol.startswith("'")


def terminal(symbol):
    return symbol[1:-1]


def value_rule(nt, value):
    """Index of the rule NT -> 'value'."""
    for i in RULES_OF[nt]:
        if RULES[i].rhs == ("'{}'".format(value),):
            return i
    raise KeyError((nt, value))


class ParseError(ValueError):
    pass


Tree = namedtuple("Tree", ["symbol", "rule", "children"])
"""A parse tree node: a non-terminal with the index of the rule expanding it
and its children, or a terminal (rule None, no children)."""


def program_tokens(program):
    """Terminal string of a program [(descriptor, action)]."""
    tokens = []
    for desc, action in program:
        tokens.extend(desc)
        tokens.extend(action.split("."))
    return tokens


def parse(tokens):
    """Parse a terminal string into a tree rooted at T. Raises ParseError."""
    tokens = list(tokens)
    pos = [0]

    def peek(offset=0):
        i = pos[0] + offset
        return tokens[i] if i < len(tokens) else None

    def expand(nt, rule):
        children = []
        for symbol in RULES[rule].rhs:
            if is_terminal(symbol):
                if peek() != terminal(symbol):
                    raise ParseError(
                        "expected {} at {}, got {}".format(
                            symbol, pos[0], peek()
                        )
                    )
                pos[0] += 1
                children.append(Tree(symbol, None, ()))
            else:
                children.append(node(symbol))
        return Tree(nt, rule, tuple(children))

    def node(nt):
        if nt == "C":
            return expand(nt, RULES_OF["C"][0 if peek() is None else 1])
        if nt == "A":
            head, second = peek(), peek(1)
            for rule in RULES_OF["A"]:
                rhs = RULES[rule].rhs
                if terminal(rhs[0]) != head:
                    continue
                if head != "position":
                    return expand(nt, rule)
                if len(rhs) == 2 and terminal(rhs[1]) == second:
                    return expand(nt, rule)
                if len(rhs) == 3 and second in DIRECTIONS:
                    return expand(nt, rule)
            raise ParseError("no action at {}: {}".format(pos[0], head))
        for rule in RULES_OF[nt]:
            rhs = RULES[rule].rhs
            if len(rhs) == 1 and is_terminal(rhs[0]):
                if terminal(rhs[0]) == peek():
                    return expand(nt, rule)
            else:
                return expand(nt, rule)
        raise ParseError("no {} at {}: {}".format(nt, pos[0], peek()))

    tree = node("T")
    if pos[0] != len(tokens):
        raise ParseError("trailing tokens from {}".format(pos[0]))
    return tree


def derivation(tree):
    """Leftmost derivation (rule indices) of a parse tree."""
    rules = []

    def visit(node):
        if node.rule is not None:
            rules.append(node.rule)
            for child in node.children:
                visit(child)

    visit(tree)
    return rules


def tree_tokens(tree):
    """Terminal string (yield) of a parse tree."""
    if tree.rule is None:
        return [terminal(tree.symbol)]
    tokens = []
    for child in tree.children:
        tokens.extend(tree_tokens(child))
    return tokens


def derive(rules):
    """Terminal string of a leftmost derivation. Raises ParseError."""
    stack = ["T"]
    tokens = []
    for rule in rules:
        if rule == NOOP:
            if stack:
                raise ParseError("no-op before the derivation ends")
            continue
        while stack and is_terminal(stack[-1]):
            tokens.append(terminal(stack.pop()))
        if not stack or RULES[rule].lhs != stack[-1]:
            raise ParseError("rule {} does not expand {}".format(rule, stack))
        stack.pop()
        stack.extend(reversed(RULES[rule].rhs))
    while stack and is_terminal(stack[-1]):
        tokens.append(terminal(stack.pop()))
    if stack:
        raise ParseError("derivation is incomplete: {}".format(stack))
    return tokens


def tokens_program(tokens):
    """Program [(descriptor, action)] of a terminal string."""
    program = []
    i = 0
    while i < len(tokens):
        desc = tuple(tokens[i : i + 4])
        attr = tokens[i + 4]
        if attr != "position":
            n = 2
        elif tokens[i + 5] in ("in", "out"):
            n = 2
        else:
            n = 3
        program.append((desc, ".".join(tokens[i + 4 : i + 4 + n])))
        i += 4 + n
    return program
