"""Masked leftmost derivation of a transformation (the GVAE decoder's mask).

The decoder pops the leftmost non-terminal, takes the mask of its rules,
picks one rule (here: given, or sampled at random) and pushes its right-hand
side. The mask comes from an executor (executor.py): at the start of each
step it computes the set V of valid steps (R, A), and each rule of the step is
allowed only if it can still complete a step in V.

With use_init_desc (the default) V is computed in every scene given, from
each scene's init_desc. Without it, only the program decoded so far
constrains V (ProgramExecutor), so the mask allows a superset of the
programs.
"""
import random

from .cfg import K, NOOP, RULES, RULES_OF, is_terminal, terminal
from .executor import Executor, ProgramExecutor
from .geometry import Geometry

MAX_STEPS = 4
# The value terminal of each rule NT -> 'value'.
RULE_VALUE = {
    i: terminal(r.rhs[0])
    for i, r in enumerate(RULES)
    if len(r.rhs) == 1 and is_terminal(r.rhs[0])
}
# The terminals of each A rule, e.g. ("position", "out").
A_TERMINALS = [
    (rule, tuple(terminal(s) for s in RULES[rule].rhs if is_terminal(s)))
    for rule in RULES_OF["A"]
]


class MaskError(ValueError):
    pass


class MaskedDecoder:
    def __init__(
        self,
        members=None,
        geometry=None,
        allow_repeated_moves=False,
        max_steps=MAX_STEPS,
        use_init_desc=True,
    ):
        """members: a list of one or more scenes, one init_desc each (a list of
        stage-0 objects, each {"size", "color", "material", "shape",
        "position": [x, y]}). A step must be valid in every scene given: one
        scene masks a single pair, all members of a group require validity
        for the whole group. Needed with use_init_desc, ignored without.
        geometry: Geometry(), read from properties.json, by default."""
        geometry = geometry or Geometry()
        if use_init_desc:
            if members is None:
                raise ValueError("use_init_desc needs the members' init_desc")
            self.executor = Executor(members, geometry, allow_repeated_moves)
        else:
            self.executor = ProgramExecutor(geometry, allow_repeated_moves)
        self.use_init_desc = use_init_desc
        self.max_steps = max_steps
        # Stack of (symbol, role); role is "R" or "A" for the symbols of a
        # step's descriptor or action.
        self.stack = [("T", None)]
        self.rules = []
        self.tokens = []
        self._desc = []
        self._action = []
        self._reset_step()

    def _reset_step(self):
        # Drop V and what is derived from it; they are rebuilt lazily.
        self._valid_map = None
        self._desc_prefixes = None
        self._action_prefixes = {}
        self._mask = None

    @property
    def done(self):
        return not self.stack

    @property
    def valid_map(self):
        """V for the next step: {descriptor: set(actions)}."""
        if self._valid_map is None:
            self._valid_map = self.executor.valid_steps()
        return self._valid_map

    def _desc_prefix_set(self):
        # Every prefix of a descriptor in V, so that R's rules test one
        # membership each.
        if self._desc_prefixes is None:
            self._desc_prefixes = set(
                desc[:n] for desc in self.valid_map for n in range(1, 5)
            )
        return self._desc_prefixes

    def _action_prefix_set(self, desc):
        # Every action of V for desc and its prefixes of whole parts, as
        # text: "position", "position.front", "position.front.1".
        prefixes = self._action_prefixes.get(desc)
        if prefixes is None:
            prefixes = set()
            for action in self.valid_map.get(desc, ()):
                parts = action.split(".")
                for n in range(1, len(parts) + 1):
                    prefixes.add(".".join(parts[:n]))
            self._action_prefixes[desc] = prefixes
        return prefixes

    @property
    def valid(self):
        """V for the next step as a set of (descriptor, action)."""
        return set(
            (desc, action)
            for desc, actions in self.valid_map.items()
            for action in actions
        )

    def _actions(self):
        """Actions of V for the chosen descriptor that agree with the action
        so far."""
        actions = self.valid_map.get(tuple(self._desc), ())
        if not self._action:
            return list(actions)
        prefix = ".".join(self._action)
        return [
            a for a in actions if a == prefix or a.startswith(prefix + ".")
        ]

    def mask(self):
        """Allowed rules (list of K bools) for the top of the stack."""
        # Cached until the next apply(), which also needs it.
        if self._mask is None:
            self._mask = self._compute_mask()
        return list(self._mask)

    def _compute_mask(self):
        allowed = [False] * K
        if self.done:
            allowed[NOOP] = True
            return allowed
        symbol, role = self.stack[-1]
        if symbol == "T":
            allowed[RULES_OF["T"][0]] = bool(self.valid_map)
        elif symbol == "C":
            stop, more = RULES_OF["C"]
            allowed[stop] = True
            allowed[more] = len(self.executor.steps) < self.max_steps and bool(
                self.valid_map
            )
        elif symbol in ("S", "R"):
            allowed[RULES_OF[symbol][0]] = True
        elif symbol == "A":
            actions = [a.split(".") for a in self._actions()]
            for rule, rhs in A_TERMINALS:
                for parts in actions:
                    if parts[0] != rhs[0]:
                        continue
                    if len(rhs) == 2 and parts[1] != rhs[1]:
                        continue
                    if len(rhs) == 1 and parts[1] in ("in", "out"):
                        continue
                    allowed[rule] = True
                    break
        elif role == "R":
            prefixes = self._desc_prefix_set()
            desc = tuple(self._desc)
            for rule in RULES_OF[symbol]:
                allowed[rule] = desc + (RULE_VALUE[rule],) in prefixes
        else:
            # A rule is allowed if the action so far plus its value is an
            # action of V, or the start of one.
            prefixes = self._action_prefix_set(tuple(self._desc))
            for rule in RULES_OF[symbol]:
                action = ".".join(self._action + [RULE_VALUE[rule]])
                allowed[rule] = action in prefixes
        return allowed

    def apply(self, rule):
        """Expand the top of the stack with rule. Raises MaskError if the
        mask does not allow it."""
        if self._mask is None:
            self._mask = self._compute_mask()
        if not self._mask[rule]:
            raise MaskError(
                "rule {} not allowed at {} (steps so far {})".format(
                    rule,
                    self.stack[-1] if self.stack else None,
                    self.executor.steps,
                )
            )
        self._mask = None
        self.rules.append(rule)
        if rule == NOOP:
            return
        symbol, role = self.stack.pop()
        if symbol == "R":
            role = "R"
        elif symbol == "A":
            role = "A"
        for child in reversed(RULES[rule].rhs):
            if is_terminal(child):
                self.stack.append((child, None))
            else:
                self.stack.append((child, role))
        # Terminals on top are emitted right away.
        while self.stack and is_terminal(self.stack[-1][0]):
            token = terminal(self.stack.pop()[0])
            self.tokens.append(token)
            if role == "R":
                self._desc.append(token)
            elif role == "A":
                self._action.append(token)
        self._finish_step()

    def _finish_step(self):
        """Apply the step to the executor once its action is complete."""
        if not self._action:
            return
        top = self.stack[-1][1] if self.stack else None
        if top == "A":
            return
        desc, action = tuple(self._desc), ".".join(self._action)
        if action not in self.valid_map.get(desc, ()):
            raise MaskError("step {} {} is not valid".format(desc, action))
        self.executor.apply(desc, action)
        self._reset_step()
        self._desc = []
        self._action = []

    def decode(self, rules):
        """Apply a whole derivation; returns self."""
        for rule in rules:
            self.apply(rule)
        return self

    def sample(self, rng=None, stop_prob=0.3):
        """Sample a derivation under the mask; returns the rules.

        C -> eps is taken with probability stop_prob when both are allowed.
        """
        rng = rng or random.Random()
        while not self.done:
            allowed = [i for i, ok in enumerate(self.mask()) if ok]
            if not allowed:
                raise MaskError("dead end at {}".format(self.stack[-1]))
            if self.stack[-1][0] == "C" and len(allowed) == 2:
                rule = allowed[0] if rng.random() < stop_prob else allowed[1]
            else:
                rule = rng.choice(allowed)
            self.apply(rule)
        return self.rules


def accepts(
    members,
    rules,
    geometry=None,
    allow_repeated_moves=False,
    use_init_desc=True,
):
    """Whether the mask allows every rule of a derivation, and where not.

    Returns (True, None) or (False, message).
    """
    decoder = MaskedDecoder(
        members,
        geometry,
        allow_repeated_moves,
        use_init_desc=use_init_desc,
    )
    try:
        decoder.decode(rules)
    except MaskError as e:
        return False, str(e)
    if not decoder.done:
        return False, "derivation is incomplete"
    return True, None
