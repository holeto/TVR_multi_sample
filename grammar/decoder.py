"""Masked leftmost derivation of a transformation (the GVAE decoder's mask).

The decoder pops the leftmost non-terminal, takes the mask of its rules,
picks one rule (here: given, or sampled at random) and pushes its right-hand
side. The mask comes from an executor (executor.py): at the start of each
step it computes the set V of valid steps (R, A), and each rule of the step is
allowed only if it can still complete a step in V.

With use_init_desc (the default) V is computed in every member from its
init_desc. Without it, only the program decoded so far constrains V
(ProgramExecutor), so the mask allows a superset of the programs.
"""
import random

from .cfg import K, NOOP, RULES, RULES_OF, is_terminal, terminal
from .executor import Executor, ProgramExecutor
from .geometry import Geometry

MAX_STEPS = 4


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
        """members: one init_desc per member (a list of stage-0 objects, each
        {"size", "color", "material", "shape", "position": [x, y]}); needed
        with use_init_desc, ignored without. geometry: Geometry(), read from
        properties.json, by default."""
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
        self._valid_map = None
        self._desc = []
        self._action = []

    @property
    def done(self):
        return not self.stack

    @property
    def valid_map(self):
        """V for the next step: {descriptor: set(actions)}."""
        if self._valid_map is None:
            self._valid_map = self.executor.valid_steps()
        return self._valid_map

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
            actions = self._actions()
            for rule in RULES_OF["A"]:
                rhs = [terminal(s) for s in RULES[rule].rhs if is_terminal(s)]
                for a in actions:
                    parts = a.split(".")
                    if parts[0] != rhs[0]:
                        continue
                    if len(rhs) == 2 and parts[1] != rhs[1]:
                        continue
                    if len(rhs) == 1 and parts[1] in ("in", "out"):
                        continue
                    allowed[rule] = True
                    break
        elif role == "R":
            for rule in RULES_OF[symbol]:
                prefix = tuple(self._desc) + (terminal(RULES[rule].rhs[0]),)
                allowed[rule] = any(
                    d[: len(prefix)] == prefix for d in self.valid_map
                )
        else:
            actions = self._actions()
            for rule in RULES_OF[symbol]:
                action = ".".join(
                    self._action + [terminal(RULES[rule].rhs[0])]
                )
                allowed[rule] = any(
                    a == action or a.startswith(action + ".") for a in actions
                )
        return allowed

    def apply(self, rule):
        """Expand the top of the stack with rule. Raises MaskError if the
        mask does not allow it."""
        if not self.mask()[rule]:
            raise MaskError(
                "rule {} not allowed at {} (steps so far {})".format(
                    rule,
                    self.stack[-1] if self.stack else None,
                    self.executor.steps,
                )
            )
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
        self._valid_map = None
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
