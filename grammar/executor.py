"""Executors behind the decoder's mask.

Executor keeps one state per member of a transformation, built from that
member's init_desc (the stage-0 objects: size, color, material, shape, grid
position), and decides which steps (R, A) are valid in every member.
ProgramExecutor works without init_desc: it judges a step only by the program
decoded so far (see its docstring for what that still covers).

Positions are sets of cells. A plain object has one cell. After 'in' it is the
set P(o) of candidate cells, after 'out' the set of possible destinations.
Such sets are permissive: they block a step only if every cell blocks it. The
candidate cells of 'in' are also reserved: other objects must avoid all of
them, because the generator enforces exactly that.
"""
from .cfg import COLORS, DIRECTIONS, MATERIALS, SHAPES, SIZES, STEPS

DESC_ATTRS = ["size", "color", "material", "shape"]
ATTR_VALUES = {
    "size": SIZES,
    "color": COLORS,
    "material": MATERIALS,
    "shape": SHAPES,
}

UNTOUCHED, TOUCHED, MOVED_IN, MOVED_OUT = (
    "untouched",
    "touched",
    "moved-in",
    "moved-out",
)


def descriptor(obj):
    """Stage-0 descriptor of an object description, as R spells it."""
    return tuple(obj[attr] for attr in DESC_ATTRS)


def key_text(desc):
    """Descriptor as the generator's obj_key, e.g. large_red_metal_cube."""
    return "_".join(desc)


def actions():
    """Every action A as text, e.g. color.red, position.front.1."""
    result = []
    for attr in ["color", "shape", "size", "material"]:
        result.extend("{}.{}".format(attr, v) for v in ATTR_VALUES[attr])
    for direction in DIRECTIONS:
        result.extend("position.{}.{}".format(direction, n) for n in STEPS)
    result.extend(["position.out", "position.in"])
    return result


ACTIONS = actions()


class Obj:
    def __init__(self, desc):
        self.desc = descriptor(desc)
        self.start = tuple(desc["position"])
        self.size = desc["size"]
        self.cells = [self.start]
        self.edited = set()
        self.status = UNTOUCHED
        self.moved = False


class Member:
    """Executor state of one member."""

    def __init__(self, objects, geometry, allow_repeated_moves=False):
        self.geometry = geometry
        self.allow_repeated_moves = allow_repeated_moves
        self.objects = [Obj(obj) for obj in objects]
        # Index of the moved-in object -> its candidate cells.
        self.reserved = {}

    def find(self, desc):
        """Index of the only object with this stage-0 descriptor, or None."""
        found = [i for i, o in enumerate(self.objects) if o.desc == desc]
        return found[0] if len(found) == 1 else None

    def visible_now(self, o):
        if o.status == MOVED_IN:
            return True
        if o.status == MOVED_OUT:
            return False
        return self.geometry.visible(o.cells[0])

    def blocked(self, idx, cell, size, reserved=True):
        """Whether object idx would collide at cell with the given size."""
        geometry = self.geometry
        for j, other in enumerate(self.objects):
            if j == idx:
                continue
            if all(
                geometry.overlaps(cell, size, c, other.size)
                for c in other.cells
            ):
                return True
        if reserved:
            for owner, cells in self.reserved.items():
                if owner == idx:
                    continue
                owner_size = self.objects[owner].size
                if any(
                    geometry.overlaps(cell, size, c, owner_size) for c in cells
                ):
                    return True
        return False

    def free_moves(self, idx, visible):
        """Moves of object idx from its cell whose destination is on the grid,
        has the given visibility and is free."""
        o = self.objects[idx]
        result = set()
        for move in self.geometry.moves:
            cell = self.geometry.destination(o.cells[0], move)
            if (
                self.geometry.on_grid(cell)
                and self.geometry.visible(cell) == visible
                and not self.blocked(idx, cell, o.size)
            ):
                result.add(move)
        return result

    def valid(self, idx, action):
        """Whether object idx can undergo action here, except the 'in'/'out'
        common-move test, which spans all members (see moves_for)."""
        o = self.objects[idx]
        if o.status == MOVED_OUT:
            return False
        parts = action.split(".")
        attr = parts[0]
        if attr != "position":
            value = parts[1]
            if not self.visible_now(o) or attr in o.edited:
                return False
            if value == o.desc[DESC_ATTRS.index(attr)]:
                return False
            if attr == "size" and self.geometry.is_bigger(value, o.size):
                return any(
                    not self.blocked(idx, cell, value) for cell in o.cells
                )
            return True
        if parts[1] in ("in", "out"):
            return o.status == UNTOUCHED and self.geometry.visible(
                o.start
            ) == (parts[1] == "out")
        # Inner move.
        if o.status == MOVED_IN or not self.visible_now(o):
            return False
        if o.moved and not self.allow_repeated_moves:
            return False
        cell = self.geometry.destination(o.cells[0], (parts[1], parts[2]))
        return (
            self.geometry.on_grid(cell)
            and self.geometry.visible(cell)
            and cell != o.start
            and not self.blocked(idx, cell, o.size)
        )

    def moves_for(self, idx, kind):
        """Feasible (direction, n) of an 'in' or 'out' of object idx."""
        return self.free_moves(idx, visible=(kind == "in"))

    def apply(self, idx, action):
        o = self.objects[idx]
        parts = action.split(".")
        attr = parts[0]
        if attr != "position":
            o.edited.add(attr)
            if attr == "size":
                o.size = parts[1]
            o.status = MOVED_IN if o.status == MOVED_IN else TOUCHED
        elif parts[1] == "in":
            # Candidates: every feasible in-move, judged against the objects
            # only, as the generator does.
            cells = []
            for move in self.geometry.moves:
                cell = self.geometry.destination(o.start, move)
                if (
                    self.geometry.on_grid(cell)
                    and self.geometry.visible(cell)
                    and not self.blocked(idx, cell, o.size, reserved=False)
                ):
                    cells.append(cell)
            o.cells = sorted(set(cells))
            o.status = MOVED_IN
            o.moved = True
            self.reserved[idx] = o.cells
        elif parts[1] == "out":
            o.cells = sorted(
                self.geometry.destination(o.cells[0], move)
                for move in self.free_moves(idx, visible=False)
            )
            o.status = MOVED_OUT
            o.moved = True
        else:
            o.cells = [
                self.geometry.destination(o.cells[0], (parts[1], parts[2]))
            ]
            o.status = TOUCHED
            o.moved = True


class StepRules:
    """The order rule and the set V, common to both executors.

    Subclasses provide descriptors(), step_valid(desc, action) and
    apply(desc, action), which must append to self.steps.
    """

    def __init__(self):
        # Steps applied so far: (descriptor, action).
        self.steps = []

    def in_order(self, desc, action):
        """The order rule: (R, pair text) >= the previous step's, unless the
        previous step was this object's own 'in'."""
        if not self.steps:
            return True
        prev_desc, prev_action = self.steps[-1]
        if prev_desc == desc and prev_action == "position.in":
            return True
        return (key_text(desc), action) >= (key_text(prev_desc), prev_action)

    def valid_steps(self):
        """V: valid steps for the next step, as {descriptor: set(actions)}
        (descriptors without a valid action are left out)."""
        result = {}
        for desc in self.descriptors():
            actions = set(
                action
                for action in ACTIONS
                if self.in_order(desc, action)
                and self.step_valid(desc, action)
            )
            if actions:
                result[desc] = actions
        return result


class Executor(StepRules):
    """Executor states of all members of one transformation."""

    def __init__(self, members, geometry, allow_repeated_moves=False):
        super().__init__()
        self.members = [
            Member(objects, geometry, allow_repeated_moves)
            for objects in members
        ]

    def descriptors(self):
        """Descriptors that name exactly one object in every member."""
        candidates = set(o.desc for o in self.members[0].objects)
        return sorted(
            desc
            for desc in candidates
            if all(m.find(desc) is not None for m in self.members)
        )

    def step_valid(self, desc, action):
        """Validity of the step (R, A) in every member, without the order."""
        indices = [m.find(desc) for m in self.members]
        if any(i is None for i in indices):
            return False
        if not all(m.valid(i, action) for m, i in zip(self.members, indices)):
            return False
        parts = action.split(".")
        if parts[0] == "position" and parts[1] in ("in", "out"):
            common = None
            for m, i in zip(self.members, indices):
                moves = m.moves_for(i, parts[1])
                common = moves if common is None else common & moves
            return bool(common)
        return True

    def apply(self, desc, action):
        for m in self.members:
            m.apply(m.find(desc), action)
        self.steps.append((desc, action))


class History:
    """What the program so far says about one object."""

    def __init__(self):
        self.edited = set()
        self.status = UNTOUCHED
        self.moved = False
        # Sum of the object's moves, in grid units.
        self.offset = (0, 0)


class ProgramExecutor(StepRules):
    """Executor without init_desc: it knows only the program so far.

    Kept: the order rule; one edit per attribute; an edit's value differs
    from R's own value (R is the stage-0 descriptor); 'in' and 'out' only for
    an untouched object; nothing after 'out'; no move after 'in'; one move per
    object (or, with allow_repeated_moves, no moves adding up to zero).
    Lost: everything that needs the scene. R need not exist or be unique,
    visibility is unknown (any untouched object may edit, move, go out or
    come in), and nothing geometric is checked (grid, overlaps, growth,
    common in/out moves, reserved cells).
    """

    def __init__(self, geometry, allow_repeated_moves=False):
        super().__init__()
        self.geometry = geometry
        self.allow_repeated_moves = allow_repeated_moves
        self.history = {}
        self._descriptors = [
            (size, color, material, shape)
            for size in SIZES
            for color in COLORS
            for material in MATERIALS
            for shape in SHAPES
        ]

    def descriptors(self):
        return self._descriptors

    def _history(self, desc):
        return self.history.get(desc) or History()

    def step_valid(self, desc, action):
        h = self._history(desc)
        if h.status == MOVED_OUT:
            return False
        parts = action.split(".")
        attr = parts[0]
        if attr != "position":
            if attr in h.edited:
                return False
            return parts[1] != desc[DESC_ATTRS.index(attr)]
        if parts[1] in ("in", "out"):
            return h.status == UNTOUCHED
        if h.status == MOVED_IN:
            return False
        if h.moved and not self.allow_repeated_moves:
            return False
        dx, dy = self.geometry.moves[(parts[1], parts[2])]
        return (h.offset[0] + dx, h.offset[1] + dy) != (0, 0)

    def apply(self, desc, action):
        h = self.history.setdefault(desc, History())
        parts = action.split(".")
        attr = parts[0]
        if attr != "position":
            h.edited.add(attr)
            if h.status == UNTOUCHED:
                h.status = TOUCHED
        elif parts[1] == "in":
            h.status, h.moved = MOVED_IN, True
        elif parts[1] == "out":
            h.status, h.moved = MOVED_OUT, True
        else:
            dx, dy = self.geometry.moves[(parts[1], parts[2])]
            h.offset = (h.offset[0] + dx, h.offset[1] + dy)
            h.status, h.moved = TOUCHED, True
        self.steps.append((desc, action))
