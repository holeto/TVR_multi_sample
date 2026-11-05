"""Executors behind the decoder's mask.

Executor keeps one state per scene given, built from that scene's init_desc
(the stage-0 objects: size, color, material, shape, grid position), and
decides which steps (R, A) are valid in every scene given. A scene is one
pair's own scene or every member of a group.
ProgramExecutor works without init_desc: it judges a step only by the program
decoded so far (see its docstring for what that still covers).

Positions are sets of cells. A plain object has one cell. After 'in' or
'out' it is the set of possible destinations of a move of that kind. Such
sets are permissive: they block a step only if every cell blocks it. The
candidate cells P(o) of 'in' are also reserved: other objects must avoid all
of them, because the generator enforces exactly that.

Two collision tests are kept apart, as in the generator (see geometry.py).
Moves, including the move of 'in' itself, and size changes use
Geometry.collides (exact contact is free), so the destinations of 'in' and
'out' (Obj.cells) are judged by it. P(o) and the reservation use
Geometry.overlaps, as Renderer.in_candidates does; P(o) may therefore miss a
destination at exact contact. P(o) sees moved objects at their float-test
cells (Obj.float_cells), so the reserved sets stay what they were before
collides existed.
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
# Masks test every action in every scene, so each is split only once.
ACTION_PARTS = {action: tuple(action.split(".")) for action in ACTIONS}
DESC_INDEX = {attr: i for i, attr in enumerate(DESC_ATTRS)}


def action_parts(action):
    return ACTION_PARTS.get(action) or tuple(action.split("."))


class Obj:
    def __init__(self, desc):
        self.desc = descriptor(desc)
        self.start = tuple(desc["position"])
        self.size = desc["size"]
        self.cells = [self.start]
        # Only for a moved-in or moved-out object: its destinations under the
        # float test (cells holds those under the exact one, a superset); for
        # 'in' they are P(o). None otherwise.
        self.float_cells = None
        self.edited = set()
        self.status = UNTOUCHED
        self.moved = False


class Member:
    """Executor state of one scene."""

    def __init__(self, objects, geometry, allow_repeated_moves=False):
        self.geometry = geometry
        self.allow_repeated_moves = allow_repeated_moves
        self.objects = [Obj(obj) for obj in objects]
        # Index of the moved-in object -> its candidate cells.
        self.reserved = {}
        # Stage-0 descriptors never change; ambiguous ones map to None.
        self._index = {}
        for i, o in enumerate(self.objects):
            self._index[o.desc] = None if o.desc in self._index else i
        self._obstacle_cache = {}

    def find(self, desc):
        """Index of the only object with this stage-0 descriptor, or None."""
        return self._index.get(desc)

    def visible_now(self, o):
        if o.status == MOVED_IN:
            return True
        if o.status == MOVED_OUT:
            return False
        return self.geometry.visible(o.cells[0])

    def blocked(self, idx, cell, size, reserved=True, exact=True):
        """Whether object idx would collide at cell with the given size.
        exact=False uses the generator's float test, as for 'in' candidates;
        reserved cells are always judged by it."""
        geometry = self.geometry
        if exact and geometry.cell_sizes is not None:
            # Geometry.collides inlined, since this loop dominates the time
            # of a mask.
            x, y = cell
            for limit, cells in self._obstacles(idx, size):
                for cx, cy in cells:
                    if (x - cx) * (x - cx) + (y - cy) * (y - cy) >= limit:
                        break
                else:
                    return True
        else:
            test = geometry.collides if exact else geometry.overlaps
            for j, other in enumerate(self.objects):
                if j == idx:
                    continue
                cells = other.cells
                if not exact and other.float_cells is not None:
                    cells = other.float_cells
                if all(test(cell, size, c, other.size) for c in cells):
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

    def _obstacles(self, idx, size):
        # The other objects as [(limit, cells)] for the exact test of
        # object idx with the given size: a cell collides with an obstacle
        # if its squared distance in cells to every one of its cells is
        # below limit = (r_a + r_b + gap)^2. Cached until the next apply.
        key = (idx, size)
        obstacles = self._obstacle_cache.get(key)
        if obstacles is None:
            cell_sizes = self.geometry.cell_sizes
            reach = cell_sizes[size] + self.geometry.cell_gap
            obstacles = [
                ((reach + cell_sizes[other.size]) ** 2, other.cells)
                for j, other in enumerate(self.objects)
                if j != idx
            ]
            self._obstacle_cache[key] = obstacles
        return obstacles

    def free_moves(self, idx, visible, exact=True):
        """Moves of object idx from its cell whose destination is on the grid,
        has the given visibility and is free (see blocked for exact)."""
        o = self.objects[idx]
        result = set()
        for move in self.geometry.moves:
            cell = self.geometry.destination(o.cells[0], move)
            if (
                self.geometry.on_grid(cell)
                and self.geometry.visible(cell) == visible
                and not self.blocked(idx, cell, o.size, exact=exact)
            ):
                result.add(move)
        return result

    def valid(self, idx, action):
        """Whether object idx can undergo action here, except the 'in'/'out'
        common-move test, which spans all scenes given (see moves_for)."""
        return bool(self.valid_actions(idx, [action]))

    def valid_actions(self, idx, actions):
        """The actions, in their order, that valid() allows for object idx.
        Judges them together, so the object's state is read once."""
        o = self.objects[idx]
        if o.status == MOVED_OUT:
            return []
        geometry = self.geometry
        visible = self.visible_now(o)
        can_move = (
            o.status != MOVED_IN
            and visible
            and (not o.moved or self.allow_repeated_moves)
        )
        result = []
        for action in actions:
            parts = action_parts(action)
            attr = parts[0]
            if attr != "position":
                value = parts[1]
                if not visible or attr in o.edited:
                    continue
                if value == o.desc[DESC_INDEX[attr]]:
                    continue
                if attr == "size" and geometry.is_bigger(value, o.size):
                    ok = any(
                        not self.blocked(idx, cell, value) for cell in o.cells
                    )
                else:
                    ok = True
            elif parts[1] in ("in", "out"):
                ok = o.status == UNTOUCHED and geometry.visible(o.start) == (
                    parts[1] == "out"
                )
            elif not can_move:
                ok = False
            else:
                # Inner move.
                cell = geometry.destination(o.cells[0], (parts[1], parts[2]))
                ok = (
                    geometry.on_grid(cell)
                    and geometry.visible(cell)
                    and cell != o.start
                    and not self.blocked(idx, cell, o.size)
                )
            if ok:
                result.append(action)
        return result

    def free_in_cell(self, idx, move, exact):
        # Whether the in-move of object idx from its start ends on the grid,
        # in view and clear of the other objects (reserved cells ignored).
        o = self.objects[idx]
        cell = self.geometry.destination(o.start, move)
        return (
            self.geometry.on_grid(cell)
            and self.geometry.visible(cell)
            and not self.blocked(idx, cell, o.size, reserved=False, exact=exact)
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
            # The generator accepts the move by the exact test, but records
            # and reserves only the cells free by the float test, P(o). Both
            # are judged against the objects only, as in_candidates does.
            exact_cells, float_cells = (
                sorted(
                    set(
                        self.geometry.destination(o.start, move)
                        for move in self.geometry.moves
                        if self.free_in_cell(idx, move, exact=flag)
                    )
                )
                for flag in (True, False)
            )
            o.cells, o.float_cells = exact_cells, float_cells
            o.status = MOVED_IN
            o.moved = True
            self.reserved[idx] = float_cells
        elif parts[1] == "out":
            start = o.cells[0]
            exact_cells, float_cells = (
                sorted(
                    self.geometry.destination(start, move)
                    for move in self.free_moves(idx, False, exact=flag)
                )
                for flag in (True, False)
            )
            o.cells, o.float_cells = exact_cells, float_cells
            o.status = MOVED_OUT
            o.moved = True
        else:
            o.cells = [
                self.geometry.destination(o.cells[0], (parts[1], parts[2]))
            ]
            o.status = TOUCHED
            o.moved = True
        # The step changed a cell or size; the tests above used the old ones.
        self._obstacle_cache = {}


class StepRules:
    """The order rule and the set V, common to both executors.

    Subclasses provide descriptors(), step_valid(desc, action) and
    apply(desc, action), which must append to self.steps; they may override
    valid_actions(desc, actions) to judge a descriptor's actions together.
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

    def actions_in_order(self, desc):
        """The actions of ACTIONS that in_order allows for desc, in order.
        The pairs compare by descriptor text first, so only the previous
        step's own descriptor needs a test per action."""
        if not self.steps:
            return ACTIONS
        prev_desc, prev_action = self.steps[-1]
        if prev_desc == desc and prev_action == "position.in":
            return ACTIONS
        key, prev_key = key_text(desc), key_text(prev_desc)
        if key != prev_key:
            return ACTIONS if key > prev_key else []
        return [action for action in ACTIONS if action >= prev_action]

    def valid_actions(self, desc, actions):
        """The actions, in their order, that step_valid allows for desc."""
        return [action for action in actions if self.step_valid(desc, action)]

    def valid_steps(self):
        """V: valid steps for the next step, as {descriptor: set(actions)}
        (descriptors without a valid action are left out)."""
        result = {}
        for desc in self.descriptors():
            actions = self.actions_in_order(desc)
            if actions:
                actions = self.valid_actions(desc, actions)
            if actions:
                result[desc] = set(actions)
        return result


class Executor(StepRules):
    """Executor states of the scenes in `members` (one scene or a group)."""

    def __init__(self, members, geometry, allow_repeated_moves=False):
        super().__init__()
        self.members = [
            Member(objects, geometry, allow_repeated_moves)
            for objects in members
        ]
        # R names stage-0 objects, so the descriptors and their object in
        # each scene are fixed: descriptor -> [index per scene].
        self._indices = {}
        for desc in set(o.desc for o in self.members[0].objects):
            indices = [m.find(desc) for m in self.members]
            if all(i is not None for i in indices):
                self._indices[desc] = indices
        self._descriptors = sorted(self._indices)

    def descriptors(self):
        """Descriptors that name exactly one object in every scene given."""
        return list(self._descriptors)

    def step_valid(self, desc, action):
        """Validity of the step (R, A) in every scene given, ignoring order."""
        return bool(self.valid_actions(desc, [action]))

    def valid_actions(self, desc, actions):
        indices = self._indices.get(desc)
        if indices is None:
            return []
        for m, i in zip(self.members, indices):
            actions = m.valid_actions(i, actions)
            if not actions:
                return []
        result = []
        for action in actions:
            attr, kind = action_parts(action)[:2]
            if attr == "position" and kind in ("in", "out"):
                if not self.common_move(indices, kind):
                    continue
            result.append(action)
        return result

    def common_move(self, indices, kind):
        """Whether one (direction, n) of an 'in' or 'out' is feasible in
        every scene, for the object indices[k] of scene k."""
        common = None
        for m, i in zip(self.members, indices):
            moves = m.moves_for(i, kind)
            common = moves if common is None else common & moves
            if not common:
                return False
        return common is not None

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
        self._moves_key = tuple(sorted(geometry.moves.items()))
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

    # Valid actions of a descriptor the program has not named yet, as
    # (list, set), by (moves, descriptor); shared by all instances.
    _unnamed = {}

    def valid_actions(self, desc, actions):
        if desc in self.history:
            return super().valid_actions(desc, actions)
        # A fresh History judges the step, so the result depends only on
        # the descriptor and the geometry's moves.
        key = (self._moves_key, desc)
        cached = ProgramExecutor._unnamed.get(key)
        if cached is None:
            valid = super().valid_actions(desc, ACTIONS)
            cached = ProgramExecutor._unnamed[key] = (valid, set(valid))
        if actions is ACTIONS:
            return cached[0]
        return [action for action in actions if action in cached[1]]

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
