"""Scene geometry of TRANCE, read from properties.json.

The numbers and formulas follow the generator (trance/option.py and
trance/group.py), so that the executor judges overlaps exactly as it does:
positions are integer grid cells, footprints are compared in metres.

The generator has two collision tests. trance/group.py compares float64
distances in metres (overlaps). Moves and size changes are checked by
trance/scene.py on Blender's float32 locations, where two objects at exactly
the minimum distance are not overlapped; collides tests that in integer cells.
"""
import json
import math
import os
import warnings

PROPERTIES_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "trance",
    "resource",
    "properties.json",
)


class Geometry:
    def __init__(self, properties=None):
        if properties is None:
            with open(PROPERTIES_JSON) as f:
                properties = json.load(f)
        coordinate = properties["coordinate"]
        self.unit = coordinate["unit"]
        self.min_gap = coordinate["min_gap"]
        self.sizes = dict(properties["size"])
        # Radii and gap in whole grid cells, for the exact test of collides.
        # Without whole multiples collides falls back to the float test.
        self.cell_sizes = self.cell_gap = None
        in_cells = {k: v / self.unit for k, v in self.sizes.items()}
        in_cells["gap"] = self.min_gap / self.unit
        if all(abs(v - round(v)) < 1e-9 for v in in_cells.values()):
            self.cell_gap = round(in_cells.pop("gap"))
            self.cell_sizes = {k: round(v) for k, v in in_cells.items()}
        else:
            warnings.warn(
                "radii and min_gap are not whole grid cells; collides uses"
                " the float test, so exact contact stays blocked"
            )
        grid = {
            k: int(math.ceil(v / self.unit)) for k, v in coordinate.items()
        }
        self.x_range = (grid["x_min"], grid["x_max"])
        self.y_range = (grid["y_min"], grid["y_max"])
        self.vis_x_range = (grid["vis_x_min"], grid["vis_x_max"])
        self.vis_y_range = (grid["vis_y_min"], grid["vis_y_max"])
        step = grid["step"]
        position = properties["position"]
        # (direction, n) -> (dx, dy), n as the grammar's string.
        self.moves = {}
        for direction, (vx, vy) in position["direction"].items():
            for n in position["step"]:
                self.moves[(direction, str(n))] = (vx * n * step, vy * n * step)

    def on_grid(self, cell):
        x, y = cell
        return (
            self.x_range[0] <= x <= self.x_range[1]
            and self.y_range[0] <= y <= self.y_range[1]
        )

    def visible(self, cell):
        x, y = cell
        return (
            self.vis_x_range[0] <= x <= self.vis_x_range[1]
            and self.vis_y_range[0] <= y <= self.vis_y_range[1]
        )

    def destination(self, cell, move):
        dx, dy = self.moves[move]
        return (cell[0] + dx, cell[1] + dy)

    def overlaps(self, cell_a, size_a, cell_b, size_b):
        """Whether two objects at grid cells are too close, as trance/group.py
        tests it (float64 metres; exact contact may count either way)."""
        unit = self.unit
        return math.hypot(
            cell_a[0] * unit - cell_b[0] * unit,
            cell_a[1] * unit - cell_b[1] * unit,
        ) < (self.sizes[size_a] + self.sizes[size_b] + self.min_gap)

    def collides(self, cell_a, size_a, cell_b, size_b):
        """Whether two objects at grid cells overlap in Blender's check: closer
        than r_a + r_b + min_gap, so exact contact is free. Compares squared
        integer distances in cells, without rounding. It differs from
        overlaps only at exactly the minimum distance."""
        if self.cell_sizes is None:
            return self.overlaps(cell_a, size_a, cell_b, size_b)
        limit = (
            self.cell_sizes[size_a] + self.cell_sizes[size_b] + self.cell_gap
        )
        dx = cell_a[0] - cell_b[0]
        dy = cell_a[1] - cell_b[1]
        return dx * dx + dy * dy < limit * limit

    def is_bigger(self, size_a, size_b):
        return self.sizes[size_a] > self.sizes[size_b]
