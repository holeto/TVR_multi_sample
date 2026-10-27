"""Scene geometry of TRANCE, read from properties.json.

The numbers and formulas follow the generator (trance/option.py and
trance/group.py), so that the executor judges overlaps exactly as it does:
positions are integer grid cells, footprints are compared in metres.
"""
import json
import math
import os

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
        """Whether two objects at grid cells are too close."""
        unit = self.unit
        return math.hypot(
            cell_a[0] * unit - cell_b[0] * unit,
            cell_a[1] * unit - cell_b[1] * unit,
        ) < (self.sizes[size_a] + self.sizes[size_b] + self.min_gap)

    def is_bigger(self, size_a, size_b):
        return self.sizes[size_a] > self.sizes[size_b]
