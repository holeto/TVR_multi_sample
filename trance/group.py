# -*- coding: utf-8 -*-
"""
Helpers for generating several samples (members) of one transformation.

The first member (the anchor) is sampled as usual. The other members replay the
anchor's atomic transformations on new initial scenes. Target objects are
identified across members by their descriptor (stage-0 color, material, shape
and optionally size), since their positions and slots differ. Members can also
share some non-target objects with the anchor, at new positions.

Pure Python (no bpy), so it can be tested outside Blender. Must stay compatible
with Blender 2.79's Python 3.5.
"""
import math
from collections import defaultdict

DESC_KEYS = ["size", "color", "material", "shape"]


def desc_keys(keep_size=True):
    return DESC_KEYS if keep_size else DESC_KEYS[1:]


def obj_desc(obj, keep_size=True):
    """The attributes that identify an object across members."""
    return {k: obj[k] for k in desc_keys(keep_size)}


def obj_key(obj, keep_size=True):
    """String key of an object, e.g. "large_red_metal_cube"."""
    return "_".join(obj[k] for k in desc_keys(keep_size))


def unique_key_indices(objects, keep_size=True):
    """Indices of objects whose key appears exactly once in `objects`."""
    keys = [obj_key(obj, keep_size) for obj in objects]
    return [i for i, key in enumerate(keys) if keys.count(key) == 1]


def extract_plan(info, is_visible, all_sizes, keep_size=True, split="."):
    """Describe the transformation of a sample so it can be replayed.

    Args:
        info: sample info as built by `Recorder` (all stages are recorded).
        is_visible: function (x, y) -> bool on the integer grid.
        all_sizes: all size names.
        keep_size: whether target sizes must equal the anchor's.

    Returns a dict:
        roles: one entry per transformed object, in order of first appearance:
            obj_idx: index in the anchor.
            key, desc: identity of the object (stage-0 attributes).
            sizes: allowed initial sizes in a replay.
            constraints: [(dx, dy, visible)] for every stage, where (dx, dy)
                is the offset from the initial position. A replayed start
                position must give the same visibility at every stage.
        steps: [(role, pair, move_type)] in order; move_type is None for
            non-position pairs.
    """
    states = info["states"]
    roles = []
    role_of_idx = {}
    steps = []
    for trans in info["transformations"]:
        idx = trans["obj_idx"]
        if idx not in role_of_idx:
            role_of_idx[idx] = len(roles)
            init_obj = states[0]["objects"][idx]
            x0, y0 = init_obj["position"]
            constraints = []
            for state in states:
                x, y = state["objects"][idx]["position"]
                constraints.append((x - x0, y - y0, bool(is_visible(x, y))))
            roles.append(
                {
                    "obj_idx": idx,
                    "key": obj_key(init_obj, keep_size),
                    "desc": obj_desc(init_obj, keep_size),
                    "init_size": init_obj["size"],
                    "first_size_target": None,
                    "constraints": constraints,
                }
            )
        role = role_of_idx[idx]
        attr = trans["attr"]
        if attr == "size" and roles[role]["first_size_target"] is None:
            roles[role]["first_size_target"] = trans["pair"].split(split)[1]
        move_type = trans["type"] if attr == "position" else None
        steps.append((role, trans["pair"], move_type))

    for role in roles:
        if keep_size:
            role["sizes"] = [role["init_size"]]
        else:
            # A size step fails if it does not change the size.
            role["sizes"] = [
                s for s in sorted(all_sizes) if s != role["first_size_target"]
            ]
        del role["init_size"]
        del role["first_size_target"]

    return {"roles": roles, "steps": steps}


def abstract_pair(pair, move_type, split="."):
    """The pair as the grammar writes it.

    In- and out-moves drop their direction and step ("position.in",
    "position.out"); other pairs are unchanged.
    """
    if move_type in ("in", "out"):
        return split.join(["position", move_type])
    return pair


def abstract_transformation(info, split="."):
    """The steps of a sample as the grammar writes them (S -> R A)."""
    steps = []
    for trans in info["transformations"]:
        steps.append(
            {
                "obj_key": trans["obj_key"],
                "obj_desc": trans["obj_desc"],
                "pair": abstract_pair(trans["pair"], trans.get("type"), split),
            }
        )
    return {"trans_id": info.get("trans_id", info["idx"]), "steps": steps}


def _commute(step_a, step_b):
    # The final state never depends on the order of steps. Around an in- or
    # out-move of an object its other steps keep their order: it is edited
    # only once it is visible, and moved out only if it was not changed. The
    # validity of any other order is checked by replaying it.
    role_a, pair_a, move_a = step_a
    role_b, pair_b, move_b = step_b
    return role_a != role_b or not {move_a, move_b} & {"in", "out"}


def canonical_steps(plan):
    """Order of the steps of `plan` that depends only on what they do.

    Commuting steps are sorted by (target key, pair as the grammar writes
    it); steps that do not commute keep their relative order. Returns the
    reordered list of steps.
    """
    steps = plan["steps"]
    roles = plan["roles"]
    remaining = list(range(len(steps)))
    result = []
    while remaining:
        ready = [
            i
            for i in remaining
            if all(
                _commute(steps[j], steps[i]) for j in remaining if j < i
            )
        ]
        best = min(
            ready,
            key=lambda i: (
                roles[steps[i][0]]["key"],
                abstract_pair(steps[i][1], steps[i][2]),
                i,
            ),
        )
        remaining.remove(best)
        result.append(steps[best])
    return result


def target_indices(info):
    """Indices of transformed objects, in order of first appearance."""
    indices = []
    for trans in info["transformations"]:
        if trans["obj_idx"] not in indices:
            indices.append(trans["obj_idx"])
    return indices


def shared_candidates(info, is_visible):
    """Indices of stage-0 objects that may be shared with other members.

    These are the non-target objects in the visible area. Non-target objects
    never move, so invisible ones never appear in any image.
    """
    targets = target_indices(info)
    return [
        i
        for i, obj in enumerate(info["states"][0]["objects"])
        if i not in targets and is_visible(*obj["position"])
    ]


def unoccluded_candidates(info, is_visible, cameras):
    """Shared candidates that are not occluded at stage 0 in any of
    `cameras`."""
    state = info["states"][0]
    return [
        i
        for i in shared_candidates(info, is_visible)
        if all(
            state["status"][cam][state["objects"][i]["name"]] == "visible"
            for cam in cameras
        )
    ]


def repeated_attr_steps(info):
    """Non-position steps that edit an attribute already edited before.

    Returns a list of (step, obj_idx, attr).
    """
    seen = set()
    result = []
    for i, trans in enumerate(info["transformations"]):
        if trans["attr"] == "position":
            continue
        key = (trans["obj_idx"], trans["attr"])
        if key in seen:
            result.append((i, trans["obj_idx"], trans["attr"]))
        seen.add(key)
    return result


def move_violations(info, allow_repeated_moves=False):
    """Steps that break the rules for moves the grammar relies on.

    - no step on an object after it was moved out;
    - no move of an object after it was moved in;
    - at most one move per object, unless `allow_repeated_moves`;
    - no move that brings an object back to its initial position.

    Returns a list of (step, obj_idx, reason).
    """
    states = info["states"]
    moves = defaultdict(list)
    result = []
    for i, trans in enumerate(info["transformations"]):
        idx = trans["obj_idx"]
        done = moves[idx]
        if "out" in done:
            result.append((i, idx, "step after moving out"))
        elif trans["attr"] == "position":
            position = tuple(states[i + 1]["objects"][idx]["position"])
            if "in" in done:
                result.append((i, idx, "move after moving in"))
            elif done and not allow_repeated_moves:
                result.append((i, idx, "second move"))
            elif position == tuple(states[0]["objects"][idx]["position"]):
                result.append((i, idx, "move back to the start"))
        if trans["attr"] == "position":
            done.append(trans["type"])
    return result


def footprint(obj, coordinate, sizes):
    """Centre and radius of an object description, in real units.

    coordinate, sizes: "coordinate" and "size" of properties.json.
    """
    x, y = obj["position"]
    unit = coordinate["unit"]
    return (x * unit, y * unit), sizes[obj["size"]]


def overlaps(footprint_a, footprint_b, min_gap):
    """Whether two footprints are too close, as in Scene.is_overlapped."""
    (x_a, y_a), r_a = footprint_a
    (x_b, y_b), r_b = footprint_b
    return math.hypot(x_a - x_b, y_a - y_b) < r_a + r_b + min_gap


def in_move_options(
    objects, idx, move_options, is_valid, is_visible, coordinate, sizes
):
    """In-moves of objects[idx] whose destination is on the grid, visible and
    overlaps no other object. Empty if the object is visible.

    move_options: [(dx, dy, direction, step)], as Option.move_options.
    Returns a sorted list of (direction, step).
    """
    obj = objects[idx]
    x, y = obj["position"]
    if is_visible(x, y):
        return []
    min_gap = coordinate["min_gap"]
    others = [
        footprint(other, coordinate, sizes)
        for i, other in enumerate(objects)
        if i != idx
    ]
    result = []
    for dx, dy, direction, step in move_options:
        cell = (x + dx, y + dy)
        if not is_valid(*cell) or not is_visible(*cell):
            continue
        moved = footprint(dict(obj, position=cell), coordinate, sizes)
        if not any(overlaps(moved, other, min_gap) for other in others):
            result.append((direction, step))
    return sorted(result)


def candidate_conflicts(info, coordinate, sizes):
    """Steps that put another object onto a candidate of an earlier in-move.

    The grammar's 'in' has no direction, so afterwards the object is only
    known to be at one of the `candidates` recorded with the move. Later moves
    and growths of other objects must not overlap any of them.

    Returns a list of (step, obj_idx, obj_idx of the moved-in object).
    """
    states = info["states"]
    transformations = info["transformations"]
    result = []
    for k, trans_in in enumerate(transformations):
        if trans_in.get("type") != "in":
            continue
        for j in range(k + 1, len(transformations)):
            trans = transformations[j]
            idx = trans["obj_idx"]
            if idx == trans_in["obj_idx"]:
                continue
            obj = states[j + 1]["objects"][idx]
            old_size = states[j]["objects"][idx]["size"]
            grows = sizes[obj["size"]] > sizes[old_size]
            if trans["attr"] != "position" and not grows:
                continue
            moved_in = states[j + 1]["objects"][trans_in["obj_idx"]]
            for cell in trans_in.get("candidates", []):
                if overlaps(
                    footprint(obj, coordinate, sizes),
                    footprint(dict(moved_in, position=cell), coordinate, sizes),
                    coordinate["min_gap"],
                ):
                    result.append((j, idx, trans_in["obj_idx"]))
                    break
    return result


def occluded_targets(info, initial_cameras, final_cameras):
    """Targets that are occluded where they must be observable.

    A target must not be occluded in the initial cameras at stage 0 and in the
    final cameras at the final stage. Being invisible is fine: that is a target
    moved in (stage 0) or moved out (final stage).

    Returns a list of (obj_idx, stage, camera).
    """
    states = info["states"]
    checks = [(0, initial_cameras), (len(states) - 1, final_cameras)]
    result = []
    for idx in target_indices(info):
        name = states[0]["objects"][idx]["name"]
        for stage, cameras in checks:
            status = states[stage]["status"]
            for cam in sorted(cameras):
                if status[cam][name] == "occluded":
                    result.append((idx, stage, cam))
    return result
