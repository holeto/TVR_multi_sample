# -*- coding: utf-8 -*-
"""
Helpers for generating several samples (members) of one transformation.

The first member (the anchor) is sampled as usual. The other members replay the
anchor's atomic transformations on new initial scenes. Target objects are
identified across members by their descriptor (stage-0 color, material, shape
and optionally size), since their positions and slots differ.

Pure Python (no bpy), so it can be tested outside Blender. Must stay compatible
with Blender 2.79's Python 3.5.
"""

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


def target_indices(info):
    """Indices of transformed objects, in order of first appearance."""
    indices = []
    for trans in info["transformations"]:
        if trans["obj_idx"] not in indices:
            indices.append(trans["obj_idx"])
    return indices


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
