# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from record import Recorder, SampleManager  # noqa: E402
from scene import Scene  # noqa: E402

import group  # noqa: E402
import utils  # noqa: E402

"""
This script is used for rendering TRANCE samples.
Render on GPUs:
# blender --background --python render.py -- --config configs/standard.yaml
Render on CPUs:
# blender --background --python render.py -- --config configs/standard.yaml \
    --gpu false --render_tile_size 16

Blender must be installed first. Please refer to `install_blender.sh`.
"""


class Renderer:
    def __init__(self, config):
        self.config = config

        self.prepare_log_and_dir()
        self.prepare_basic_settings()
        self.load_scene()
        self.prepare_sampler()

    def prepare_log_and_dir(self):
        self.log_dir = (
            Path(self.config.output_log_dir)
            .expanduser()
            .joinpath(
                "{}_{}_{}".format(
                    self.config.problem,
                    self.config.start,
                    self.config.start + self.config.n_sample - 1,
                )
            )
        )
        self.logger = utils.get_logger(
            __name__, log_dir=self.log_dir, use_tqdm=True
        )

        self.output_dir = (
            Path(self.config.output_data_dir)
            .expanduser()
            .joinpath(self.config.problem)
        )
        self.image_dir = self.output_dir.joinpath("image")
        self.blend_dir = self.output_dir.joinpath("blend")
        self.info_dir = self.output_dir.joinpath("info")
        self.seg_dir = self.output_dir.joinpath("seg")
        # The canonical steps of every transformation, as the grammar writes
        # them. One line is appended per newly generated transformation.
        self.abstract_path = self.output_dir.joinpath(
            "transformation_abstract_{}.jsonl".format(self.config.n_per_trans)
        )

        directories = [self.image_dir, self.seg_dir, self.info_dir]

        if self.config.output_blend:
            directories.append(self.blend_dir)
        if self.config.output_log:
            directories.append(self.log_dir)
        for directory in directories:
            directory.mkdir(exist_ok=True, parents=True)

    def prepare_basic_settings(self):
        self.recorder = Recorder()
        # One recorder per member of the current transformation.
        self.recorders = [self.recorder]

        self.global_seed = utils.hash_seed(
            self.config.problem, self.config.seed
        )

        self.start = config.start
        self.end = config.start + config.n_sample
        self.current_idx = self.start
        self.current_member = 0
        self.current_stage = 0

        self.scene_reload_time = 10

        self.n_replay_try = 0
        self.n_replay_fail = 0
        self.n_canonical_fail = 0
        self.n_core_fail = 0
        self.n_in_fail = 0

        if not self.config.replay_keep_size:
            self.logger.warning(
                "Target descriptors have no size (replay_keep_size: false), "
                "so {} is not written.".format(self.abstract_path.name)
            )

    @property
    def group_mode(self):
        return self.config.n_per_trans > 1

    def load_scene(self):
        self.logger.info("Loading scene into blender...")
        self.scene = Scene(
            self.config.base_scene_file,
            self.config.shape_dir,
            self.config.material_dir,
            self.config.properties_json,
        )
        self.scene.set_render_args(
            resolution_x=self.config.width,
            resolution_y=self.config.height,
            tile_x=self.config.render_tile_size,
            tile_y=self.config.render_tile_size,
            gpu=self.config.gpu,
            render_num_samples=self.config.render_num_samples,
            transparent_min_bounces=self.config.transparent_min_bounces,
            transparent_max_bounces=self.config.transparent_max_bounces,
            min_bounces=self.config.min_bounces,
            max_bounces=self.config.max_bounces,
        )

        self.valid_cameras = self.scene.cameras.keys()
        self.initial_cameras = list(
            set(self.valid_cameras) & set(self.config.initial_cameras)
        )
        self.final_cameras = list(
            set(self.valid_cameras) & set(self.config.final_cameras)
        )
        self.logger.info(
            "Available cameras for initial state: {}".format(
                self.initial_cameras
            )
        )
        self.logger.info(
            "Available cameras for final state: {}".format(self.final_cameras)
        )

    def prepare_sampler(self):
        valid_pairs = [
            pair
            for pair in self.scene.option.pairs
            if pair.startswith(tuple(self.config.valid_attr))
        ]
        assert config.min_init_vis <= config.max_init_vis <= config.total_obj
        assert (
            0
            <= config.min_sample_shared_objects
            <= config.max_sample_shared_objects
        )
        # Anchors need visible core distractors besides a target, so fewer
        # visible objects are never drawn for them.
        min_init_vis = config.min_init_vis
        if self.group_mode and config.min_sample_shared_objects > 0:
            min_init_vis = max(
                min_init_vis, config.min_sample_shared_objects + 1
            )
        # And min_invisible_distractors objects that can move in (in-targets
        # and core decoys), so they keep that many objects out of view.
        max_init_vis = config.max_init_vis
        assert (
            0
            <= config.min_invisible_distractors
            <= config.max_invisible_distractors
        )
        if self.group_mode and config.min_invisible_distractors > 0:
            max_init_vis = min(
                max_init_vis,
                config.total_obj - config.min_invisible_distractors,
            )
        assert min_init_vis <= max_init_vis
        self.sample_manager = SampleManager(
            valid_init_vis=range(min_init_vis, max_init_vis + 1),
            total_obj=config.total_obj,
            valid_steps=range(config.min_trans, config.max_trans + 1),
            valid_pairs=valid_pairs,
            valid_move_type=["inner", "out", "in"],
            pair_split=self.scene.option.SPLIT,
        )

    @property
    def str_stage(self):
        if self.current_stage == 0:
            return "initial"
        elif self.current_stage == self.num_step:
            return "final"
        else:
            return "{:02d}".format(self.current_stage)

    @property
    def sample_tag(self):
        if self.group_mode:
            return "{:06d}_{:02d}".format(self.current_idx, self.current_member)
        return "{:06d}".format(self.current_idx)

    @property
    def images_path(self):
        cameras = (
            self.initial_cameras
            if self.current_stage == 0
            else self.final_cameras
        )
        return {
            cam: os.path.join(
                str(self.image_dir),
                "{}_{}_img_{}-{}.{}.png".format(
                    self.config.dataset_prefix,
                    self.config.problem,
                    self.sample_tag,
                    self.str_stage,
                    cam,
                ),
            )
            for cam in cameras
        }

    @property
    def segs_path(self):
        return {
            cam: os.path.join(
                str(self.seg_dir),
                "{}_{}_seg_{}-{}.{}.png".format(
                    self.config.dataset_prefix,
                    self.config.problem,
                    self.sample_tag,
                    self.str_stage,
                    cam,
                ),
            )
            for cam in self.final_cameras
        }

    @property
    def blend_path(self):
        return os.path.join(
            str(self.blend_dir),
            "{}_{}_{}-{}.blend".format(
                self.config.dataset_prefix,
                self.config.problem,
                self.sample_tag,
                self.str_stage,
            ),
        )

    @property
    def json_path(self):
        return os.path.join(
            str(self.info_dir),
            "{}_{}_{}.json".format(
                self.config.dataset_prefix,
                self.config.problem,
                self.sample_tag,
            ),
        )

    @property
    def prefix(self):
        if self.group_mode:
            return "#{}.{}-{} ".format(
                self.current_idx, self.current_member, self.current_stage
            )
        return "#{}-{} ".format(self.current_idx, self.current_stage)

    def info(self, message):
        self.logger.info(self.prefix + message)

    def warning(self, message):
        self.logger.warning(self.prefix + message)

    def error(self, message):
        self.logger.error(self.prefix + message, exc_info=True)

    def exclude_output_dir(self, path_dict):
        return {key: os.path.basename(path) for key, path in path_dict.items()}

    def set_seed(self, member=0):
        seed = [self.global_seed, self.current_idx]
        if member > 0:
            seed.append(member)
        random.seed(str(seed))
        np.random.seed(seed)

    def run(self, replace_start=-1):
        for self.current_idx in tqdm.trange(self.start, self.end, ncols=80):
            # Reload to avoiding the unknown issue of slowing down.
            if self.current_idx % self.scene_reload_time == 0:
                self.load_scene()

            # Skip existing samples.
            json_paths = self.member_json_paths()
            existing = [path for path in json_paths if os.path.isfile(path)]
            if len(existing) == len(json_paths):
                infos = [self.sample_manager.record_json(json_paths[0])]
                for path in json_paths[1:]:
                    with open(path, "r") as f:
                        infos.append(json.load(f))
                if self.current_idx >= replace_start:
                    self.render_group(infos)
                continue
            # A partially generated transformation is generated again.
            for path in existing:
                self.warning("Removing incomplete member {}".format(path))
                os.remove(path)

            self.logger.info("---")
            self.set_seed()
            while True:
                try:
                    if self.build_group():
                        self.sample_manager.sample_success()
                        self.summary()
                        break
                    else:
                        self.sample_manager.sample_fail(
                            keep_init_vis=self.core_failed
                        )
                        self.clear_history()
                except KeyboardInterrupt:
                    self.warning("Keyboard Interrupted.")
                    sys.exit(1)
                except Exception as e:
                    self.error(str(e))

    """
    Rule A: no overlapping.
    Rule B: transformation must be observable
    1. occluded objects can not be transformmed (visible in initial states)
    2. transformed objects can not be moved out (visible in final states)
    3. transformed objects can not be occluded (visible in final states)
    """

    def member_json_paths(self):
        paths = []
        for self.current_member in range(self.config.n_per_trans):
            paths.append(self.json_path)
        self.current_member = 0
        return paths

    def build_group(self):
        # The first member (anchor) is sampled with the balanced sampler, the
        # others replay its transformation on new initial scenes.
        self.current_member = 0
        self.recorder = Recorder()
        self.recorders = [self.recorder]
        self.core_failed = False
        if not self.build_sample():
            return False
        # Every transformation gets the same number of objects that can move
        # in, in-targets included, so that their number does not tell whether
        # it has an in-move. At least one of them must be a decoy.
        n_in = self.config.min_invisible_distractors
        n_targets = len(self.in_targets(self.recorder.info))
        if self.group_mode and n_in > 0 and n_targets >= n_in:
            self.warning(
                "[Fail] {} in-targets leave no decoy of {}.".format(
                    n_targets, n_in
                )
            )
            self.n_in_fail += 1
            return False
        core = self.choose_core(self.recorder.info)
        chosen = self.choose_decoys(self.recorder.info)
        if core is None or chosen is None:
            # The visible count stays used, so the balanced sampler does not
            # keep drawing counts that can not hold the core.
            self.n_core_fail += 1
            self.core_failed = True
            return False

        # Steps are recorded in canonical order, on the anchor's initial
        # scene with its decoys re-placed. If that is invalid on the anchor or
        # on a member, the transformation is dropped.
        decoys, objects = chosen
        anchor = self.build_canonical(self.recorder.info, objects)
        if anchor is None:
            self.n_canonical_fail += 1
            return False
        self.current_member = 0
        self.recorder = anchor
        self.recorders = [anchor]
        if not self.build_members(core, decoys):
            self.n_replay_fail += 1
            return False

        # Save the information about the samples into json files.
        infos = []
        for self.current_member, recorder in enumerate(self.recorders):
            self.info("Saveing info...")
            if not recorder.save(self.json_path, self.config.json_indent):
                return False
            with open(self.json_path, "r") as f:
                infos.append(json.load(f))
        self.save_abstract(infos)
        self.render_group(infos)

        return True

    def save_abstract(self, infos):
        if not self.config.replay_keep_size:
            return
        line = group.abstract_transformation(
            infos[0], split=self.scene.option.SPLIT
        )
        line["members"] = [
            os.path.basename(path) for path in self.member_json_paths()
        ]
        with open(str(self.abstract_path), "a") as f:
            f.write(json.dumps(line) + "\n")

    def extract_plan(self, info):
        return group.extract_plan(
            info,
            self.scene.option.is_visible,
            self.scene.option.size.keys(),
            keep_size=self.config.replay_keep_size,
            split=self.scene.option.SPLIT,
        )

    def build_canonical(self, info, objects=None):
        # Commuting steps are sorted, so that their order carries no
        # information the images do not show. The steps are replayed on the
        # initial state of `info`, or on `objects` if given; returns the
        # recorder of the replay (the current one if neither the order nor
        # the objects change), or None if the replay is invalid.
        plan = self.extract_plan(info)
        steps = group.canonical_steps(plan)
        if steps == plan["steps"] and objects is None:
            return self.recorder
        if objects is None:
            objects = info["states"][0]["objects"]
        self.info("[Order] Replaying the steps in canonical order.")
        self.recorder = Recorder()
        self.current_stage = 0
        self.num_step = len(steps)
        self.scene.set_scene(objects, info["cameras"], info["lamps"])
        self.scene.view_directions = self.scene._compute_directions()
        self.recorder.init(self.scene, self.current_idx)
        self.record()
        slots = [role["obj_idx"] for role in plan["roles"]]
        if (
            self.replay_steps({"steps": steps}, slots)
            and self.validate_observability()
        ):
            self.info("* Canonical replay is valid.")
            return self.recorder
        self.warning("[Order] Canonical replay is invalid, dropping it.")
        return None

    def choose_core(self, info):
        # Distractors that recur in every member: visible non-targets of the
        # anchor that are not occluded in its initial images. Returns their
        # indices in the anchor, or None if there are too few.
        n_core = self.config.min_sample_shared_objects
        if not self.group_mode or n_core == 0:
            return []
        unique = self.unique_indices(info)
        candidates = [
            i
            for i in group.unoccluded_candidates(
                info, self.scene.option.is_visible, self.initial_cameras
            )
            if i in unique
        ]
        if len(candidates) < n_core:
            self.warning(
                "[Fail] {} shared distractors needed, {} available.".format(
                    n_core, len(candidates)
                )
            )
            return None
        return random.sample(candidates, n_core)

    def unique_indices(self, info):
        # Distractors must be unique, so that each one passes the decoder's
        # "occurs once in every member" test like a target.
        return group.unique_key_indices(
            info["states"][0]["objects"], self.config.replay_keep_size
        )

    def in_targets(self, info):
        return set(
            trans["obj_idx"]
            for trans in info["transformations"]
            if trans.get("type") == "in"
        )

    def decoy_options(self, objects, idx):
        option = self.scene.option
        moves = group.in_move_options(
            objects,
            idx,
            option.move_options,
            option.is_position_valid,
            option.is_visible,
            option.original_coordinate,
            option.size,
        )
        return moves

    def decoy_candidates(self, info):
        # Invisible non-targets of the anchor with a unique descriptor and an
        # in-move. Returns [(index, moves)].
        objects = info["states"][0]["objects"]
        unique = self.unique_indices(info)
        targets = group.target_indices(info)
        result = []
        for i, obj in enumerate(objects):
            if (
                i in targets
                or i not in unique
                or self.scene.option.is_visible(*obj["position"])
            ):
                continue
            moves = self.decoy_options(objects, i)
            if moves:
                result.append((i, moves))
        return result

    def choose_decoys(self, info):
        # Invisible distractors that recur in every member, each with an
        # in-move (direction, step) of its own that enters the view in every
        # member, so that an in-target is not the only object that could be
        # moved in. In the anchor they are re-placed: invisible objects are
        # never rendered, so a new start changes no image. Each start is
        # chosen so that its move enters the view onto a free cell, off every
        # cell the targets take during the transformation.
        # Returns ([(index in the anchor, move)], stage-0 objects with the new
        # starts, or None if nothing changes), or None if it fails.
        n_in = self.config.min_invisible_distractors
        if not self.group_mode or n_in == 0:
            return [], None
        n_core = n_in - len(self.in_targets(info))
        option = self.scene.option
        states = info["states"]
        objects = [dict(obj) for obj in states[0]["objects"]]
        unique = self.unique_indices(info)
        targets = group.target_indices(info)
        candidates = [
            i
            for i, obj in enumerate(objects)
            if i not in targets
            and i in unique
            and not option.is_visible(*obj["position"])
        ]
        if len(candidates) < n_core:
            self.warning(
                "[Fail] {} invisible distractors needed, {} available.".format(
                    n_core, len(candidates)
                )
            )
            return None
        chosen = random.sample(candidates, n_core)
        # Moves are drawn independently of the in-targets' moves (a decoy
        # that never shared the target's direction would single it out).
        moves = [
            (direction, step)
            for _, _, direction, step in sorted(option.move_options)
        ]

        # Occupancy: the other objects at stage 0 and the targets at every
        # stage.
        option.reset()
        for i, obj in enumerate(objects):
            if i not in chosen:
                x, y = obj["position"]
                option.record_obj_addition(x, y, option.get_size(obj["size"]))
        for state in states[1:]:
            for i in targets:
                obj = state["objects"][i]
                x, y = obj["position"]
                option.record_obj_addition(x, y, option.get_size(obj["size"]))

        decoys = []
        for idx in chosen:
            radius = option.get_size(objects[idx]["size"])
            for move in random.sample(moves, len(moves)):
                dx, dy = option.get_delta_position(*move)
                position = option.random_trajectory_start(
                    radius, [(0, 0, False), (dx, dy, True)], [(dx, dy)]
                )
                if position is not None:
                    break
            else:
                self.warning("[Fail] No start for decoy {}.".format(idx))
                return None
            x, y = position
            option.record_obj_addition(x, y, radius)
            option.record_obj_addition(x + dx, y + dy, radius)
            objects[idx]["position"] = (x, y)
            decoys.append((idx, move))
        for idx, move in decoys:
            if move not in self.decoy_options(objects, idx):
                self.warning("[Fail] Decoy {} can not move in.".format(idx))
                return None
        return decoys, objects

    def build_members(self, core, decoys):
        # Replay the transformation of the anchor (self.recorder) on new
        # initial scenes.
        if not self.group_mode:
            return True
        anchor_info = self.recorder.info
        plan = self.extract_plan(anchor_info)
        for member in range(1, self.config.n_per_trans):
            self.logger.info("---")
            self.current_member = member
            self.set_seed(member)
            self.recorder = Recorder()
            self.recorders.append(self.recorder)
            if not self.build_replay(
                plan, anchor_info, member, core, decoys
            ):
                return False
        return True

    def render_group(self, infos):
        for self.current_member, info in enumerate(infos):
            self.render_from_json(info)
        self.current_member = 0

    def build_sample(self):
        # Sample the initial state.
        self.current_stage = 0
        if not self.sample_init_state():
            return False
        self.record()

        # Decide the number of steps.
        self.num_step = self.sample_manager.random_step()
        self.logger.info(
            "Number of atomic transformations: {}.".format(self.num_step)
        )

        # Apply randomly sampled atomic transformations one by one.
        for _ in range(self.num_step):
            self.current_stage += 1
            while True:
                pair = self.sample_manager.random_pair()
                if pair is None:
                    return False
                response = self.try_pair(pair)
                if response is None:
                    self.warning(
                        "[Fair] ({}/{}) {} -> no feasible object.".format(
                            len(self.sample_manager.tried_pairs),
                            self.sample_manager.top_pair,
                            pair,
                        )
                    )
                    self.sample_manager.stage_fail()
                else:
                    self.record(trans_response=response)
                    self.sample_manager.stage_success()
                    break

        # Show the sequence of transformation.
        for i, trans_info in enumerate(self.recorder.info["transformations"]):
            self.logger.info(
                "[{}] {} -> {}{}".format(
                    i,
                    trans_info["pair"],
                    trans_info["obj_idx"],
                    " ({})".format(trans_info["type"])
                    if trans_info["attr"] == "position"
                    else "",
                )
            )

        # The per-step checks miss some occlusions, so check the whole sample.
        if not self.validate_observability():
            return False

        return True

    def build_replay(self, plan, anchor_info, member, core, decoys):
        self.num_step = len(plan["steps"])
        for attempt in range(1, self.config.replay_max_tries + 1):
            self.n_replay_try += 1
            self.current_stage = 0
            slots = self.sample_replay_state(
                plan, anchor_info, member, core, decoys
            )
            if slots is None:
                continue
            self.record()
            core_slots = [
                shared["obj_idx"]
                for shared in self.recorder.info["shared_objs"]
                if shared["core"]
            ]
            if self.occluded_at_init(slots + core_slots):
                continue
            if not self.replay_steps(plan, slots):
                continue
            if self.validate_observability():
                self.info("* Replay succeeded (attempt {}).".format(attempt))
                return True
        self.warning(
            "[Fail] No replay after {} attempts.".format(
                self.config.replay_max_tries
            )
        )
        return False

    def sample_replay_state(self, plan, anchor_info, member, core, decoys):
        self.info("Start to building replay state.")
        self.scene.reset()
        if self.config.replay_share_jitter:
            self.scene.set_cameras_lamps(
                anchor_info["cameras"], anchor_info["lamps"]
            )
            self.scene.view_directions = self.scene._compute_directions()
        else:
            self.scene.perturb_camera(self.config.camera_jitter)
            self.scene.perturb_lamp(self.config.lamp_jitter)
        self.recorder.init(self.scene, self.current_idx, member)

        option = self.scene.option
        roles = plan["roles"]
        total_obj = self.config.total_obj

        # Place targets first, at starts from which all of their moves are
        # feasible. Their space is reserved before other objects are placed.
        targets = []
        for role in roles:
            size = random.choice(role["sizes"])
            radius = option.get_size(size)
            position = option.random_trajectory_start(
                radius, role["constraints"]
            )
            if position is None:
                self.warning(
                    "[Fail] No start position for {}.".format(role["key"])
                )
                return None
            option.record_obj_addition(position[0], position[1], radius)
            targets.append((size, position))

        # Copy visible non-target objects of the anchor, so that targets are
        # not the only objects the members have in common: the core ones
        # into every member, and up to (max - min) further ones per member.
        anchor_objects = anchor_info["states"][0]["objects"]
        unique = self.unique_indices(anchor_info)
        rest = [
            i
            for i in group.shared_candidates(anchor_info, option.is_visible)
            if i not in core and i in unique
        ]
        n_extra = random.randint(
            0,
            min(len(rest), self.config.max_sample_shared_objects - len(core)),
        )
        shared = []
        for anchor_idx in core + random.sample(rest, n_extra):
            radius = option.get_size(anchor_objects[anchor_idx]["size"])
            position = option.random_position(radius, True)
            if position is None:
                self.warning("[Fail] No position for a shared object.")
                return None
            option.record_obj_addition(position[0], position[1], radius)
            shared.append((anchor_idx, position))
        n_shared = len(shared)

        # Copy invisible non-targets of the anchor (decoys) the same way: the
        # core ones with the group's in-move, further ones with a move drawn
        # here. Each starts where its move would enter the view onto a free
        # cell; the cell is reserved so that other objects stay off it.
        core_decoys = [idx for idx, _ in decoys]
        rest = [
            candidate
            for candidate in self.decoy_candidates(anchor_info)
            if candidate[0] not in core_decoys
        ]
        n_extra = random.randint(
            0,
            min(
                len(rest),
                self.config.max_invisible_distractors
                - self.config.min_invisible_distractors,
            ),
        )
        hidden = []
        for anchor_idx, move in decoys + [
            (idx, random.choice(moves))
            for idx, moves in random.sample(rest, n_extra)
        ]:
            radius = option.get_size(anchor_objects[anchor_idx]["size"])
            dx, dy = option.get_delta_position(*move)
            position = option.random_trajectory_start(
                radius, [(0, 0, False), (dx, dy, True)], [(dx, dy)]
            )
            if position is None:
                self.warning("[Fail] No position for an invisible decoy.")
                return None
            x, y = position
            option.record_obj_addition(x, y, radius)
            option.record_obj_addition(x + dx, y + dy, radius)
            hidden.append((anchor_idx, position, move))

        # Decide the visibilities of objects.
        n_vis_targets = len([r for r in roles if r["constraints"][0][2]])
        n_invis_targets = len(roles) - n_vis_targets
        low = max(self.config.min_init_vis, n_vis_targets + n_shared)
        high = min(
            self.config.max_init_vis, total_obj - n_invis_targets - len(hidden)
        )
        if low > high:
            self.warning("[Fail] No valid number of visible objects.")
            return None
        n_vis = random.randint(low, high)
        visible_choices = [True] * n_vis + [False] * (total_obj - n_vis)
        random.shuffle(visible_choices)

        # Random slots for targets, so that obj_idx carries no information.
        slots = []
        free_slots = list(range(total_obj))
        for role in roles:
            slot = random.choice(
                [
                    i
                    for i in free_slots
                    if visible_choices[i] == role["constraints"][0][2]
                ]
            )
            free_slots.remove(slot)
            slots.append(slot)
        shared_slots = random.sample(
            [i for i in free_slots if visible_choices[i]], n_shared
        )
        self.recorder.info["shared_objs"] = [
            {
                "obj_idx": slot,
                "anchor_idx": anchor_idx,
                "core": anchor_idx in core,
            }
            for slot, (anchor_idx, _) in zip(shared_slots, shared)
        ]
        hidden_slots = random.sample(
            [i for i in free_slots if not visible_choices[i]], len(hidden)
        )
        self.recorder.info["decoy_objs"] = [
            {
                "obj_idx": slot,
                "anchor_idx": anchor_idx,
                "core": anchor_idx in core_decoys,
                "move": list(move),
            }
            for slot, (anchor_idx, _, move) in zip(hidden_slots, hidden)
        ]

        # Other objects must not share a descriptor with a target or a copy.
        copies = [idx for idx, _ in shared] + [idx for idx, _, _ in hidden]
        exclude = set(role["key"] for role in roles) | set(
            group.obj_key(anchor_objects[idx], self.config.replay_keep_size)
            for idx in copies
        )

        def key_fn(attrs):
            return group.obj_key(attrs, self.config.replay_keep_size)

        for i, visible in enumerate(visible_choices):
            if i in slots:
                role = roles[slots.index(i)]
                size, (x, y) = targets[slots.index(i)]
                _, obj = self.scene.add_object(
                    role["desc"]["shape"],
                    size,
                    option.random_rotation(),
                    x,
                    y,
                    role["desc"]["material"],
                    role["desc"]["color"],
                )
            elif i in shared_slots or i in hidden_slots:
                if i in shared_slots:
                    anchor_idx, (x, y) = shared[shared_slots.index(i)]
                else:
                    anchor_idx, (x, y), _ = hidden[hidden_slots.index(i)]
                anchor_obj = anchor_objects[anchor_idx]
                _, obj = self.scene.add_object(
                    anchor_obj["shape"],
                    anchor_obj["size"],
                    option.random_rotation(),
                    x,
                    y,
                    anchor_obj["material"],
                    anchor_obj["color"],
                )
            else:
                obj = self.scene.create_random_object(
                    visible=visible, exclude=exclude, key_fn=key_fn
                )
            if obj is None:
                self.warning(
                    "{} No enough space for {} objects.".format(
                        i, "visible" if visible else "invisible"
                    )
                )
                return None
            self.info(
                "{} ({:^9s}) Put a {:6s} {:6s} {:6s} {:8s}"
                " at ({:3d}, {:3d}){}".format(
                    i,
                    "visible" if visible else "invisible",
                    obj["size"],
                    obj["color"],
                    obj["material"],
                    obj["shape"],
                    *obj["position"],
                    " (target)"
                    if i in slots
                    else " (shared)"
                    if i in shared_slots
                    else " (decoy)"
                    if i in hidden_slots
                    else ""
                )
            )

        # Each decoy's move must still end on a free cell.
        objects = self.scene.objects_description
        for slot, (_, _, move) in zip(hidden_slots, hidden):
            if tuple(move) not in self.decoy_options(objects, slot):
                self.warning("[Fail] Decoy {} can not move in.".format(slot))
                return None

        return slots

    def occluded_at_init(self, slots):
        # Rule 1. Occluded objects can not be transformed. Core distractors
        # must be seen in every member too.
        status = self.recorder.info["states"][0]["status"]
        for slot in slots:
            name = self.scene.get_name(self.scene.objects[slot])
            for cam in self.initial_cameras:
                if status[cam][name] == "occluded":
                    self.warning("[Fail] object {} is occluded.".format(slot))
                    return True
        return False

    def replay_steps(self, plan, slots):
        for role, pair, move_type in plan["steps"]:
            self.current_stage += 1
            obj = self.scene.objects[slots[role]]
            is_success, response = self.transform(obj, pair, move_type)
            if not is_success:
                self.warning("[Fail] Replay of {} failed.".format(pair))
                return False
            self.record(trans_response=response)
        return True

    def validate_observability(self):
        # Rules 1 and 3 on the whole sample. Unlike the per-step checks, this
        # covers shape, material and color steps and all initial cameras.
        failures = group.occluded_targets(
            self.recorder.info, self.initial_cameras, self.final_cameras
        )
        for idx, stage, cam in failures:
            self.warning(
                "[Fail] target {} is occluded at stage {} in {}.".format(
                    idx, stage, cam
                )
            )
        return len(failures) == 0

    def render_from_json(self, info):
        cameras = info["cameras"]
        lamps = info["lamps"]
        self.num_step = len(info["states"]) - 1
        render_stages = list(range(self.num_step + 1))
        if not self.config.render_intermediate:
            render_stages = [render_stages[0], render_stages[-1]]
        for s in render_stages:
            self.current_stage = s
            for _, path in self.images_path.items():
                if not os.path.isfile(path):
                    objects = info["states"][s]["objects"]
                    self.scene.set_scene(objects, cameras, lamps)
                    if not self.render_main():
                        return False
                    break
        return True

    def sample_init_state(self):
        self.info("Start to building initial state.")

        # Reset & initialize blender environment.
        self.info("Reset environment.")
        self.scene.reset()
        self.scene.perturb_camera(self.config.camera_jitter)
        self.scene.perturb_lamp(self.config.lamp_jitter)

        # Reset recorder.
        self.info("Reset recorder")
        self.recorder.init(self.scene, self.current_idx)

        # Decide the visibilities of objects.
        self.num_init_visible = self.sample_manager.random_init()
        visible_choices = [True] * self.num_init_visible + [False] * (
            self.config.total_obj - self.num_init_visible
        )
        random.shuffle(visible_choices)

        # Randomly create and place objects.
        self.info(
            "Creating {} random objects, {} visible...".format(
                self.config.max_init_vis, self.num_init_visible
            )
        )
        for i, visible in enumerate(visible_choices):
            obj = self.scene.create_random_object(visible=visible)
            if obj is None:
                self.warning(
                    "{} No enough space for {} objects.".format(
                        i, "visible" if visible else "invisible"
                    )
                )
                self.warning("Failed to build initial state. Retry...")
                return False
            else:
                self.info(
                    "{} ({:^9s}) Put a {:6s} {:6s} {:6s} {:8s}"
                    " at ({:3d}, {:3d})".format(
                        i,
                        "visible" if visible else "invisible",
                        obj["size"],
                        obj["color"],
                        obj["material"],
                        obj["shape"],
                        *obj["position"]
                    )
                )

        self.info("* Successfully build initial state.")
        return True

    def try_pair(self, pair):
        # Try sampled atomic transformations on objects.
        while True:
            move_type = (
                self.sample_manager.random_move_type()
                if pair.startswith("position")
                else None
            )
            if pair.startswith("position") and move_type is None:
                return None
            objs = self.get_valid_objs(pair, move_type=move_type)
            self.info("Valid objects for {}: {}".format(pair, objs))
            while len(objs) > 0:
                obj_idx = self.sample_manager.random_obj(valid_objs=objs)
                objs.remove(obj_idx)
                is_success, response = self.transform(
                    self.scene.objects[obj_idx], pair, move_type
                )
                if is_success:
                    return response
            if not pair.startswith("position"):
                return None
            else:
                self.warning(
                    "[Fail] ({}/{}) move {}".format(
                        len(self.sample_manager.tried_move_type),
                        self.sample_manager.top_move_type,
                        move_type,
                    )
                )

    def get_valid_objs(self, pair, move_type=None):
        # Rule 1. Occluded objects can not be transformmed.
        if move_type == "in":
            choices = self.scene.invisible_objects
        else:
            choices = self.get_non_occluded_objs()

        # A second edit of the same attribute would hide the first one.
        attr = pair.split(self.scene.option.SPLIT)[0]
        if attr != "position":
            choices = [
                obj for obj in choices if not self.was_transformed(obj, attr)
            ]
        else:
            choices = [obj for obj in choices if self.can_move(obj)]

        choices = [self.scene.get_index(obj) for obj in choices]

        # Steps name their target by its descriptor, so it must be unique.
        unique = group.unique_key_indices(
            self.recorder.info["states"][0]["objects"],
            self.config.replay_keep_size,
        )
        choices = [idx for idx in choices if idx in unique]

        return choices

    def transform(self, obj, pair, move_type=None):
        trans = [obj, *pair.split(self.scene.option.SPLIT)]
        attr = trans.pop(1)

        trans_info = {
            "pair": pair,
            "obj_name": self.scene.get_name(obj),
            "obj_idx": self.scene.get_index(obj),
        }
        # Identity of the object across members: its initial attributes.
        init_obj = self.recorder.info["states"][0]["objects"][
            trans_info["obj_idx"]
        ]
        trans_info["obj_key"] = group.obj_key(
            init_obj, self.config.replay_keep_size
        )
        trans_info["obj_desc"] = group.obj_desc(
            init_obj, self.config.replay_keep_size
        )
        trans_func = getattr(self, "t_{}".format(attr))
        args = trans + [move_type] if move_type else trans
        self.info("{} <- {}".format(trans_info["obj_name"], pair))
        is_success, response = trans_func(*args)

        response.update(trans_info)

        return is_success, response

    def t_position(self, obj, direction, step, move_type):
        step = int(step)
        response = {
            "attr": "position",
            "old": self.scene.get_position(obj),
            "new": "",
            "target": (direction, step),
            "type": "",
            "options": [],
        }

        x, y = response["old"]
        dx, dy = self.scene.option.get_delta_position(direction, step)
        new_x = x + dx
        new_y = y + dy

        response["new"] = (new_x, new_y)

        if not self.scene.option.is_position_valid(new_x, new_y):
            self.warning(
                "[Fail] position ({}, {}) is invalid.".format(new_x, new_y)
            )
            return False, response

        # Several moves of one object must not cancel out.
        init_obj = self.recorder.info["states"][0]["objects"][
            self.scene.get_index(obj)
        ]
        if (new_x, new_y) == tuple(init_obj["position"]):
            self.warning("[Fail] move back to the initial position.")
            return False, response

        vis_old = self.scene.option.is_visible(x, y)
        vis_new = self.scene.option.is_visible(new_x, new_y)

        if vis_old and vis_new:
            response["type"] = "inner"
        elif vis_old and (not vis_new):
            response["type"] = "out"
        elif (not vis_old) and vis_new:
            response["type"] = "in"
        else:
            self.warning("[Fail] move object inside invisible area.")
            return False, response

        if response["type"] != move_type:
            self.warning(
                "[Fail] move type ({} vs. {}) not match.".format(
                    response["type"], move_type
                )
            )
            return False, response

        # Rule 2. Transformed objects can not be moved out.
        if response["type"] == "out" and self.was_transformed(obj):
            self.warning("[Fail] Modified objects can not be moved out.")
            return False, response

        self.scene.set_position(obj, new_x, new_y)

        if self.scene.is_overlapped(obj):
            self.warning("[Fail] Overlap.")
        elif self.takes_in_candidate(obj) is not None:
            self.warning("[Fail] Takes a candidate of a moved-in object.")
        # Rule 3. Transformed objects can not be occluded.
        elif self.cause_occlusion(obj):
            self.warning("[Fail] Cause occlusion.")
        else:
            if response["type"] == "out":
                response["options"].extend(
                    self.scene.option.get_move_options(x, y)["invisible"]
                )
            else:
                response["options"].append((direction, step))
            if response["type"] == "in":
                response["candidates"] = self.in_candidates(obj, x, y)
            return True, response

        # Revert.
        self.scene.set_position(obj, x, y)
        return False, response

    def t_shape(self, obj, new_shape):
        response = {
            "attr": "shape",
            "old": self.scene.get_shape(obj),
            "new": None,
            "target": None,
        }
        old_shape = response["old"]
        if new_shape == old_shape:
            self.warning("[Fail] no change.")
            return False, response
        obj = self.scene.set_shape(obj, new_shape)
        response["new"] = response["target"] = new_shape

        return True, response

    def t_size(self, obj, new_size):
        response = {
            "attr": "size",
            "old": self.scene.get_size(obj),
            "new": new_size,
            "target": new_size,
        }
        old_size = response["old"]
        if new_size == old_size:
            self.warning("[Fail] no change.")
            return False, response

        self.scene.set_size(obj, new_size)

        grows = self.scene.option.is_bigger(new_size, old_size)
        if grows and self.scene.is_overlapped(obj):
            self.warning("[Fail] Overlap.")
        elif grows and self.takes_in_candidate(obj) is not None:
            self.warning("[Fail] Takes a candidate of a moved-in object.")
        # Rule 3. transformed objects can not be occluded (visible in final states)
        elif self.cause_occlusion(obj):
            self.warning("[Fail] Cause occlusion.")
        else:
            return True, response

        self.scene.set_size(obj, old_size)
        return False, response

    def t_material(self, obj, new_material):
        response = {
            "attr": "material",
            "old": self.scene.get_material(obj),
            "new": new_material,
            "target": new_material,
        }
        old_material = response["old"]
        if new_material == old_material:
            self.warning("[Fail] no change.")
            return False, response

        self.scene.set_material(obj, new_material)
        return True, response

    def t_color(self, obj, new_color):
        response = {
            "attr": "color",
            "old": self.scene.get_color(obj),
            "new": new_color,
            "target": new_color,
        }
        old_color = response["old"]
        if new_color == old_color:
            self.warning("[Fail] no change.")
            return False, response

        self.scene.set_color(obj, new_color)
        return True, response

    def render_main(self):
        if not self.config.no_render:
            self.info("[Render] main ({})...".format(self.current_stage))
            for key, image_path in self.images_path.items():
                if not os.path.isfile(image_path):
                    time_used = self.scene.render(
                        image_path,
                        self.scene.cameras[key],
                        config.width,
                        config.height,
                    )
                    self.info("- {}: {}".format(key, utils.time2str(time_used)))
                    if not os.path.isfile(image_path):
                        return False
        return True

    def render_seg(self):
        self.info("[Render] seg ({})...".format(self.current_stage))
        for key, seg_path in self.segs_path.items():
            time_seg_used = self.scene.render_shadeless(
                seg_path,
                self.scene.cameras[key],
                config.seg_width,
                config.seg_height,
            )
            self.info("- {}: {}".format(key, utils.time2str(time_seg_used)))
            if not os.path.isfile(seg_path):
                return False
        return True

    def record(self, trans_response=None):
        status = self.get_objs_status()

        self.recorder.record_scene(
            self.scene,
            self.current_stage,
            self.exclude_output_dir(self.images_path),
            self.exclude_output_dir(self.segs_path),
            status["n_pixels"],
            status["status"],
        )
        if trans_response is not None:
            self.recorder.record_trans(
                self.current_stage - 1, self.current_stage, trans_response
            )

        if self.config.output_blend:
            self.info("[Save] Blender file.")
            self.scene.save(self.blend_path)

    def get_objs_status(self, stage=None):
        if stage is None:
            stage = self.current_stage
        states = self.recorder.info["states"]
        assert (
            -len(states) <= stage <= len(states)
        ), "stage {} is out of range".format(stage)
        if -len(states) <= stage < len(states):
            return {
                "status": states[stage]["status"],
                "n_pixels": states[stage]["n_pixels"],
            }
        else:
            assert self.render_seg(), "[Error] render seg failed."
            n_pixels = {}
            cams = list(set(self.initial_cameras) | set(self.final_cameras))
            for cam in cams:
                image_path = self.segs_path[cam]
                n_pixels[cam] = self.count_pixels(image_path)
            status = self.pixels2status(n_pixels)
            return {"status": status, "n_pixels": n_pixels}

    def count_pixels(self, seg_path):
        assert os.path.isfile(seg_path), "[Error] {} doesn't exist.".format(
            seg_path
        )
        pixels = {}
        colors = utils.count_color(seg_path)
        for n, color in sorted(colors):
            obj = self.scene.option.get_seg_object(color)
            if obj is not None:
                pixels[obj] = n
        return pixels

    def pixels2status(self, n_pixels):
        status = defaultdict(dict)
        min_pixels = self.config.occlusion_threshold
        for cam, val in n_pixels.items():
            for obj in self.scene.objects:
                if obj in self.scene.invisible_objects:
                    status[cam][obj.name] = "invisible"
                elif obj.name not in val or val[obj.name] < min_pixels:
                    status[cam][obj.name] = "occluded"
                else:
                    status[cam][obj.name] = "visible"
        return status

    def cause_occlusion(self, obj):
        status = self.get_objs_status(stage=self.current_stage)["status"]
        for x in self.scene.objects:
            if x == obj or self.was_transformed(x):
                x_name = self.scene.get_name(x)
                for cam in self.final_cameras:
                    if status[cam][x_name] == "occluded":
                        return self.scene.get_index(x)
        return None

    def get_non_occluded_objs(self):
        status = self.get_objs_status(stage=0)["status"]
        objects = self.scene.visible_objects.copy()
        for cam in self.initial_cameras:
            for obj_name, state in status[cam].items():
                obj = self.scene.b_objects[obj_name]
                if state == "occluded" and obj in objects:
                    objects.remove(obj)
        return objects

    def was_transformed(self, obj, attr=None):
        for trans_info in self.recorder.info["transformations"]:
            if obj.name == trans_info["obj_name"] and (
                attr is None or trans_info["attr"] == attr
            ):
                return True
        return False

    def move_types(self, obj):
        return [
            trans_info["type"]
            for trans_info in self.recorder.info["transformations"]
            if obj.name == trans_info["obj_name"]
            and trans_info["attr"] == "position"
        ]

    def can_move(self, obj):
        # The grammar does not follow an object after an in- or out-move, and
        # several moves of one object can add up to a single one.
        types = self.move_types(obj)
        if "in" in types or "out" in types:
            return False
        return len(types) == 0 or self.config.allow_repeated_moves

    def footprint(self, obj, position=None):
        desc = dict(self.scene.describe(obj))
        if position is not None:
            desc["position"] = position
        option = self.scene.option
        return group.footprint(desc, option.original_coordinate, option.size)

    def in_candidates(self, obj, x, y):
        # The grammar's 'in' has no direction, so afterwards the object is
        # only known to be at one of the free destinations of in-moves from
        # its start (x, y).
        option = self.scene.option
        min_gap = option.original_coordinate["min_gap"]
        others = [self.footprint(o) for o in self.scene.objects if o != obj]
        cells = []
        for direction, step in option.get_move_options(x, y)["visible"]:
            dx, dy = option.get_delta_position(direction, step)
            cell = self.footprint(obj, (x + dx, y + dy))
            if not any(group.overlaps(cell, o, min_gap) for o in others):
                cells.append((x + dx, y + dy))
        return sorted(cells)

    def takes_in_candidate(self, obj):
        # No other object may later overlap a cell where a moved-in object may
        # be. Returns the index of the moved-in object, or None.
        idx = self.scene.get_index(obj)
        min_gap = self.scene.option.original_coordinate["min_gap"]
        obj_footprint = self.footprint(obj)
        for trans_info in self.recorder.info["transformations"]:
            if trans_info.get("type") != "in" or trans_info["obj_idx"] == idx:
                continue
            moved_in = self.scene.objects[trans_info["obj_idx"]]
            for cell in trans_info["candidates"]:
                if group.overlaps(
                    obj_footprint, self.footprint(moved_in, cell), min_gap
                ):
                    return trans_info["obj_idx"]
        return None

    def clear_history(self):
        self.logger.info("Removing files after failed...")
        for recorder in self.recorders:
            if not hasattr(recorder, "info"):
                continue
            for scene in recorder.info["states"]:
                for image in scene["images"].values():
                    img_path = self.image_dir / image
                    if img_path.exists():
                        img_path.unlink()
                for seg in scene["segs"].values():
                    seg_path = self.seg_dir / seg
                    if seg_path.exists():
                        seg_path.unlink()

    def summary(self):
        self.logger.info("---")
        self.logger.info("Data dir: {}".format(self.output_dir.resolve()))
        if self.config.output_log:
            self.logger.info("Log dir: {}".format(self.log_dir.resolve()))

        self.logger.info("---")
        state = self.sample_manager.state()
        self.logger.info(
            "Progress: {} (load: {})".format(
                state["sample_success"] + state["sample_load"],
                state["sample_load"],
            )
        )
        self.logger.info(
            "Sample success: {} ({:.4f})".format(
                state["sample_success"], state["rate_sample_success"]
            )
        )
        self.logger.info(
            "Stage success: {} ({:.4f})".format(
                state["stage_success"], state["rate_stage_success"]
            )
        )
        self.logger.info(
            "Canonical replay invalid (transformations dropped): {}".format(
                self.n_canonical_fail
            )
        )
        if self.group_mode:
            self.logger.info(
                "Too few core distractors (transformations dropped): "
                "{}".format(self.n_core_fail)
            )
            self.logger.info(
                "Too many in-targets (transformations dropped): {}".format(
                    self.n_in_fail
                )
            )
            self.logger.info(
                "Replay attempts: {} (transformations dropped: {})".format(
                    self.n_replay_try, self.n_replay_fail
                )
            )
        self.logger.info("Total time: {}".format(utils.time2str(state["time"])))
        self.logger.info(
            "Average sample time: {}".format(
                utils.time2str(state["time_avg_sample"])
            )
        )
        self.logger.info(
            "Average stage time: {}".format(
                utils.time2str(state["time_avg_stage"])
            )
        )

        self.logger.info("---")
        self.logger.info("Initial Visible Object:")
        for init_vis, num in sorted(self.sample_manager.n_init_vis.items()):
            self.logger.info("- {}: {}".format(init_vis, num))

        self.logger.info("---")
        self.logger.info("Steps:")
        for key, num in sorted(self.sample_manager.n_step.items()):
            self.logger.info("- {}: {}".format(key, num))

        self.logger.info("---")
        self.logger.info("Object:")
        for key, num in sorted(self.sample_manager.n_obj.items()):
            self.logger.info("- {}: {}".format(key, num))

        self.logger.info("---")
        self.logger.info("Pair:")
        for key, num in sorted(self.sample_manager.n_pair["gram_1"].items()):
            self.logger.info("- {}: {}".format(key, num))

        self.logger.info("---")
        self.logger.info("Move Type:")
        for key, num in sorted(self.sample_manager.n_move_type.items()):
            self.logger.info("- {}: {}".format(key, num))

        self.logger.info("---")
        self.logger.info("Balance State")
        for key, state in sorted(self.sample_manager.balance_state.items()):
            self.logger.info("# {}".format(key))
            for k, v in sorted(state.items()):
                self.logger.info("- {}: {}".format(k, v))


if __name__ == "__main__":
    config = utils.Config("configs/standard.yaml")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--info",
        action="help",
        default=argparse.SUPPRESS,
        help="show this help message and exit.",
    )
    parser.add_argument("-c", "--config", help="YAML config file.")
    parser.add_argument(
        "--replace_start",
        default=-1,
        type=int,
        help="Replace possibly existed samples from <replace_start>.",
    )
    config.extend_parser(parser)

    argv = []
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        argv = sys.argv[(idx + 1) :]
    args = parser.parse_args(argv)
    if args.config:
        config = utils.Config(args.config)
    config.merge_args(args)

    render = Renderer(config)
    render.run(args.replace_start)
