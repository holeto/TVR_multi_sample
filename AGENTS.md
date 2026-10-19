

## What this repo is

A fork of [TVR](https://github.com/hughplay/TVR) (Transformation driven Visual Reasoning, CVPR 2021) built on the [DeepCodebase](https://github.com/hughplay/DeepCodebase) template (PyTorch Lightning + Hydra). Given an initial and a final scene image, models infer the transformation(s) between them on the TRANCE dataset. The fork's stated goal (see `README.md`) is to let the dataset generator produce **multiple examples per single transformation**, to test generalization. The generator in `trance/` is the place for that work; `trance/render.py` currently samples one scene per index.

There are two largely independent halves:
- `trance/` — Blender-based dataset generator (run inside the `blender` docker service, not the training container).
- `src/` + `conf/` + `train.py`/`test.py` — model training and evaluation.

## Environment

Everything runs in docker (`docker.py` wraps docker-compose and writes `.env`; `.env` must define `PROJECT`, since `conf/train.yaml` reads `${oc.env:PROJECT}`).

```bash
python docker.py startd            # first-time setup / start container (add --build to rebuild)
python docker.py                   # enter the container
python docker.py prepare --service blender --build   # generator image
python docker.py --service blender                  # enter generator container
```

Data paths are hard-coded to container paths in `conf/dataset/*.yaml` and `conf/criterion/*.yaml` (`/data/trance/{data.h5,properties.json,values.json}`). `DATA_ROOT` → `/data`, `LOG_ROOT` → `/log`.

**Warning: the docker build uses a third-party pip mirror by default.** Upstream `docker/Dockerfile` runs `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple` (Tsinghua TUNA mirror of PyPI) and copies `docker/misc/.condarc` (Tsinghua conda channels) into the image, so every package, including the pip default inside the container afterwards, comes from that mirror rather than PyPI. If the user is setting up docker (building or rebuilding the image), check `docker/Dockerfile` first. If those lines are present, stop and ask the user whether they would rather use PyPI directly before building. Do not decide for them. If they choose PyPI, remove the `pip config set` line and the `COPY misc/.condarc` line from the Dockerfile. In the current working copy these lines were already removed at the user's request, so a build uses PyPI.

## Commands

```bash
# one-time: merge raw TRANCE images + meta into /data/trance/data.h5
python scripts/preprocess.py /data/trance

# train (Hydra); experiments live in conf/experiment/<problem>_<model>.yaml
python train.py experiment=event_cnn_concat logging.wandb.tags="[event, base]"
python train.py experiment=event_cnn_concat debug=overfit     # see conf/debug/*
python scripts/batch_train.py scripts/training/train_models.sh --gpus 0,1,2,3

# re-test a finished run (loads config + ckpt from the logdir)
python test.py <logdir> --ckpt last --update_func test_original [--update_wandb]

# generate data (inside blender container, from trance/)
blender --background --python render.py -- --config configs/standard.yaml --gpu false --render_tile_size 16

# lint/format (pre-commit: black & flake8 at line-length 80, isort, docformatter, prettier for yaml)
pre-commit run -a

# tests (pytest config in setup.cfg; needs the dataset at /data/trance and PROJECT set)
pytest tests/test_data.py -k event
```

Note: `tests/test_conf.py` is stale template code (asserts `src.model.lenet.LeNet`, which does not exist) and will fail.

To resume training, set `hydra.run.dir=<previous_log_dir>`; `try_resume` in `src/utils/exptool.py` then replaces `cfg` with the stored one.

## Architecture

**Config-driven composition.** `train.py` is a thin Hydra entry point: it instantiates `cfg.dataset` (a `LightningDataModule`), `cfg.pipeline` (a `LightningModule`, constructed with the whole `cfg`, `_recursive_=False`), and a `pl.Trainer` from `cfg.pl_trainer`. `conf/train.yaml` holds the defaults list; `conf/experiment/*.yaml` (`# @package _global_`) override `/dataset`, `/model`, `/criterion` per experiment. Class paths are given by `_target_`, so adding a model/dataset means adding a class plus a yaml group entry. `exp_id`/logdir names derive from the `_target_` tails (see `register_omegaconf_resolver`).

**Three task settings** (`dataset` / `criterion` groups of the same name):
- `basic` – single-step transformation (`dataset.trance.Basic`).
- `event` – multi-step sequence up to `max_step=4` (`Event`); order handled by RNN/Transformer decoders with optional teacher forcing; `event_reinforce` criterion adds REINFORCE training.
- `view` – like event but final image from a different camera (`View` subclasses `Event`; final cameras are Left/Center/Right).

**Data flow** (`src/dataset/trance.py`): `TRANCEDataModule` builds per-split datasets per `dataset_cfg.problems`; images come from `data.h5` (or raw files if `read_raw_image`). Transformations are encoded as "pair" strings (`attr.value`, e.g. `position.<direction>.<step>`) via `values.json`; `src/dataset/components/vectorizer.py` converts them to indices.

**Model** (`src/model/trancenet.py`, `components/`): all architectures (CNN/ResNet × Concat/Subtract, BCNN, DUDA, `*Former` Transformer variants using `x_transformer.py`) live in one file, selected by `conf/model/*.yaml`.

**Training loop** (`src/pipeline/reason.py`, `ReasonLitModule`): a single `step()` shared by train/val/test; metrics are accumulated by `ReasonCriterion` (`src/criterion/reason.py`) using `Evaluator`/`Recorder` components from `src/criterion/components/`. Optimizer/scheduler come from `conf/optim`, `conf/scheduler`.

**Evaluation / analysis**: `test.py` re-loads a run via `Experiment` (`src/utils/exptool.py`), applies named config-override functions (`test_original`, or custom ones such as swapping the dataset), and appends to `<logdir>/results.jsonl`. `notebooks/`, `scripts/gen_latex_table.py` consume those results. `src/demo/` is a FastAPI server (`uvicorn src.demo.api_server.main:app`) plus a Vue UI for browsing the dataset and predictions.

**Generator** (`trance/`): `render.py` (`Renderer`) loads the Blender base scene, samples an initial state (`sample_init_state`), applies 1–4 atomic transformations (`try_pair`/`transform`, with the observability rules documented above `build_sample`), writes a per-sample JSON via `utils/record.py`, then renders images (`render_from_json`). Behaviour is controlled by `trance/configs/standard.yaml` (seed, transformation counts, cameras, output dir). `trance/option.py` defines the attribute/position options loaded from `resource/{properties,values}.json`.

## Repo quirks

- The working tree shows nearly every file as modified; the diffs are line-ending (CRLF) changes from the Windows/WSL mount, not content changes. Use `git diff --ignore-cr-at-eol` to see real changes, and avoid mass-reformatting files.
- Shell scripts copied into the docker image (`docker/misc/init_workspace`, `docker/misc/.localrc`) must keep LF line endings. If a Windows-side git checkout turns them into CRLF, the image build fails with `/usr/bin/bash^M: bad interpreter` (exit code 126). Fix with `sed -i 's/\r$//' <file>`.
- `conf/local/` is for untracked machine-specific overrides (`local/default.yaml`, optional).
