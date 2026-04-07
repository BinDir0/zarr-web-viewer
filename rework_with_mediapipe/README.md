# rework_with_mediapipe

Independent review and recovery pipeline for egocentric dirty clips.

The workflow is:

1. Discover dirty clips from `zarr-viewer` annotations.
2. Precompute MediaPipe all-hand proposals and candidate chains offline.
3. Review clips in a lightweight Flask app where annotators choose wearer-left / wearer-right / none.
4. Fit MANO only after review.
5. Export hand-only training zarr files.

## Structure

- `app/`: Flask app, SQLite schema, and review APIs.
- `pipeline/`: dirty clip discovery, frame loading, MediaPipe proposals, and chain building.
- `jobs/`: CLI entrypoints for preprocess and fit/export.
- `mano/`: MANO fitting code.
- `export/`: zarr export helpers.
- `config/`: YAML configuration.

## Quick Start

```bash
cd /path/to/zarr-web-viewer/rework_with_mediapipe
python3 -m jobs.run_preprocess --discover --process --limit 50
python3 -m app.server
```

Then open `http://127.0.0.1:9481`.

## Environment Setup

1. Install Python dependencies from `requirements.txt`.
2. Put a MediaPipe hand landmarker task model at:

```bash
rework_with_mediapipe/cache/hand_landmarker.task
```

3. Make sure the parent `zarr-web-viewer` repo has a valid:
   - `annotations.db`
   - `config.yaml`

The dirty clip discovery logic reads those two files directly from the parent repo.

## Data Wiring

- `rework_with_mediapipe` does not scan raw data on its own.
- It reads dirty episodes from the parent repo's `annotations.db`.
- It reads the factory / legacy dataset roots from the parent repo's `config.yaml`.
- For factory data, it expects `_video_index.json` and the tar shards described in that config.
- For legacy raw data, it expects `extracted_images/*.jpg` under the resolved crop directory.

## Local Overrides

If your local paths differ, copy `config/local.example.yaml` to `config/local.yaml` and edit:

- `paths.mediapipe_repo_root`
- `paths.manopth_root`
- `paths.mano_models_root`

You usually do not need to override `app_root` or the parent `zarr-viewer` paths unless you moved the directory layout.

## Notes

- Review is designed to be fast. Annotators only choose left/right/none from precomputed candidate chains.
- No online MediaPipe or MANO fitting runs inside the review page.
- Complex clips are routed to `qa_needed` instead of forcing heavy annotation.
- Put environment-specific path overrides in `config/local.yaml` if your `mediapipe` or `manopth` repositories live elsewhere.
