from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from app.config import AppConfig
from app.db import connect


def _load_bundle(cfg: AppConfig, bundle_relpath: str) -> Dict:
    with open(cfg.app_root / bundle_relpath / "bundle.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _read_frame(cfg: AppConfig, relpath: str) -> np.ndarray:
    return np.asarray(Image.open(cfg.app_root / relpath).convert("RGB"), dtype=np.uint8)


def export_fit_ok_clips(cfg: AppConfig) -> Dict:
    import zarr

    output_path = cfg.exports_dir / str(cfg.export.get("target_name", "hard_hand_clips.zarr"))
    conn = connect(cfg.app_db)
    rows = conn.execute(
        "SELECT * FROM clips WHERE status = 'fit_ok' AND fit_payload_json IS NOT NULL ORDER BY id"
    ).fetchall()
    images: List[np.ndarray] = []
    hand_full: List[np.ndarray] = []
    hand_pca: List[np.ndarray] = []
    wrist_cam: List[np.ndarray] = []
    shape: List[np.ndarray] = []
    landmarks_2d: List[np.ndarray] = []
    landmarks_3d_rel: List[np.ndarray] = []
    valid: List[np.ndarray] = []
    reproj_error: List[np.ndarray] = []
    episode_ends: List[int] = []

    for row in rows:
        bundle = _load_bundle(cfg, row["bundle_relpath"])
        fit = json.loads(row["fit_payload_json"])
        left = fit["sides"].get("left")
        right = fit["sides"].get("right")
        frame_lookup = {int(frame["frame_idx"]): frame["relpath"] for frame in bundle["frames"]}
        left_lookup = {}
        right_lookup = {}
        if left:
            for idx, frame_idx in enumerate(left["frame_indices"]):
                left_lookup[int(frame_idx)] = idx
        if right:
            for idx, frame_idx in enumerate(right["frame_indices"]):
                right_lookup[int(frame_idx)] = idx

        clip_frames = sorted(frame_lookup.keys())
        for frame_idx in clip_frames:
            images.append(_read_frame(cfg, frame_lookup[frame_idx]))
            full_row = np.zeros((90,), dtype=np.float32)
            pca_row = np.zeros((30,), dtype=np.float32)
            wrist_row = np.zeros((18,), dtype=np.float32)
            shape_row = np.zeros((20,), dtype=np.float32)
            lm2_row = np.zeros((2, 21, 3), dtype=np.float32)
            lm3_row = np.zeros((2, 21, 3), dtype=np.float32)
            valid_row = np.zeros((2,), dtype=np.bool_)
            err_row = np.full((2,), np.nan, dtype=np.float32)

            if left and frame_idx in left_lookup:
                idx = left_lookup[frame_idx]
                full_row[:45] = np.asarray(left["hand_pose_full"][idx], dtype=np.float32)
                pca_row[:15] = np.asarray(left["hand_pose_pca"][idx], dtype=np.float32)
                wrist_row[:3] = np.asarray(left["wrist_cam"][idx], dtype=np.float32)
                wrist_row[6:12] = np.asarray(left["wrist_rot6"][idx], dtype=np.float32)
                shape_row[:10] = np.asarray(left["betas"], dtype=np.float32)
                valid_row[0] = True
                err_row[0] = float(left["reproj_error"][idx])
            if right and frame_idx in right_lookup:
                idx = right_lookup[frame_idx]
                full_row[45:] = np.asarray(right["hand_pose_full"][idx], dtype=np.float32)
                pca_row[15:] = np.asarray(right["hand_pose_pca"][idx], dtype=np.float32)
                wrist_row[3:6] = np.asarray(right["wrist_cam"][idx], dtype=np.float32)
                wrist_row[12:18] = np.asarray(right["wrist_rot6"][idx], dtype=np.float32)
                shape_row[10:] = np.asarray(right["betas"], dtype=np.float32)
                valid_row[1] = True
                err_row[1] = float(right["reproj_error"][idx])

            chain_by_index = {int(chain["chain_index"]): chain for chain in bundle["chains"]}
            if left and frame_idx in left_lookup:
                proposal = chain_by_index[left["chain_index"]]["proposals"][left_lookup[frame_idx]]
                lm2_row[0] = np.asarray(proposal["landmarks_2d"], dtype=np.float32)
                lm3_row[0] = np.asarray(proposal["landmarks_3d_rel"], dtype=np.float32)
            if right and frame_idx in right_lookup:
                proposal = chain_by_index[right["chain_index"]]["proposals"][right_lookup[frame_idx]]
                lm2_row[1] = np.asarray(proposal["landmarks_2d"], dtype=np.float32)
                lm3_row[1] = np.asarray(proposal["landmarks_3d_rel"], dtype=np.float32)

            hand_full.append(full_row)
            hand_pca.append(pca_row)
            wrist_cam.append(wrist_row)
            shape.append(shape_row)
            landmarks_2d.append(lm2_row)
            landmarks_3d_rel.append(lm3_row)
            valid.append(valid_row)
            reproj_error.append(err_row)
        episode_ends.append(len(images))

    root = zarr.open(str(output_path), mode="w")
    data = root.create_group("data")
    state = data.create_group("state")
    aux = data.create_group("aux")
    mask = data.create_group("mask")
    qc = data.create_group("qc")
    meta = root.create_group("meta")

    if images:
        data.create_dataset("image", data=np.stack(images), overwrite=True)
        state.create_dataset("hand_full", data=np.stack(hand_full), overwrite=True)
        state.create_dataset("hand_pca", data=np.stack(hand_pca), overwrite=True)
        state.create_dataset("wrist_cam", data=np.stack(wrist_cam), overwrite=True)
        state.create_dataset("shape", data=np.stack(shape), overwrite=True)
        aux.create_dataset("landmarks_2d", data=np.stack(landmarks_2d), overwrite=True)
        aux.create_dataset("landmarks_3d_rel", data=np.stack(landmarks_3d_rel), overwrite=True)
        mask.create_dataset("valid", data=np.stack(valid), overwrite=True)
        qc.create_dataset("reproj_error", data=np.stack(reproj_error), overwrite=True)
    meta.create_dataset("episode_ends", data=np.asarray(episode_ends, dtype=np.int64), overwrite=True)
    if rows:
        conn.executemany(
            "UPDATE clips SET status = 'exported', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            [(int(row["id"]),) for row in rows],
        )
        conn.commit()
    conn.close()
    return {
        "status": "ok",
        "output_path": str(output_path),
        "num_clips": len(rows),
        "num_frames": len(images),
    }
