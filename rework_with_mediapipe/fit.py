from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import MediaPipeReviewConfig


@dataclass
class SideObservations:
    chain_index: int
    frame_indices: List[int]
    landmarks_2d: np.ndarray
    landmarks_3d: np.ndarray


def _ensure_torch_and_manopth(cfg: MediaPipeReviewConfig):
    repo_root = str(cfg.manopth_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        import torch  # type: ignore
        from manopth.manolayer import ManoLayer  # type: ignore
    except ImportError as exc:
        raise RuntimeError("缺少 torch/manopth 依赖，无法运行 MANO 拟合。") from exc
    return torch, ManoLayer


def _rotmat_to_rot6(rotmat) -> np.ndarray:
    return rotmat[:, :2].reshape(-1).astype(np.float32)


def _axis_angle_to_rot6(axis_angle, torch) -> np.ndarray:
    angle = torch.norm(axis_angle) + 1e-8
    axis = axis_angle / angle
    x, y, z = axis
    c = torch.cos(angle)
    s = torch.sin(angle)
    C = 1 - c
    rot = torch.stack(
        [
            torch.stack([x * x * C + c, x * y * C - z * s, x * z * C + y * s]),
            torch.stack([y * x * C + z * s, y * y * C + c, y * z * C - x * s]),
            torch.stack([z * x * C - y * s, z * y * C + x * s, z * z * C + c]),
        ]
    )
    return _rotmat_to_rot6(rot.detach().cpu().numpy())


def _chain_from_choice(bundle: Dict, choice: str) -> Optional[Dict]:
    if choice in {"none", "unsure"}:
        return None
    prefix, _, raw_index = choice.partition(":")
    if prefix != "chain":
        return None
    chain_index = int(raw_index)
    for chain in bundle["chains"]:
        if int(chain["chain_index"]) == chain_index:
            return chain
    return None


def _build_side_observations(bundle: Dict, choice: str) -> Optional[SideObservations]:
    chain = _chain_from_choice(bundle, choice)
    if chain is None:
        return None
    proposals = sorted(chain["proposals"], key=lambda item: item["frame_idx"])
    return SideObservations(
        chain_index=int(chain["chain_index"]),
        frame_indices=[int(item["frame_idx"]) for item in proposals],
        landmarks_2d=np.asarray([item["landmarks_2d"] for item in proposals], dtype=np.float32),
        landmarks_3d=np.asarray([item["landmarks_3d_rel"] for item in proposals], dtype=np.float32),
    )


def _fit_similarity_xy(pred_xy, obs_xy):
    pred_center = pred_xy.mean(axis=0, keepdims=True)
    obs_center = obs_xy.mean(axis=0, keepdims=True)
    pred0 = pred_xy - pred_center
    obs0 = obs_xy - obs_center
    scale = float(np.linalg.norm(obs0) / max(np.linalg.norm(pred0), 1e-6))
    aligned = pred0 * scale + obs_center
    err = np.linalg.norm(aligned - obs_xy, axis=-1).mean()
    return scale, err


def fit_side(side: str, obs: SideObservations, cfg: MediaPipeReviewConfig) -> Dict:
    torch, ManoLayer = _ensure_torch_and_manopth(cfg)
    device = torch.device(cfg.fit_device)
    mano_layer = ManoLayer(
        use_pca=cfg.mano_use_pca,
        ncomps=cfg.mano_ncomps,
        center_idx=cfg.mano_center_idx,
        flat_hand_mean=cfg.mano_flat_hand_mean,
        side=side,
        mano_root=str(cfg.mano_models_root),
    ).to(device)
    T = len(obs.frame_indices)
    pose_pca = torch.zeros((T, cfg.mano_ncomps), device=device, requires_grad=True)
    global_orient = torch.zeros((T, 3), device=device, requires_grad=True)
    betas = torch.zeros((1, 10), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([pose_pca, global_orient, betas], lr=cfg.fit_learning_rate)
    obs_3d = torch.tensor(obs.landmarks_3d, dtype=torch.float32, device=device)
    obs_2d = torch.tensor(obs.landmarks_2d[..., :2], dtype=torch.float32, device=device)
    prev_pred = None
    for stage_idx, steps in enumerate((cfg.fit_stage1_steps, cfg.fit_stage2_steps, cfg.fit_stage3_steps)):
        for _ in range(steps):
            optimizer.zero_grad()
            theta = torch.cat([global_orient, pose_pca], dim=1)
            _, joints_mm = mano_layer(theta, betas.repeat(T, 1))
            pred = joints_mm / 1000.0
            pred_rel = pred - pred[:, :1]
            obs_rel = obs_3d - obs_3d[:, :1]
            numer = (obs_rel * pred_rel).sum(dim=(1, 2), keepdim=True)
            denom = (pred_rel * pred_rel).sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            scale = numer / denom
            rel3d_loss = ((pred_rel * scale - obs_rel) ** 2).mean()

            pred_xy = pred[:, :, :2]
            pred_center = pred_xy.mean(dim=1, keepdim=True)
            obs_center = obs_2d.mean(dim=1, keepdim=True)
            pred_xy0 = pred_xy - pred_center
            obs_xy0 = obs_2d - obs_center
            scale2d = (
                torch.linalg.norm(obs_xy0.reshape(T, -1), dim=1, keepdim=True)
                / torch.linalg.norm(pred_xy0.reshape(T, -1), dim=1, keepdim=True).clamp_min(1e-6)
            ).view(T, 1, 1)
            proj_xy = pred_xy0 * scale2d + obs_center
            reproj_loss = ((proj_xy - obs_2d) ** 2).mean()

            temporal_loss = torch.tensor(0.0, device=device)
            if prev_pred is not None and stage_idx >= 1:
                temporal_loss = ((pred_rel[1:] - pred_rel[:-1]) ** 2).mean()
            pose_prior = (pose_pca ** 2).mean()
            shape_prior = (betas ** 2).mean()
            loss = (
                cfg.fit_relative_3d_weight * rel3d_loss
                + cfg.fit_reprojection_weight * reproj_loss
                + cfg.fit_temporal_weight * temporal_loss
                + cfg.fit_pose_prior_weight * pose_prior
                + cfg.fit_shape_weight * shape_prior
            )
            loss.backward()
            optimizer.step()
            prev_pred = pred_rel.detach()

    with torch.no_grad():
        theta = torch.cat([global_orient, pose_pca], dim=1)
        _, joints_mm = mano_layer(theta, betas.repeat(T, 1))
        joints = (joints_mm / 1000.0).cpu().numpy()
        hand_full = (
            pose_pca.cpu().numpy() @ mano_layer.th_selected_comps.cpu().numpy()
            + mano_layer.th_hands_mean.cpu().numpy()
        ).astype(np.float32)
        reproj_errors = []
        wrist_cam = []
        rot6 = []
        for frame_idx in range(T):
            pred_xy = joints[frame_idx, :, :2]
            obs_xy = obs.landmarks_2d[frame_idx, :, :2]
            scale2d, error = _fit_similarity_xy(pred_xy, obs_xy)
            reproj_errors.append(float(error))
            wrist_xy = obs.landmarks_2d[frame_idx, 0, :2]
            pseudo_z = 1.0 / max(scale2d, 1e-6)
            wrist_cam.append([float(wrist_xy[0] * 2 - 1), float(wrist_xy[1] * 2 - 1), float(pseudo_z)])
            rot6.append(_axis_angle_to_rot6(global_orient[frame_idx], torch))
        return {
            "side": side,
            "chain_index": obs.chain_index,
            "frame_indices": obs.frame_indices,
            "hand_pose_pca": pose_pca.detach().cpu().numpy().astype(np.float32).tolist(),
            "hand_pose_full": hand_full.tolist(),
            "global_orient": global_orient.detach().cpu().numpy().astype(np.float32).tolist(),
            "betas": betas.detach().cpu().numpy().astype(np.float32)[0].tolist(),
            "wrist_cam": wrist_cam,
            "wrist_rot6": [item.tolist() for item in rot6],
            "reproj_error": reproj_errors,
            "valid_ratio": 1.0,
        }


def fit_reviewed_clip(bundle: Dict, review: Dict, cfg: MediaPipeReviewConfig) -> Dict:
    left_obs = _build_side_observations(bundle, review["left_choice"])
    right_obs = _build_side_observations(bundle, review["right_choice"])
    fit = {
        "clip_id": bundle["clip_id"],
        "left_choice": review["left_choice"],
        "right_choice": review["right_choice"],
        "sides": {},
    }
    if left_obs is not None:
        fit["sides"]["left"] = fit_side("left", left_obs, cfg)
    if right_obs is not None:
        fit["sides"]["right"] = fit_side("right", right_obs, cfg)
    if review["left_choice"] == "unsure" or review["right_choice"] == "unsure":
        fit["status"] = "qa_needed"
        fit["message"] = "At least one hand marked as unsure."
        return fit
    if not fit["sides"]:
        fit["status"] = "fit_ok_negative"
        fit["message"] = "No wearer hand visible in this clip."
        fit["median_reproj_error"] = 0.0
        fit["p95_reproj_error"] = 0.0
        return fit
    all_errors = []
    for side_fit in fit["sides"].values():
        all_errors.extend(side_fit["reproj_error"])
    median_err = float(np.median(all_errors)) if all_errors else math.inf
    p95_err = float(np.percentile(all_errors, 95)) if all_errors else math.inf
    fit["median_reproj_error"] = median_err
    fit["p95_reproj_error"] = p95_err
    fit["status"] = (
        "fit_ok"
        if median_err <= cfg.max_median_reproj_error_px and p95_err <= cfg.max_p95_reproj_error_px
        else "qa_needed"
    )
    return fit


def save_fit_artifacts(bundle_dir: Path, fit_payload: Dict) -> Dict[str, str]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    json_path = bundle_dir / "fit.json"
    npz_path = bundle_dir / "fit.npz"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(fit_payload, f, ensure_ascii=False, indent=2)

    arrays = {}
    for side in ("left", "right"):
        payload = fit_payload.get("sides", {}).get(side)
        prefix = f"{side}_"
        if payload is None:
            arrays[prefix + "frame_indices"] = np.zeros((0,), dtype=np.int32)
            arrays[prefix + "hand_pose_pca"] = np.zeros((0, 15), dtype=np.float32)
            arrays[prefix + "hand_pose_full"] = np.zeros((0, 45), dtype=np.float32)
            arrays[prefix + "global_orient"] = np.zeros((0, 3), dtype=np.float32)
            arrays[prefix + "betas"] = np.zeros((10,), dtype=np.float32)
            arrays[prefix + "wrist_cam"] = np.zeros((0, 3), dtype=np.float32)
            arrays[prefix + "wrist_rot6"] = np.zeros((0, 6), dtype=np.float32)
            arrays[prefix + "reproj_error"] = np.zeros((0,), dtype=np.float32)
            continue
        arrays[prefix + "frame_indices"] = np.asarray(payload["frame_indices"], dtype=np.int32)
        arrays[prefix + "hand_pose_pca"] = np.asarray(payload["hand_pose_pca"], dtype=np.float32)
        arrays[prefix + "hand_pose_full"] = np.asarray(payload["hand_pose_full"], dtype=np.float32)
        arrays[prefix + "global_orient"] = np.asarray(payload["global_orient"], dtype=np.float32)
        arrays[prefix + "betas"] = np.asarray(payload["betas"], dtype=np.float32)
        arrays[prefix + "wrist_cam"] = np.asarray(payload["wrist_cam"], dtype=np.float32)
        arrays[prefix + "wrist_rot6"] = np.asarray(payload["wrist_rot6"], dtype=np.float32)
        arrays[prefix + "reproj_error"] = np.asarray(payload["reproj_error"], dtype=np.float32)
    arrays["median_reproj_error"] = np.asarray([fit_payload.get("median_reproj_error", 0.0)], dtype=np.float32)
    arrays["p95_reproj_error"] = np.asarray([fit_payload.get("p95_reproj_error", 0.0)], dtype=np.float32)
    np.savez_compressed(npz_path, **arrays)
    return {"fit_json": str(json_path), "fit_npz": str(npz_path)}
