from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import AppConfig


@dataclass
class SideObservations:
    chain_index: int
    frame_indices: List[int]
    landmarks_2d: np.ndarray
    landmarks_3d: np.ndarray


def _ensure_torch_and_manopth(cfg: AppConfig):
    repo_root = str(cfg.manopth_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import torch  # type: ignore
    from manopth.manolayer import ManoLayer  # type: ignore

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
    frame_indices = [int(item["frame_idx"]) for item in proposals]
    landmarks_2d = np.asarray([item["landmarks_2d"] for item in proposals], dtype=np.float32)
    landmarks_3d = np.asarray([item["landmarks_3d_rel"] for item in proposals], dtype=np.float32)
    return SideObservations(
        chain_index=int(chain["chain_index"]),
        frame_indices=frame_indices,
        landmarks_2d=landmarks_2d,
        landmarks_3d=landmarks_3d,
    )


def _fit_similarity_xy(pred_xy, obs_xy):
    pred_center = pred_xy.mean(axis=0, keepdims=True)
    obs_center = obs_xy.mean(axis=0, keepdims=True)
    pred0 = pred_xy - pred_center
    obs0 = obs_xy - obs_center
    scale = float(np.linalg.norm(obs0) / max(np.linalg.norm(pred0), 1e-6))
    aligned = pred0 * scale + obs_center
    err = np.linalg.norm(aligned - obs_xy, axis=-1).mean()
    return scale, obs_center[0] - pred_center[0] * scale, err


def fit_side(side: str, obs: SideObservations, cfg: AppConfig) -> Dict:
    torch, ManoLayer = _ensure_torch_and_manopth(cfg)
    device = torch.device(str(cfg.fit.get("device", "cpu")))
    mano_cfg = cfg.raw.get("mano", {})
    mano_layer = ManoLayer(
        use_pca=bool(mano_cfg.get("use_pca", True)),
        ncomps=int(mano_cfg.get("ncomps", 15)),
        center_idx=int(mano_cfg.get("center_idx", 0)),
        flat_hand_mean=bool(mano_cfg.get("flat_hand_mean", True)),
        side=side,
        mano_root=str(cfg.mano_models_root),
    ).to(device)
    T = len(obs.frame_indices)
    pose_pca = torch.zeros((T, 15), device=device, requires_grad=True)
    global_orient = torch.zeros((T, 3), device=device, requires_grad=True)
    betas = torch.zeros((1, 10), device=device, requires_grad=True)
    optimizer = torch.optim.Adam(
        [pose_pca, global_orient, betas],
        lr=float(cfg.fit.get("learning_rate", 0.03)),
    )
    obs_3d = torch.tensor(obs.landmarks_3d, dtype=torch.float32, device=device)
    obs_2d = torch.tensor(obs.landmarks_2d[..., :2], dtype=torch.float32, device=device)
    prev_pred = None
    stage_steps = [
        int(cfg.fit.get("stage1_steps", 80)),
        int(cfg.fit.get("stage2_steps", 150)),
        int(cfg.fit.get("stage3_steps", 200)),
    ]
    for stage_idx, steps in enumerate(stage_steps):
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
                float(cfg.fit.get("relative_3d_weight", 2.0)) * rel3d_loss
                + float(cfg.fit.get("reprojection_weight", 5.0)) * reproj_loss
                + float(cfg.fit.get("temporal_weight", 0.05)) * temporal_loss
                + float(cfg.fit.get("pose_prior_weight", 0.001)) * pose_prior
                + float(cfg.fit.get("shape_weight", 0.001)) * shape_prior
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
            scale2d, trans2d, error = _fit_similarity_xy(pred_xy, obs_xy)
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


def fit_reviewed_clip(bundle: Dict, review: Dict, cfg: AppConfig) -> Dict:
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
        if median_err <= float(cfg.export.get("max_median_reproj_error_px", 18.0))
        and p95_err <= float(cfg.export.get("max_p95_reproj_error_px", 40.0))
        else "qa_needed"
    )
    return fit
