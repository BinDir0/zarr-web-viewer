from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import MediaPipeReviewConfig


@dataclass
class SideObservations:
    track_id: int
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


def _track_lookup(bundle: Dict) -> Dict[int, Dict]:
    return {int(track["track_id"]): track for track in bundle.get("tracks", [])}


def _build_side_observation(bundle: Dict, track_id: int) -> Optional[SideObservations]:
    track = _track_lookup(bundle).get(int(track_id))
    if track is None:
        return None
    proposals = sorted(track["proposals"], key=lambda item: item["frame_idx"])
    return SideObservations(
        track_id=int(track["track_id"]),
        frame_indices=[int(item["frame_idx"]) for item in proposals],
        landmarks_2d=np.asarray([item["landmarks_2d"] for item in proposals], dtype=np.float32),
        landmarks_3d=np.asarray([item["landmarks_3d_rel"] for item in proposals], dtype=np.float32),
    )


def _track_proposal_lookup(bundle: Dict) -> Dict[int, Dict[int, Dict]]:
    lookup: Dict[int, Dict[int, Dict]] = {}
    for track in bundle.get("tracks", []):
        proposal_lookup = {
            int(item["frame_idx"]): item
            for item in sorted(track.get("proposals", []), key=lambda item: int(item["frame_idx"]))
        }
        lookup[int(track["track_id"])] = proposal_lookup
    return lookup


def _build_side_observation_from_frames(
    proposal_lookup: Dict[int, Dict[int, Dict]],
    track_id: int,
    frame_indices: Sequence[int],
) -> Optional[SideObservations]:
    frame_list = [int(frame_idx) for frame_idx in frame_indices]
    track_proposals = proposal_lookup.get(int(track_id), {})
    proposals = [track_proposals[frame_idx] for frame_idx in frame_list if frame_idx in track_proposals]
    if not proposals:
        return None
    return SideObservations(
        track_id=int(track_id),
        frame_indices=[int(item["frame_idx"]) for item in proposals],
        landmarks_2d=np.asarray([item["landmarks_2d"] for item in proposals], dtype=np.float32),
        landmarks_3d=np.asarray([item["landmarks_3d_rel"] for item in proposals], dtype=np.float32),
    )


def _normalize_keyframe_state(review: Dict) -> Dict[str, Optional[int] | bool]:
    return {
        "left_track_id": None if review.get("left_track_id") is None else int(review["left_track_id"]),
        "right_track_id": None if review.get("right_track_id") is None else int(review["right_track_id"]),
        "left_missing_box": bool(review.get("left_missing_box", False)),
        "right_missing_box": bool(review.get("right_missing_box", False)),
    }


def _state_signature(state: Dict[str, Optional[int] | bool]) -> tuple:
    return (
        state["left_track_id"],
        state["right_track_id"],
        bool(state["left_missing_box"]),
        bool(state["right_missing_box"]),
    )


def _frame_assignments_from_keyframes(bundle: Dict, review: Dict) -> Dict[str, object]:
    ordered_frames = [int(item["frame_idx"]) for item in bundle.get("frames", [])]
    if not ordered_frames:
        return {
            "left_missing_box": False,
            "right_missing_box": False,
            "left_track_ids": [],
            "right_track_ids": [],
            "assignment_segments": [],
            "kept_segments": [],
            "dropped_segments": [],
            "valid_frame_ranges": [],
            "frame_roles": {},
        }
    frame_to_pos = {frame_idx: pos for pos, frame_idx in enumerate(ordered_frames)}
    keyframe_reviews = {
        int(item["frame_idx"]): item
        for item in sorted(review.get("keyframe_reviews", []), key=lambda item: int(item["frame_idx"]))
    }
    bundle_keyframes = sorted(bundle.get("keyframes", []), key=lambda item: int(item["frame_idx"]))
    segments = sorted(bundle.get("segments", []), key=lambda item: int(item["start_frame"]))
    if not segments:
        segments = [
            {
                "segment_id": 0,
                "start_frame": int(ordered_frames[0]),
                "end_frame": int(ordered_frames[-1]),
            }
        ]
        if not bundle_keyframes:
            bundle_keyframes = [{"frame_idx": int(ordered_frames[0]), "segment_id": 0, "kind": "anchor"}]

    default_role = {
        "left_track_id": None,
        "right_track_id": None,
        "left_missing_box": False,
        "right_missing_box": False,
        "dropped": False,
    }
    frame_roles: Dict[int, Dict[str, Optional[int] | bool]] = {
        int(frame_idx): {
            **default_role,
        }
        for frame_idx in ordered_frames
    }
    assignment_segments: List[Dict] = []
    for segment in segments:
        segment_id = int(segment["segment_id"])
        segment_start = int(segment["start_frame"])
        segment_end = int(segment["end_frame"])
        segment_keyframes = [
            keyframe_reviews[int(item["frame_idx"])]
            for item in bundle_keyframes
            if int(item.get("segment_id", segment_id)) == segment_id and int(item["frame_idx"]) in keyframe_reviews
        ]
        if not segment_keyframes:
            continue
        segment_keyframes.sort(key=lambda item: int(item["frame_idx"]))
        current_state = _normalize_keyframe_state(segment_keyframes[0])
        current_start_pos = frame_to_pos[segment_start]
        for next_review in segment_keyframes[1:]:
            next_frame = int(next_review["frame_idx"])
            next_state = _normalize_keyframe_state(next_review)
            if _state_signature(next_state) == _state_signature(current_state):
                continue
            next_pos = frame_to_pos[next_frame]
            chunk_end_pos = max(current_start_pos, next_pos - 1)
            assignment_segments.append(
                {
                    "segment_id": segment_id,
                    "start_frame": int(ordered_frames[current_start_pos]),
                    "end_frame": int(ordered_frames[chunk_end_pos]),
                    **current_state,
                }
            )
            current_start_pos = next_pos
            current_state = next_state
        assignment_segments.append(
            {
                "segment_id": segment_id,
                "start_frame": int(ordered_frames[current_start_pos]),
                "end_frame": segment_end,
                **current_state,
            }
        )

    dropped_segments: List[Dict] = []
    kept_segments: List[Dict] = []
    valid_frame_ranges: List[List[int]] = []
    for item in assignment_segments:
        is_dropped = bool(item["left_missing_box"]) or bool(item["right_missing_box"])
        segment_payload = {**item, "dropped": is_dropped}
        if is_dropped:
            dropped_segments.append(segment_payload)
        else:
            kept_segments.append(segment_payload)
            valid_frame_ranges.append([int(item["start_frame"]), int(item["end_frame"])])
        start_pos = frame_to_pos[int(item["start_frame"])]
        end_pos = frame_to_pos[int(item["end_frame"])]
        for pos in range(start_pos, end_pos + 1):
            frame_roles[int(ordered_frames[pos])] = {
                "left_track_id": item["left_track_id"],
                "right_track_id": item["right_track_id"],
                "left_missing_box": bool(item["left_missing_box"]),
                "right_missing_box": bool(item["right_missing_box"]),
                "dropped": is_dropped,
            }

    left_missing_box = any(bool(item["left_missing_box"]) for item in assignment_segments)
    right_missing_box = any(bool(item["right_missing_box"]) for item in assignment_segments)
    left_track_ids = sorted(
        {
            int(item["left_track_id"])
            for item in assignment_segments
            if item.get("left_track_id") is not None
        }
    )
    right_track_ids = sorted(
        {
            int(item["right_track_id"])
            for item in assignment_segments
            if item.get("right_track_id") is not None
        }
    )
    return {
        "left_missing_box": left_missing_box,
        "right_missing_box": right_missing_box,
        "left_track_ids": left_track_ids,
        "right_track_ids": right_track_ids,
        "assignment_segments": assignment_segments,
        "kept_segments": kept_segments,
        "dropped_segments": dropped_segments,
        "valid_frame_ranges": valid_frame_ranges,
        "frame_roles": frame_roles,
    }


def _build_side_observations_from_frame_roles(bundle: Dict, frame_roles: Dict[int, Dict[str, Optional[int] | bool]], side: str) -> List[SideObservations]:
    proposal_lookup = _track_proposal_lookup(bundle)
    ordered_frames = [int(item["frame_idx"]) for item in bundle.get("frames", [])]
    observations: List[SideObservations] = []
    current_track_id: Optional[int] = None
    current_frames: List[int] = []
    role_key = f"{side}_track_id"
    for frame_idx in ordered_frames:
        role = frame_roles.get(int(frame_idx), {})
        if bool(role.get("dropped", False)):
            track_id = None
        else:
            track_id = role.get(role_key)
        track_id = None if track_id is None else int(track_id)
        if current_track_id is not None and track_id == current_track_id:
            current_frames.append(int(frame_idx))
            continue
        if current_track_id is not None and current_frames:
            obs = _build_side_observation_from_frames(proposal_lookup, current_track_id, current_frames)
            if obs is not None:
                observations.append(obs)
        current_track_id = track_id
        current_frames = [int(frame_idx)] if track_id is not None else []
    if current_track_id is not None and current_frames:
        obs = _build_side_observation_from_frames(proposal_lookup, current_track_id, current_frames)
        if obs is not None:
            observations.append(obs)
    return observations


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
            "track_id": obs.track_id,
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


def _build_side_observations(bundle: Dict, track_ids: Sequence[int]) -> List[SideObservations]:
    observations: List[SideObservations] = []
    for track_id in track_ids:
        obs = _build_side_observation(bundle, int(track_id))
        if obs is not None:
            observations.append(obs)
    return observations


def fit_reviewed_clip(bundle: Dict, review: Dict, cfg: MediaPipeReviewConfig) -> Dict:
    review_version = str(review.get("review_version") or "")
    assignment_segments: List[Dict] = []
    if review_version == "keyframe_v1":
        derived = _frame_assignments_from_keyframes(bundle, review)
        left_track_ids = list(derived["left_track_ids"])
        right_track_ids = list(derived["right_track_ids"])
        left_missing_box = bool(derived["left_missing_box"])
        right_missing_box = bool(derived["right_missing_box"])
        assignment_segments = list(derived["assignment_segments"])
        kept_segments = list(derived["kept_segments"])
        dropped_segments = list(derived["dropped_segments"])
        valid_frame_ranges = list(derived["valid_frame_ranges"])
        frame_roles = dict(derived["frame_roles"])
        left_observations = _build_side_observations_from_frame_roles(bundle, frame_roles, "left")
        right_observations = _build_side_observations_from_frame_roles(bundle, frame_roles, "right")
    else:
        left_track_ids = [int(item) for item in review.get("left_track_ids", [])]
        right_track_ids = [int(item) for item in review.get("right_track_ids", [])]
        left_missing_box = bool(review.get("left_missing_box", False))
        right_missing_box = bool(review.get("right_missing_box", False))
        kept_segments = []
        dropped_segments = []
        valid_frame_ranges = []
        left_observations = _build_side_observations(bundle, left_track_ids)
        right_observations = _build_side_observations(bundle, right_track_ids)
    fit = {
        "clip_id": bundle["clip_id"],
        "review_version": review_version or "legacy_track_v0",
        "left_track_ids": left_track_ids,
        "right_track_ids": right_track_ids,
        "left_missing_box": left_missing_box,
        "right_missing_box": right_missing_box,
        "assignment_segments": assignment_segments,
        "kept_segments": kept_segments,
        "dropped_segments": dropped_segments,
        "valid_frame_ranges": valid_frame_ranges,
        "sides": {
            "left": {"missing_box": left_missing_box, "fragments": []},
            "right": {"missing_box": right_missing_box, "fragments": []},
        },
    }
    for obs in left_observations:
        fit["sides"]["left"]["fragments"].append(fit_side("left", obs, cfg))
    for obs in right_observations:
        fit["sides"]["right"]["fragments"].append(fit_side("right", obs, cfg))
    if not fit["sides"]["left"]["fragments"] and not fit["sides"]["right"]["fragments"]:
        if dropped_segments:
            fit["status"] = "qa_needed_missing_box"
            fit["message"] = "All usable frames were dropped because at least one governing keyframe had a missing proposal."
        else:
            fit["status"] = "fit_ok_negative"
            fit["message"] = "No wearer hand visible in this clip."
        fit["median_reproj_error"] = 0.0
        fit["p95_reproj_error"] = 0.0
        return fit
    all_errors = []
    for side_name in ("left", "right"):
        for fragment in fit["sides"][side_name]["fragments"]:
            all_errors.extend(fragment["reproj_error"])
    median_err = float(np.median(all_errors)) if all_errors else math.inf
    p95_err = float(np.percentile(all_errors, 95)) if all_errors else math.inf
    fit["median_reproj_error"] = median_err
    fit["p95_reproj_error"] = p95_err
    fit["status"] = (
        "fit_ok"
        if median_err <= cfg.max_median_reproj_error_px and p95_err <= cfg.max_p95_reproj_error_px
        else "fit_failed"
    )
    if dropped_segments:
        fit["message"] = "Some governed frame ranges were dropped because a keyframe was marked missing_box."
    return fit


def _empty_fragment_arrays(prefix: str, arrays: Dict[str, np.ndarray]) -> None:
    arrays[prefix + "fragment_track_ids"] = np.zeros((0,), dtype=np.int32)
    arrays[prefix + "fragment_offsets"] = np.zeros((1,), dtype=np.int32)
    arrays[prefix + "frame_indices"] = np.zeros((0,), dtype=np.int32)
    arrays[prefix + "hand_pose_pca"] = np.zeros((0, 15), dtype=np.float32)
    arrays[prefix + "hand_pose_full"] = np.zeros((0, 45), dtype=np.float32)
    arrays[prefix + "global_orient"] = np.zeros((0, 3), dtype=np.float32)
    arrays[prefix + "betas"] = np.zeros((0, 10), dtype=np.float32)
    arrays[prefix + "wrist_cam"] = np.zeros((0, 3), dtype=np.float32)
    arrays[prefix + "wrist_rot6"] = np.zeros((0, 6), dtype=np.float32)
    arrays[prefix + "reproj_error"] = np.zeros((0,), dtype=np.float32)


def _flatten_side_fragments(side_payload: Optional[Dict], prefix: str, arrays: Dict[str, np.ndarray]) -> None:
    fragments = list((side_payload or {}).get("fragments", []))
    arrays[prefix + "missing_box"] = np.asarray([1 if bool((side_payload or {}).get("missing_box", False)) else 0], dtype=np.int8)
    if not fragments:
        _empty_fragment_arrays(prefix, arrays)
        return

    track_ids: List[int] = []
    offsets = [0]
    frame_indices: List[np.ndarray] = []
    hand_pose_pca: List[np.ndarray] = []
    hand_pose_full: List[np.ndarray] = []
    global_orient: List[np.ndarray] = []
    betas: List[np.ndarray] = []
    wrist_cam: List[np.ndarray] = []
    wrist_rot6: List[np.ndarray] = []
    reproj_error: List[np.ndarray] = []
    total = 0
    for fragment in fragments:
        frames = np.asarray(fragment["frame_indices"], dtype=np.int32)
        total += len(frames)
        offsets.append(total)
        track_ids.append(int(fragment["track_id"]))
        frame_indices.append(frames)
        hand_pose_pca.append(np.asarray(fragment["hand_pose_pca"], dtype=np.float32))
        hand_pose_full.append(np.asarray(fragment["hand_pose_full"], dtype=np.float32))
        global_orient.append(np.asarray(fragment["global_orient"], dtype=np.float32))
        betas.append(np.asarray(fragment["betas"], dtype=np.float32))
        wrist_cam.append(np.asarray(fragment["wrist_cam"], dtype=np.float32))
        wrist_rot6.append(np.asarray(fragment["wrist_rot6"], dtype=np.float32))
        reproj_error.append(np.asarray(fragment["reproj_error"], dtype=np.float32))

    arrays[prefix + "fragment_track_ids"] = np.asarray(track_ids, dtype=np.int32)
    arrays[prefix + "fragment_offsets"] = np.asarray(offsets, dtype=np.int32)
    arrays[prefix + "frame_indices"] = np.concatenate(frame_indices, axis=0) if frame_indices else np.zeros((0,), dtype=np.int32)
    arrays[prefix + "hand_pose_pca"] = np.concatenate(hand_pose_pca, axis=0) if hand_pose_pca else np.zeros((0, 15), dtype=np.float32)
    arrays[prefix + "hand_pose_full"] = np.concatenate(hand_pose_full, axis=0) if hand_pose_full else np.zeros((0, 45), dtype=np.float32)
    arrays[prefix + "global_orient"] = np.concatenate(global_orient, axis=0) if global_orient else np.zeros((0, 3), dtype=np.float32)
    arrays[prefix + "betas"] = np.stack(betas, axis=0) if betas else np.zeros((0, 10), dtype=np.float32)
    arrays[prefix + "wrist_cam"] = np.concatenate(wrist_cam, axis=0) if wrist_cam else np.zeros((0, 3), dtype=np.float32)
    arrays[prefix + "wrist_rot6"] = np.concatenate(wrist_rot6, axis=0) if wrist_rot6 else np.zeros((0, 6), dtype=np.float32)
    arrays[prefix + "reproj_error"] = np.concatenate(reproj_error, axis=0) if reproj_error else np.zeros((0,), dtype=np.float32)


def _ranges_array(ranges: Sequence[Sequence[int]]) -> np.ndarray:
    if not ranges:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray([[int(item[0]), int(item[1])] for item in ranges], dtype=np.int32)


def save_fit_artifacts(bundle_dir: Path, fit_payload: Dict) -> Dict[str, str]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    json_path = bundle_dir / "fit.json"
    npz_path = bundle_dir / "fit.npz"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(fit_payload, f, ensure_ascii=False, indent=2)

    arrays = {}
    for side in ("left", "right"):
        prefix = f"{side}_"
        _flatten_side_fragments(fit_payload.get("sides", {}).get(side), prefix, arrays)
    arrays["valid_frame_ranges"] = _ranges_array(fit_payload.get("valid_frame_ranges", []))
    arrays["dropped_frame_ranges"] = _ranges_array(
        [[int(item["start_frame"]), int(item["end_frame"])] for item in fit_payload.get("dropped_segments", [])]
    )
    arrays["median_reproj_error"] = np.asarray([fit_payload.get("median_reproj_error", 0.0)], dtype=np.float32)
    arrays["p95_reproj_error"] = np.asarray([fit_payload.get("p95_reproj_error", 0.0)], dtype=np.float32)
    np.savez_compressed(npz_path, **arrays)
    return {"fit_json": str(json_path), "fit_npz": str(npz_path)}
