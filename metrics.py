"""
Pose evaluation metrics: position error, angular error, view IoU.
Adapted from eval_pose_iou_ang.py for the ScanNet scene layout.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None

H_FOV_DEG = 58.30  # ScanNet
V_FOV_DEG = 45.33  # ScanNet
NEAR = 0.05


def _normalize(v: np.ndarray, eps: float = 1e-9) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    return v / n if n >= eps else None


def get_gt_pose(scene_dir: Path, image_index: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    desc_path = scene_dir / "output" / "descriptions" / "all_descriptions.json"
    with open(desc_path) as f:
        descriptions = json.load(f)
    for d in descriptions:
        if str(d.get("image_index")) == str(image_index):
            pose = np.array(d["scene_pose"], dtype=np.float64)
            gt_pos = pose[:3, 3]
            gt_dir = _normalize(pose[:3, 2])  # forward = third column of R
            return gt_pos, gt_dir
    return None, None


# ── IoU context (expensive to build, cache externally) ───────────────────────

IoUContext = Tuple  # (raycasting_scene, mesh_id, tri_points, tri_centroids, tri_areas)

def build_iou_context(scene_dir: Path) -> IoUContext:
    if o3d is None:
        raise RuntimeError("open3d is required for IoU computation.")
    scene_id = scene_dir.name
    ply_path = scene_dir / f"{scene_id}_vh_clean_2.ply"
    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    if mesh.is_empty():
        raise ValueError(f"Empty mesh: {ply_path}")

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    tri_points = verts[tris]
    edge_a = tri_points[:, 1] - tri_points[:, 0]
    edge_b = tri_points[:, 2] - tri_points[:, 0]
    tri_areas = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    tri_centroids = tri_points.mean(axis=1)

    rc_scene = o3d.t.geometry.RaycastingScene()
    mesh_id = int(rc_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh)))
    return rc_scene, mesh_id, tri_points, tri_centroids, tri_areas


def _camera_axes(forward: np.ndarray) -> Optional[Tuple]:
    fwd = _normalize(np.asarray(forward, dtype=np.float64))
    if fwd is None:
        return None
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(fwd, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    right = _normalize(np.cross(fwd, up))
    if right is None:
        return None
    up_ortho = _normalize(np.cross(right, fwd))
    return (fwd, right, up_ortho) if up_ortho is not None else None


def _visible_tris(cam, forward, hfov, vfov, rc_scene, mesh_id, tri_points, tri_centroids) -> set:
    if o3d is None or forward is None:
        return set()
    axes = _camera_axes(forward)
    if axes is None:
        return set()
    fwd_ax, right_ax, up_ax = axes
    cam = np.asarray(cam, dtype=np.float64)

    rel = tri_points - cam[None, None, :]
    fwd_c = rel @ fwd_ax
    right_c = rel @ right_ax
    up_c = rel @ up_ax

    tan_h = math.tan(hfov * 0.5)
    tan_v = math.tan(vfov * 0.5)
    mask = (
        np.all(fwd_c > NEAR, axis=1) &
        np.all(np.abs(right_c) <= fwd_c * tan_h, axis=1) &
        np.all(np.abs(up_c) <= fwd_c * tan_v, axis=1)
    )
    if not np.any(mask):
        return set()

    idxs = np.nonzero(mask)[0]
    vecs = tri_centroids[idxs] - cam
    dists = np.linalg.norm(vecs, axis=1)
    valid = dists > 1e-6
    idxs = idxs[valid]
    if len(idxs) == 0:
        return set()

    dirs = vecs[valid] / dists[valid, None]
    rays = np.concatenate([np.repeat(cam[None], len(idxs), axis=0), dirs], axis=1).astype(np.float32)
    cast = rc_scene.cast_rays(o3d.core.Tensor(rays))
    prim_ids = np.asarray(cast["primitive_ids"].numpy())
    geom_ids = np.asarray(cast["geometry_ids"].numpy())
    hit = (prim_ids == idxs) & (geom_ids == mesh_id)
    return {int(i) for i in idxs[hit]}


def compute_view_iou(gt_cam, gt_dir, pred_cam, pred_dir, rc_scene, mesh_id,
                     tri_points, tri_centroids, tri_areas) -> Optional[float]:
    hfov = math.radians(H_FOV_DEG)
    vfov = math.radians(V_FOV_DEG)
    gt_vis = _visible_tris(gt_cam, gt_dir, hfov, vfov, rc_scene, mesh_id, tri_points, tri_centroids)
    pr_vis = _visible_tris(pred_cam, pred_dir, hfov, vfov, rc_scene, mesh_id, tri_points, tri_centroids)
    union = gt_vis | pr_vis
    if not union:
        return None
    inter_area = float(tri_areas[list(gt_vis & pr_vis)].sum()) if gt_vis & pr_vis else 0.0
    union_area = float(tri_areas[list(union)].sum())
    return inter_area / union_area if union_area > 1e-9 else None


def evaluate(scene_dir: Path, image_index: str,
             pred_pos: np.ndarray, pred_dir: np.ndarray,
             iou_context: Optional[IoUContext] = None) -> dict:
    gt_pos, gt_dir = get_gt_pose(scene_dir, image_index)
    if gt_pos is None:
        return {"error": f"GT pose not found for image_index={image_index}"}

    pred_dir_n = _normalize(pred_dir)
    pos_err = float(np.linalg.norm(pred_pos - gt_pos))
    dot = float(np.clip(np.dot(pred_dir_n, gt_dir), -1.0, 1.0))
    ang_err = float(math.degrees(math.acos(dot)))

    iou = None
    if iou_context is not None:
        rc_scene, mesh_id, tri_points, tri_centroids, tri_areas = iou_context
        iou = compute_view_iou(gt_pos, gt_dir, pred_pos, pred_dir_n,
                               rc_scene, mesh_id, tri_points, tri_centroids, tri_areas)

    return {
        "pos_err": round(pos_err, 4),
        "ang_err": round(ang_err, 4),
        "iou": round(iou, 4) if iou is not None else None,
        "iou_error": round(1.0 - iou, 4) if iou is not None else None,
        "gt_position": gt_pos.tolist(),
        "gt_direction": gt_dir.tolist(),
    }
