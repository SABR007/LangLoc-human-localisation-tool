import json
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, send_file, render_template, request

from metrics import build_iou_context, evaluate

app = Flask(__name__)

SCENES_DIR = Path(__file__).parent.parent / 'scenes_for_human_annotation'
ANNOTATIONS_FILE = Path(__file__).parent / 'annotations' / 'annotations.json'
EYE_HEIGHT = 1.6

# Per-scene IoU context cache (mesh + raycasting scene, expensive to build once)
_iou_cache: dict = {}


def _get_iou_context(scene_id: str):
    if scene_id not in _iou_cache:
        try:
            _iou_cache[scene_id] = build_iou_context(SCENES_DIR / scene_id)
        except Exception as e:
            app.logger.warning(f"IoU context build failed for {scene_id}: {e}")
            _iou_cache[scene_id] = None
    return _iou_cache[scene_id]


def build_pose_matrix(x, y, dx, dy, dz):
    position = np.array([x, y, EYE_HEIGHT], dtype=float)
    forward = np.array([dx, dy, dz], dtype=float)
    forward /= np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])

    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)

    m = [[0.0] * 4 for _ in range(4)]
    for i in range(3):
        m[i][0] = float(right[i])
        m[i][1] = float(up[i])
        m[i][2] = float(forward[i])
    m[0][3] = float(position[0])
    m[1][3] = float(position[1])
    m[2][3] = float(position[2])
    m[3][3] = 1.0
    return m, position, forward


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/scenes')
def get_scenes():
    scenes = sorted(d.name for d in SCENES_DIR.iterdir() if d.is_dir())
    return jsonify(scenes)


@app.route('/api/scene/<scene_id>/ply')
def get_ply(scene_id):
    ply_path = SCENES_DIR / scene_id / f'{scene_id}_vh_clean_2.ply'
    if not ply_path.exists():
        return jsonify({'error': 'PLY not found'}), 404
    return send_file(str(ply_path), mimetype='application/octet-stream')


@app.route('/api/scene/<scene_id>/descriptions')
def get_descriptions(scene_id):
    desc_path = SCENES_DIR / scene_id / 'output' / 'descriptions' / 'all_descriptions.json'
    if not desc_path.exists():
        return jsonify([])
    with open(desc_path) as f:
        data = json.load(f)
    simplified = [
        {
            'scene_index': d.get('scene_index', scene_id),
            'image_index': d.get('image_index', str(i)),
            'description': d.get('description', ''),
            'gt_scene_pose': d.get('scene_pose'),
        }
        for i, d in enumerate(data)
    ]
    return jsonify(simplified)


@app.route('/api/pose', methods=['POST'])
def compute_pose():
    body = request.json
    x, y = float(body['x']), float(body['y'])
    dx, dy, dz = float(body['dx']), float(body['dy']), float(body['dz'])
    matrix, position, direction = build_pose_matrix(x, y, dx, dy, dz)
    return jsonify({'scene_pose': matrix,
                    'position': position.tolist(),
                    'direction': direction.tolist()})


@app.route('/api/annotations', methods=['GET', 'POST'])
def annotations():
    if request.method == 'POST':
        body = request.json
        scene_id = body['scene_id']
        image_index = body['image_index']
        pred_pos = np.array(body['position'], dtype=float)
        pred_dir = np.array(body['direction'], dtype=float)

        ctx = _get_iou_context(scene_id)
        metrics = evaluate(SCENES_DIR / scene_id, image_index, pred_pos, pred_dir, ctx)

        annotation = {**body, 'metrics': metrics}

        existing = []
        if ANNOTATIONS_FILE.exists():
            with open(ANNOTATIONS_FILE) as f:
                existing = json.load(f)
        existing.append(annotation)
        with open(ANNOTATIONS_FILE, 'w') as f:
            json.dump(existing, f, indent=2)

        return jsonify({'status': 'saved', 'total': len(existing), 'metrics': metrics})

    else:
        if ANNOTATIONS_FILE.exists():
            with open(ANNOTATIONS_FILE) as f:
                return jsonify(json.load(f))
        return jsonify([])


if __name__ == '__main__':
    app.run(debug=True, port=5050)
