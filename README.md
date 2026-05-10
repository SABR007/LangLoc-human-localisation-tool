# LangLoc Human Localisation Tool

A browser-based annotation platform for collecting **human pose predictions** given a natural language description of a 3D scene.

The tool renders a coloured 3D point cloud of a ScanNet scene directly in the browser. An annotator reads a language query, then clicks to place their estimated position and gaze direction — the platform computes and saves the full 4 × 4 pose matrix in the same format used by the LangLoc evaluation pipeline.

---

## What it does

| Step | Action |
|------|--------|
| 1 | Select a scene from the dropdown |
| 2 | Pick a language query (auto-generated scene descriptions) |
| 3 | Click **Set Position** → click on the mesh to mark where you think you are (XY only — Z is fixed at **1.6 m** eye height) |
| 4 | Click **Set Direction** → click on whatever you would be looking at |
| 5 | Review the live **4 × 4 pose matrix** and position / direction vectors |
| 6 | Click **Save Annotation** — advances to the next query automatically |

---

## Output format

Each annotation is saved to `annotations/annotations.json` in the same schema as the LangLoc evaluation candidates:

```json
{
  "scene_id": "scene0006_00",
  "image_index": "001229",
  "description": "A refrigerator stands against the wall...",
  "position": [3.21, 4.05, 1.6],
  "direction": [0.707, 0.0, -0.707],
  "scene_pose": [
    [ 0.000,  0.707,  0.707, 3.21],
    [ 1.000,  0.000,  0.000, 4.05],
    [ 0.000,  0.707, -0.707, 1.60],
    [ 0.000,  0.000,  0.000, 1.00]
  ],
  "timestamp": "2026-05-11T12:34:56.789Z"
}
```

The `scene_pose` is a column-major 4 × 4 homogeneous transform: columns are `[right | up | forward | translation]`.

---

## Requirements

- Python 3.9+
- Flask 3.0+
- NumPy
- ScanNet scene folders (see below)
- A modern browser (Chrome / Firefox / Safari)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/sabr/LangLoc-human-localisation-tool.git
cd LangLoc-human-localisation-tool
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare scene data

The tool expects scene folders under a directory called `scenes/` at the project root. Each scene folder must follow the standard ScanNet layout:

```
scenes/
  scene0006_00/
    scene0006_00_vh_clean_2.ply        ← coloured mesh (required)
    output/
      descriptions/
        all_descriptions.json          ← language queries (required)
```

Point `SCENES_DIR` in `app.py` to your scenes folder if it lives elsewhere:

```python
# app.py  line 10
SCENES_DIR = Path('/path/to/your/scenes')
```

### 4. Run

```bash
python app.py
```

Then open **http://localhost:5050** in your browser.

---

## Controls

| Input | Action |
|-------|--------|
| Left-drag | Orbit / rotate the scene |
| Right-drag | Pan |
| Scroll | Zoom in / out |
| Click (Position mode) | Place yellow position marker at XY; z = 1.6 m |
| Click (Direction mode) | Set purple direction arrow toward clicked point |
| ↺ Reset | Clear current markers and start over |
| Save Annotation | Write pose to JSON and move to next query |

---

## File structure

```
LangLoc-human-localisation-tool/
├── app.py                  # Flask backend — scene serving, pose math, annotation saving
├── templates/
│   └── index.html          # Single-page UI (Three.js, no build step)
├── annotations/
│   └── annotations.json    # Created automatically on first save
├── requirements.txt
└── README.md
```

---

## Pose matrix construction

Given a clicked position `(x, y)` and a direction vector `d = (dx, dy, dz)`:

```
z_position = 1.6 m  (fixed eye height)

forward = normalise(d)
right   = normalise(world_up × forward)   where world_up = [0, 0, 1]
up      = forward × right

scene_pose = [ right_x  up_x  forward_x  x   ]
             [ right_y  up_y  forward_y  y   ]
             [ right_z  up_z  forward_z  1.6 ]
             [    0       0       0      1   ]
```

---

## Acknowledgements

Built as part of the **LangLoc** project — language-driven visual localisation in 3D scenes.  
Scene data: [ScanNet](http://www.scan-net.org/).
