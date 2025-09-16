# Real-time Text→3D-Animation Generator

## Project Structure

- `src/`: models, loader, helpers
- `scripts/`: conversion, utilities, Blender
- `data/`: converted datasets (.npy)
- `notebooks/`: tests, analyses (optional)
- `requirements.txt`, `README.md`, `launch.json` : root

## Quick Usage

## Real-time Usage (Vtuber, interactive animation, NPC movement and/or attack for procedurally generated and adaptive in a VR games, etc.)

To generate poses in streaming mode (without files), use the real-time API:
```python
from src.realtime_api import RealTimeVtuberEngine
engine = RealTimeVtuberEngine()
engine.set_context("Hello, I’m happy!")
for t in range(60):
	pose = engine.next_pose()  # numpy array (pose_dim,)
	# -> Apply pose to your 3D rig here
```

- Change the context anytime with `set_context("new emotion or text")`.
- On each frame, call `next_pose()` to get the next pose.
- Integrate this stream into your 3D engine (Unity, Unreal, Godot...): split the vector into quaternions per bone and apply them to your skeleton.

### 3D Engine Integration (pseudo-code)
```python
For each frame:
pose = engine.next_pose()  # (pose_dim,)
for i, bone in enumerate(bone_names):
	quat = pose[i*4:(i+1)*4]  # (x, y, z, w)
	# Apply quat to bone in your engine
```


### Tips

- go to `.../GenPoseRT/script` and run `run_pipline.ps1` it's a intaler/launcher and it's automaticali instal the dependens
- The model can be enriched with structured context (text, emotion, intention...).
- For embedded inference: export to ONNX or TorchScript if needed.
- Adjust the next_pose() call frequency depending on your target FPS (30–60Hz recommended).
- Add a 3D model you plan to use in run_pipeline.ps1 at line 9 so the AI can adapt to the rig and
- learn from it (highly recommended).
- IMPORTANT: the 3D model must be in .fbx or .glb format and encoded in UTF-8.

1. Install dependencies:
```powershell
python -m pip install -r requirements.txt
```

2. Convert your FBX animations into .npy with Blender:
```powershell
python scripts/prepare_packed.py --input "C:/Users/mathb/Downloads/Packed" --output "C:/Users/mathb/OneDrive/Desktop/GenPoseRT/data/converted_packed"
```

3. Load the .npy files in your code:
```python
from src.data_loader import PackedDataset
train_ds = PackedDataset("data/converted_packed")
```

4. Train or test the model with src/model.py.

## Notes
- The Blender script must be launched inside Blender (see scripts/blender/convert_fbx_to_npy.py).
- Adapt the bone mapping if your rig differs too much.
- The expected .npy files are float32 matrices of shape (T, pose_dim).

