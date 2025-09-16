"""
Run inside Blender (text editor or blender --background --python script.py) to batch-convert FBX animations
into per-frame numpy arrays (.npy). This script assumes a simple humanoid skeleton where pose can be read
as per-bone local rotations. You will need to adapt bone ordering and rotation representation to match
your runtime rig.

Usage (example):
blender --background --python scripts/blender/convert_fbx_to_npy.py -- \
    --input_dir "/path/to/Packed/Animations" --output_dir "/path/to/converted_npy"

Notes:
- This script is a starting point. Different FBX exports have different armature names and bone conventions.
- The script exports local bone quaternion rotations concatenated into a single vector per frame.
"""
import os
import sys
import argparse

def export_fbx_to_npy(fbx_path, out_path, bone_names=None):
    try:
        import bpy # pyright: ignore[reportMissingImports]
        import numpy as np
    except ImportError:
        print('This function must be run inside Blender where bpy is available')
        raise
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    armature = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE':
            armature = obj
            break
    if armature is None:
        print('No armature found in', fbx_path)
        return False
    bpy.context.view_layer.objects.active = armature
    bones = bone_names or [b.name for b in armature.data.bones]
    scene = bpy.context.scene
    frame_start = int(scene.frame_start)
    frame_end = int(scene.frame_end)
    frames = []
    for f in range(frame_start, frame_end + 1):
        scene.frame_set(f)
        row = []
        for bn in bones:
            pb = armature.pose.bones.get(bn)
            if pb is None:
                row.extend([0,0,0,1])
            else:
                q = pb.rotation_quaternion
                row.extend([q.x, q.y, q.z, q.w])
        frames.append(row)
    arr = np.array(frames, dtype=np.float32)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, arr)
    # Also write companion metadata (bone names and frame rate)
    meta = {
        'shape': list(arr.shape),
        'bones': bones,
        'frame_start': frame_start,
        'frame_end': frame_end,
        'frame_rate': scene.render.fps
    }
    meta_path = os.path.splitext(out_path)[0] + '.json'
    try:
        import json
        with open(meta_path, 'w') as mf:
            json.dump(meta, mf)
    except Exception as e:
        print('Warning: could not write metadata JSON', e)
    print('Exported', out_path, 'shape', arr.shape, 'meta->', meta_path)
    return True

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args(argv)
    in_dir = args.input_dir
    out_dir = args.output_dir
    for root, dirs, files in os.walk(in_dir):
        for f in files:
            if f.lower().endswith('.fbx'):
                in_path = os.path.join(root, f)
                rel = os.path.relpath(in_path, in_dir)
                out_path = os.path.join(out_dir, os.path.splitext(rel)[0] + '.npy')
                print('Converting', in_path, '->', out_path)
                try:
                    export_fbx_to_npy(in_path, out_path)
                except Exception as e:
                    print('Failed for', in_path, e)

if __name__ == '__main__':
    main(sys.argv[1:])
