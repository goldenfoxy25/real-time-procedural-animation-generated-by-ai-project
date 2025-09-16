import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class PackedDataset(Dataset):
    """
    Dataset wrapper for the provided Packed FBX archive.

    Expected workflow:
    - Convert FBX files to per-frame numpy arrays (.npy) using the provided Blender script
      (see scripts/blender/convert_fbx_to_npy.py).
    - Point this dataset at the folder containing the converted .npy files.

    Each .npy is expected to contain an array of shape (T, pose_dim) of dtype float32.
    The dataset will optionally create simple text prompts using the FBX filename or a
    mapping file if available.
    """
    def __init__(self, converted_root: str, pose_dim: int = 72, ext='.npy', text_map=None):
        self.converted_root = converted_root
        self.pose_dim = pose_dim
        self.ext = ext
        self.text_map = text_map or {}
        pattern = os.path.join(self.converted_root, '**', f'*{self.ext}')
        self.files = sorted(glob.glob(pattern, recursive=True))
        # look for companion metadata files (same name but .json)
        self.meta_map = {}
        for f in self.files:
            meta = os.path.splitext(f)[0] + '.json'
            if os.path.exists(meta):
                try:
                    import json
                    with open(meta, 'r') as mf:
                        self.meta_map[f] = json.load(mf)
                except Exception:
                    pass

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        arr = np.load(path)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.pose_dim:
            raise ValueError(f"Unexpected shape for {path}: {arr.shape}, expected (T,{self.pose_dim})")
        poses = torch.from_numpy(arr)
        rel = os.path.relpath(path, self.converted_root)
        text = self.text_map.get(rel, os.path.splitext(os.path.basename(path))[0].replace('_', ' '))
        item = {'text': text, 'poses': poses}
        # include metadata if available (e.g., bone ordering) so callers can remap
        if path in self.meta_map:
            item['meta'] = self.meta_map[path]
        return item

    def remap_to_target(self, poses: torch.Tensor, source_bones: list, target_bones: list):
        """
        Remap a poses tensor from source bone order to target bone order.
        poses: (T, D) where D = len(source_bones)*4 (quaternions)
        Returns remapped poses with shape (T, len(target_bones)*4).
        If a target bone is not found in source, zeros are inserted.
        """
        s_len = len(source_bones)
        t_len = len(target_bones)
        assert poses.ndim == 2
        T = poses.shape[0]
        out = np.zeros((T, t_len * 4), dtype=np.float32)
        src_idx = {b: i for i, b in enumerate(source_bones)}
        for i, tb in enumerate(target_bones):
            if tb in src_idx:
                si = src_idx[tb]
                out[:, i*4:(i+1)*4] = poses[:, si*4:(si+1)*4]
            else:
                # leave zeros -> identity quaternion could be (0,0,0,1) if desired
                out[:, i*4+3] = 1.0
        return torch.from_numpy(out)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('converted_root', nargs='?', default='.')
    args = p.parse_args()
    ds = PackedDataset(args.converted_root)
    print(f'Found {len(ds)} files under', args.converted_root)
    if len(ds) > 0:
        print('First file shape:', np.load(ds.files[0]).shape)
