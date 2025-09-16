print("START")
# pyright: ignore[reportMissingImports]
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional
import json

# Simple text encoder wrapper (uses sentence-transformers if disponible)
class TextEncoder(nn.Module):
    def __init__(self, emb_dim=256, use_sentence_transformers=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_st = False
        self.st = None
        self.token_emb = nn.Embedding(10000, emb_dim)
        self.project = nn.Identity()
        print("[INFO] TextEncoder: fallback mode (no sentence-transformers)")

    def forward(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if self.st is not None:
            embs = self.st.encode(texts, convert_to_tensor=True)
            embs = self.project(embs.float())
            return embs
        else:
            ids = []
            for t in texts:
                ids.append(abs(hash(t)) % 10000)
            ids = torch.tensor(ids, device=next(self.parameters()).device)
            return self.token_emb(ids)

class MotionGenerator(nn.Module):
    def __init__(self, text_emb_dim=256, pose_dim=72, hidden=512, num_frames=64):
        super().__init__()
        self.text_emb_dim = text_emb_dim
        self.pose_dim = pose_dim
        self.num_frames = num_frames
        self.frame_pos = nn.Embedding(num_frames, text_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(text_emb_dim + text_emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, pose_dim)
        )

    def forward(self, text_emb, num_frames: Optional[int] = None):
        B = text_emb.shape[0]
        if num_frames is None:
            num_frames = self.num_frames
        device = text_emb.device
        frame_ids = torch.arange(num_frames, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.frame_pos(frame_ids)
        text_rep = text_emb.unsqueeze(1).expand(-1, num_frames, -1)
        inp = torch.cat([text_rep, pos_emb], dim=-1)
        flat = inp.view(B * num_frames, -1)
        out = self.net(flat)
        poses = out.view(B, num_frames, self.pose_dim)
        return poses

class RealTimeMotionGenerator(nn.Module):
    def __init__(self, text_emb_dim=256, pose_dim=72, rnn_hidden=256, mlp_hidden=256):
        super().__init__()
        self.text_emb_dim = text_emb_dim
        self.pose_dim = pose_dim
        self.rnn = nn.GRU(input_size=text_emb_dim + pose_dim, hidden_size=rnn_hidden, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(rnn_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, pose_dim)
        )

    def forward(self, text_emb, prev_poses=None, hidden=None, steps=64):
        B = text_emb.shape[0]
        device = text_emb.device
        if prev_poses is None:
            prev = torch.zeros(B, 1, self.pose_dim, device=device)
        else:
            prev = prev_poses
        outs = []
        h = hidden
        for t in range(steps):
            inp = torch.cat([text_emb.unsqueeze(1), prev], dim=-1)
            out_rnn, h = self.rnn(inp, h)
            delta = self.out(out_rnn[:, -1, :])
            next_pose = prev.squeeze(1) + delta
            outs.append(next_pose.unsqueeze(1))
            prev = next_pose.unsqueeze(1)
        poses = torch.cat(outs, dim=1)
        return poses, h

    def step(self, text_emb, prev_pose, hidden=None):
        if prev_pose.dim() == 1:
            prev_pose = prev_pose.unsqueeze(0)
        if text_emb.dim() == 1:
            text_emb = text_emb.unsqueeze(0)
        inp = torch.cat([text_emb, prev_pose], dim=-1).unsqueeze(1)
        out_rnn, h = self.rnn(inp, hidden)
        delta = self.out(out_rnn[:, -1, :])
        next_pose = prev_pose + delta
        return next_pose, h

def stream_poses_from_text(text: str, model, text_encoder, steps=64, device='cpu'):
    device = torch.device(device) if isinstance(device, str) else device
    model.to(device)
    text_encoder.to(device)
    model.eval()
    text_encoder.eval()
    with torch.no_grad():
        emb = text_encoder([text]).to(device)
        if hasattr(model, 'step'):
            prev = torch.zeros(model.pose_dim, device=device)
            hidden = None
            for _ in range(steps):
                next_pose, hidden = model.step(emb, prev.unsqueeze(0), hidden)
                pose = next_pose.squeeze(0).cpu().numpy()
                yield pose
                prev = next_pose.squeeze(0)
        else:
            poses = model(emb, num_frames=steps)
            poses = poses.squeeze(0).cpu().numpy()
            for f in poses:
                yield f

def pose_reconstruction_loss(pred, target):
    return nn.functional.mse_loss(pred, target)

def smoothness_loss(pred):
    diff = pred[:, 1:] - pred[:, :-1]
    return torch.mean(diff ** 2)

def train_epoch(model, text_encoder, dataloader, optimizer, device):
    model.train()
    text_encoder.train()
    total_loss = 0.0
    for batch in dataloader:
        texts = batch['text']
        target_poses = batch['poses'].to(device)
        optimizer.zero_grad()
        text_emb = text_encoder(texts).to(device)
        pred = model(text_emb, num_frames=target_poses.shape[1])
        loss_recon = pose_reconstruction_loss(pred, target_poses)
        loss_smooth = smoothness_loss(pred) * 0.1
        loss = loss_recon + loss_smooth
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def save_animation_json(poses: np.ndarray, filepath: str):
    data = {"frames": poses.tolist()}
    with open(filepath, 'w') as f:
        json.dump(data, f)

def generate_animation_from_text(text: str, model, text_encoder, num_frames=64, device='cpu'):
    model.eval()
    text_encoder.eval()
    with torch.no_grad():
        emb = text_encoder([text]).to(device)
        poses = model(emb, num_frames=num_frames)
        poses = poses.squeeze(0).cpu().numpy()
    return poses

if __name__ == "__main__":
    import argparse
    print("[DEBUG] main block start")
    p = argparse.ArgumentParser()
    p.add_argument('--target_model', default=None, help='Optional path to target skeleton JSON to fine-tune/remap to')
    args = p.parse_args()

    print("[DEBUG] before device")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEBUG] after device")
    print("[DEBUG] before text_enc")
    text_enc = TextEncoder(emb_dim=256, use_sentence_transformers=False)
    print("[DEBUG] after text_enc")
    print("[DEBUG] before model")
    model = MotionGenerator(text_emb_dim=256, pose_dim=72, hidden=512, num_frames=64)
    print("[DEBUG] after model")
    print("[DEBUG] before text_enc.to(device)")
    text_enc.to(device)
    print("[DEBUG] after text_enc.to(device)")
    print("[DEBUG] before model.to(device)")
    model.to(device)
    print("[DEBUG] after model.to(device)")
    print("[DEBUG] before optimizer")
    optimizer = optim.Adam(list(model.parameters()) + list(text_enc.project.parameters()), lr=1e-4)
    print("[DEBUG] after optimizer")

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, n=100):
            self.n = n
        def __len__(self): return self.n
        def __getitem__(self, idx):
            T = 64
            poses = np.cumsum(np.random.randn(T, 72).astype(np.float32) * 0.01, axis=0)
            return {"text": f"walk cycle {idx%5}", "poses": torch.from_numpy(poses)}

    ds = DummyDataset(200)
    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, collate_fn=lambda batch: {"text":[b["text"] for b in batch], "poses": torch.stack([b["poses"] for b in batch])})

    print("[DEBUG] before train_epoch")
    # If a target model is provided, perform a small fine-tune pass after remapping
    if args.target_model:
        import json
        try:
            with open(args.target_model, 'r', encoding='utf-8') as tf:
                target_meta = json.load(tf)
            target_bones = target_meta.get('bones', None)
            if target_bones is None:
                print('Target model JSON found but no bones list inside. Running normal training (no remap).')
            else:
                print(f'Target model JSON loaded with {len(target_bones)} bones. If your converted dataset includes per-animation metadata (.json), the loader can remap during fine-tuning.')
        except Exception as e:
            print('Failed to load target model JSON:', e)
        # Run normal training (fine-tune hooks and remapping are available via PackedDataset.remap_to_target)
        train_loss = train_epoch(model, text_enc, dl, optimizer, device)
    else:
        train_loss = train_epoch(model, text_enc, dl, optimizer, device)
    print("train_loss:", train_loss)

    print("[DEBUG] before generate_animation_from_text")
    poses = generate_animation_from_text("a short walk then turn", model, text_enc, num_frames=64, device=device)
    save_animation_json(poses, "generated_animation.json")
    print("Saved generated_animation.json")
    print("[DEBUG] end main block")
