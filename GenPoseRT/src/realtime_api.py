"""
API temps réel pour Vtuber IA : génération de poses en streaming, sans fichiers intermédiaires.
"""
import torch
try:
    from src.model import TextEncoder, RealTimeMotionGenerator
except ImportError:
    # fallback for direct script run
    from model import TextEncoder, RealTimeMotionGenerator

class RealTimeVtuberEngine:
    def __init__(self, pose_dim=72, device='cpu'):
        self.device = torch.device(device)
        self.text_encoder = TextEncoder(emb_dim=256, use_sentence_transformers=False).to(self.device)
        self.model = RealTimeMotionGenerator(text_emb_dim=256, pose_dim=pose_dim).to(self.device)
        self.hidden = None
        self.prev_pose = torch.zeros(pose_dim, device=self.device)
        self.text_emb = None

    def set_context(self, text:str):
        """Met à jour le contexte (texte, émotion, etc.)"""
        self.text_emb = self.text_encoder([text]).to(self.device)
        self.hidden = None
        self.prev_pose = torch.zeros(self.model.pose_dim, device=self.device)

    def next_pose(self):
        """Génère la prochaine pose à partir du contexte courant."""
        if self.text_emb is None:
            raise RuntimeError("Contexte non initialisé. Appelez set_context(text) d'abord.")
        next_pose, self.hidden = self.model.step(self.text_emb, self.prev_pose, self.hidden)
        self.prev_pose = next_pose.squeeze(0)
        return self.prev_pose.detach().cpu().numpy()  # (pose_dim,)

# Exemple d'utilisation temps réel
if __name__ == "__main__":
    engine = RealTimeVtuberEngine()
    engine.set_context("Bonjour, je suis content !")
    for t in range(60):  # 2 secondes à 30 FPS
        pose = engine.next_pose()
        print(f"Frame {t}: {pose[:8]} ...")  # Affiche les 2 premiers os
    # Pour changer d'expression :
    engine.set_context("Je suis surpris !")
    for t in range(30):
        pose = engine.next_pose()
        print(f"Surprise {t}: {pose[:8]} ...")
