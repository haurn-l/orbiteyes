# src/physics_core.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TulparPhysicsCore(nn.Module):
    def __init__(self, c_in=38, physics_out_dim=32):
        super().__init__()
        # 38 özelliği, Zaman'ı (Seq_Len=269) bozmadan 3 boyutlu uzaya izdüşümler
        self.v_proj = nn.Conv1d(c_in, 3, kernel_size=1) 
        self.r_proj = nn.Conv1d(c_in, 3, kernel_size=1) 
        
        self.physics_embedding = nn.Sequential(
            nn.Linear(5, physics_out_dim), 
            nn.LayerNorm(physics_out_dim),
            nn.GELU()
        )

    def forward(self, x):
        v_conv = self.v_proj(x) 
        r_conv = self.r_proj(x) 
        
        v_vec = torch.clamp(v_conv.transpose(1, 2), min=-15.0, max=15.0)
        r_vec = torch.clamp(F.softplus(r_conv.transpose(1, 2)), min=0.1, max=3000.0)
        
        v_mag = torch.norm(v_vec, dim=-1) # (Batch, 269)
        r_mag = torch.norm(r_vec, dim=-1) # (Batch, 269)
        
        orbital_energy = (v_mag**2 / 2.0) - (1.0 / r_mag) 
        
        energy_diff = torch.cat([
            torch.zeros(orbital_energy.shape[0], 1, device=x.device), 
            orbital_energy[:, 1:] - orbital_energy[:, :-1]
        ], dim=1)
        
        physics_raw = torch.stack([v_mag, r_mag, orbital_energy, energy_diff, r_mag/v_mag], dim=-1)
        return self.physics_embedding(physics_raw)