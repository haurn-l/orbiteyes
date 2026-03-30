import torch
import torch.nn as nn
import torch.nn.functional as F

class TulparMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_output=64):
        super().__init__()
        self.d_state, self.d_model = d_state, d_model
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().repeat(d_model, 1)))
        self.dt_proj = nn.Linear(1, d_model)
        self.x_proj = nn.Linear(d_model, d_state * 2)
        self.out_proj = nn.Linear(d_model, d_output)

    def forward(self, x):
        A = -torch.exp(self.A_log).to(x.device)
        dt = F.softplus(self.dt_proj(x[:, :, 0:1]))
        B, C = torch.split(self.x_proj(x), [self.d_state, self.d_state], dim=-1)
        state = torch.zeros(x.shape[0], self.d_model, self.d_state).to(x.device)
        outputs = []
        for t in range(x.shape[1]):
            delta_A = torch.exp(A * dt[:, t, :].unsqueeze(-1))
            delta_B = (dt[:, t, :].unsqueeze(-1) * x[:, t, :].unsqueeze(-1)) * B[:, t, :].unsqueeze(1)
            state = delta_A * state + delta_B
            outputs.append((state * C[:, t, :].unsqueeze(1)).sum(dim=-1))
        return self.out_proj(torch.stack(outputs, dim=1))
    
class TulparPhysicsCore(nn.Module):
    def __init__(self, c_in=38, physics_out_dim=32):
        super().__init__()
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
        
        v_mag = torch.norm(v_vec, dim=-1) 
        r_mag = torch.norm(r_vec, dim=-1) 
        
        orbital_energy = (v_mag**2 / 2.0) - (1.0 / r_mag) 
        
        energy_diff = torch.cat([
            torch.zeros(orbital_energy.shape[0], 1, device=x.device), 
            orbital_energy[:, 1:] - orbital_energy[:, :-1]
        ], dim=1)
        
        physics_raw = torch.stack([v_mag, r_mag, orbital_energy, energy_diff, r_mag/v_mag], dim=-1)
        return self.physics_embedding(physics_raw) 

# 🔥 GÜNCELLENDİ: İsim Titan'dan TulparModel'e çevrildi!
class TulparModel(nn.Module):
    def __init__(self, champion_model, champion_feat_dim=64, c_in=38, seq_len=269, dropout_rate=0.3, 
                 physics_dim=32, mamba_d_state=32, mamba_d_model=64, fusion_dim=128, activation=nn.GELU()):
        super().__init__()
        self.physics_engine = TulparPhysicsCore(c_in, physics_dim)
        self.mamba_stream = nn.Sequential(nn.Linear(c_in, mamba_d_model), nn.LayerNorm(mamba_d_model), activation,
                                         TulparMambaBlock(mamba_d_model, mamba_d_state, mamba_d_model))
        self.champion_stream = champion_model
        
        self.align_champion = nn.Linear(champion_feat_dim, fusion_dim)
        self.align_mamba_phys = nn.Linear(physics_dim + mamba_d_model, fusion_dim)
        self.attention_fusion = nn.MultiheadAttention(fusion_dim, 4, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(fusion_dim, 64), activation, nn.Dropout(dropout_rate), nn.Linear(64, 1))

    def forward(self, x):
        kv_seq = self.align_mamba_phys(torch.cat([self.physics_engine(x.transpose(1, 2)), self.mamba_stream(x)], dim=-1))
        q_seq = self.align_champion(self.champion_stream.get_features(x)).unsqueeze(1)
        attn_out, _ = self.attention_fusion(q_seq, kv_seq, kv_seq)
        return self.classifier(attn_out.squeeze(1))