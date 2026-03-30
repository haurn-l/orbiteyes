import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # 🔥 YENİ: İlerleme çubuğu için eklendi
from src.config import config
from src.models.architecture import TulparModel
from src.training.tuner import Snake, TulparAsymmetricLoss

class TulparSüperOptuna:
    def __init__(self, champion_class, champion_name, train_loader, valid_loader):
        self.champion_class = champion_class
        self.champion_name = champion_name
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def objective(self, trial):
        # 1. EĞİTİM VE OPTİMİZASYON HİPERPARAMETRELERİ
        lr = trial.suggest_float("lr", 1e-6, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "RMSprop", "Adam"])
        grad_clip = trial.suggest_float("grad_clip", 0.5, 5.0)

        # 2. KAYIP FONKSİYONU (LOSS) HİPERPARAMETRELERİ
        penalty = trial.suggest_float("penalty", 5.0, 50.0)
        gamma_neg = trial.suggest_int("gamma_neg", 1, 5)

        # 3. MAMBA VE FİZİK MOTORU BOYUTLARI
        mamba_params = {
            "physics_dim": trial.suggest_categorical("phys_d", [16, 32, 64, 128]),
            "mamba_d_state": trial.suggest_categorical("m_state", [16, 32, 64, 128]),
            "mamba_d_model": trial.suggest_categorical("m_model", [32, 64, 128, 256]),
            "fusion_dim": trial.suggest_categorical("f_dim", [64, 128, 256, 512]),
            "dropout_rate": trial.suggest_float("drop", 0.05, 0.5)
        }

        # 4. AKTİVASYON FONKSİYONLARI
        act_name = trial.suggest_categorical("act", ["Snake", "Mish", "GELU", "SiLU", "LeakyReLU"])
        if act_name == "Snake": mamba_params["activation"] = Snake()
        elif act_name == "Mish": mamba_params["activation"] = nn.Mish()
        elif act_name == "GELU": mamba_params["activation"] = nn.GELU()
        elif act_name == "SiLU": mamba_params["activation"] = nn.SiLU()
        else: mamba_params["activation"] = nn.LeakyReLU()

        c_params = {"c_in": 38, "seq_len": 269}

        base_model = self.champion_class(**c_params).to(config.DEVICE)
        model = TulparModel(champion_model=base_model, **mamba_params).to(config.DEVICE)
        
        if optimizer_name == "AdamW": opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "RMSprop": opt = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        else: opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        crit = TulparAsymmetricLoss(gamma_neg=gamma_neg, penalty_factor=penalty)

        for epoch in range(3):
            model.train()
            for i, (X, y) in enumerate(self.train_loader):
                if i > 25: break 
                opt.zero_grad()
                loss = crit(model(X.to(config.DEVICE)), y.to(config.DEVICE))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
        
        model.eval()
        v_loss = 0
        batches = 0
        with torch.no_grad():
            for i, (X, y) in enumerate(self.valid_loader):
                if i > 15: break
                v_loss += crit(model(X.to(config.DEVICE)), y.to(config.DEVICE)).item()
                batches += 1
                
        return v_loss / max(1, batches)

    def optimize(self, n_trials=50):
        # Optuna'nın kafa karıştırıcı yazılarını kapatıyoruz
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        
        # 🔥 YENİ: Optuna denemeleri için Özel İlerleme Çubuğu (Callback)
        with tqdm(total=n_trials, desc=f"🧬 Optuna Aranıyor ({self.champion_name})", colour='magenta') as pbar:
            def tqdm_callback(study, trial):
                pbar.update(1) # Barı ilerlet
                pbar.set_postfix({'En İyi Kayıp': f"{study.best_value:.4f}"}) # Anlık en iyi skoru göster
                
            # Çalışmayı callback ile başlat
            study.optimize(self.objective, n_trials=n_trials, callbacks=[tqdm_callback])
            
        return study