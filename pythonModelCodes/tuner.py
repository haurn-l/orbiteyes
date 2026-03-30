import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm  # 🔥 GÖRSEL İZLEME İÇİN EKLENDİ

from src.config import config
from src.models.baselines import (
    LiquidCfC_Proxy, 
    TimesNet_Proxy, 
    BaselineTST, 
    Baseline_xLSTM_FCN, 
    BaselineKAN_TS
)

# =========================================================================
# 1. NÜKLEER BİLEŞENLER
# =========================================================================
class Snake(nn.Module):
    def forward(self, x): 
        return x + torch.pow(torch.sin(x), 2)

class TulparAsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, penalty_factor=15.0):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.penalty_factor = penalty_factor
        self.eps = 1e-7

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        loss_pos = -1 * self.penalty_factor * targets * torch.log(probs + self.eps)
        loss_neg = -1 * (1 - targets) * torch.pow(probs, self.gamma_neg) * torch.log(1 - probs + self.eps)
        return torch.mean(loss_pos + loss_neg)

# =========================================================================
# 2. GLADYATÖR ARENASI (ANA SINIF)
# =========================================================================
class TulparGladiatorArena:
    def __init__(self, train_loader, valid_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def select_champion(self, test_epochs=2):
        print("\n===========================================================================")
        print(" ⚔️ FAZ 1: TULPAR GLADYATÖR ARENASI (5 MODEL KAPIŞIYOR) ⚔️")
        print("===========================================================================\n")

        models_dict = {
            "Liquid_Neural_Net": LiquidCfC_Proxy,
            "TimesNet": TimesNet_Proxy,
            "Transformer_TST": BaselineTST,
            "xLSTM_FCN": Baseline_xLSTM_FCN,
            "KAN_TST": BaselineKAN_TS
        }

        best_loss = float('inf')
        champion_name = None
        champion_class = None
        criterion = TulparAsymmetricLoss()

        X_sample, _ = next(iter(self.train_loader))
        dynamic_c_in = X_sample.shape[-1]
        
        # 🔥 GERÇEK VERİ BOYUTUNU EKRANA YAZDIRIYORUZ
        total_batches = len(self.train_loader)
        print(f"📊 Veri akışı doğrulandı: {dynamic_c_in} özellik tespit edildi.")
        print(f"📦 Toplam Eğitim Yığını (Batch): {total_batches} (Bu sayı tamamen dolmalı!)\n")

        for name, model_class in models_dict.items():
            print(f"[*] {name} arenaya çıkıyor...")
            
            try:
                model = model_class(c_in=dynamic_c_in, seq_len=269).to(self.device)
            except TypeError:
                model = model_class(c_in=dynamic_c_in).to(self.device)
            
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            # --- 🔥 GÖRSEL EĞİTİM TESTİ ---
            model.train()
            for epoch in range(test_epochs):
                # tqdm ile verilerin nasıl aktığını canlı göreceksin
                pbar = tqdm(self.train_loader, desc=f"  > Eğitim ({name})", leave=False, colour='cyan')
                for X_batch, y_batch in pbar:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    optimizer.zero_grad()
                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                    loss.backward()
                    optimizer.step()
                    # Anlık kaybı barda göster
                    pbar.set_postfix({'Anlık Kayıp': f"{loss.item():.4f}"})
            
            # --- 🔥 GÖRSEL PERFORMANS ÖLÇÜMÜ ---
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                v_pbar = tqdm(self.valid_loader, desc=f"  > Test ({name})", leave=False, colour='green')
                for X_batch, y_batch in v_pbar:
                    logits = model(X_batch.to(self.device))
                    val_loss += criterion(logits, y_batch.to(self.device)).item()
            
            avg_loss = val_loss / len(self.valid_loader)
            print(f"  🏆 {name} Asimetrik Kaybı: {avg_loss:.4f}\n")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                champion_name = name
                champion_class = model_class
                
        print(f"\n🥇 ARENA ŞAMPİYONU: {champion_name}")
        return champion_name, champion_class