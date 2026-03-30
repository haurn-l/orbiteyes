# src/training/trainer.py
import torch
import os
from src.config import config
from src.training.tuner import TulparAsymmetricLoss

class TulparTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = TulparAsymmetricLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE)
        self.device = config.DEVICE

    def train_with_checkpointing(self, epochs=10):
        print(f"🚀 Tulpar Eğitimi Başladı... ({self.device})")
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # --- EĞİTİM ---
            self.model.train()
            total_train_loss = 0
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(self.train_loader)
            
            # --- DOĞRULAMA (VALIDATION) EKLENDİ ---
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for X_val, y_val in self.val_loader:
                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    val_output = self.model(X_val)
                    v_loss = self.criterion(val_output, y_val)
                    total_val_loss += v_loss.item()
                    
            avg_val_loss = total_val_loss / len(self.val_loader)
            
            print(f"Epoch [{epoch+1}/{epochs}] - Train Kayıp: {avg_train_loss:.4f} | Val Kayıp: {avg_val_loss:.4f}")
            
            # GÜNCELLENDİ: Validation kaybına göre kaydediliyor, İsim Tulpar oldu.
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), os.path.join(config.LOGS_DIR, "tulpar_best.pth"))
                print("💾 Validation düştü! Tulpar'ın en iyi hali kaydedildi!")