import os
import torch
from dataclasses import dataclass

@dataclass
class TulparConfig:
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_DIR: str = os.path.join(BASE_DIR, "data", "raw", "tle_data")
    UNIFIED_DATA_DIR: str = os.path.join(BASE_DIR, "data", "unified", "tulpar_dataset")
    LOGS_DIR: str = os.path.join(BASE_DIR, "logs")
    
    SEQ_LEN: int = 269         # GÜNCELLENDİ: Asimetrik Zaman (7 Gün)
    FEATURE_DIM: int = 38      # 38 Sütunluk Uzay Fiziği
    
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE: int = 128 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 10e9 else 16
    
    EPOCHS: int = 30
    LEARNING_RATE: float = 1e-4

config = TulparConfig()
os.makedirs(config.LOGS_DIR, exist_ok=True)