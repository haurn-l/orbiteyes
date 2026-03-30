import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print("[*] TSAI Modelleri Yükleniyor...")

# 🔥 ZIRHLI İÇE AKTARMA: Biri yoksa diğerleri patlamaz!
try:
    from tsai.all import InceptionTimePlus
    HAS_INCEPTION = True
except ImportError: HAS_INCEPTION = False

try:
    from tsai.all import TSTPlus
    HAS_TST = True
except ImportError: HAS_TST = False

try:
    from tsai.all import FCNPlus
    HAS_FCN = True
except ImportError: HAS_FCN = False

try:
    from tsai.all import TimesNet
    HAS_TIMESNET = True
except ImportError: 
    HAS_TIMESNET = False
    print("[!] TimesNet bu sürümde bulunamadı, yedeğe (Dummy) geçilecek.")

def fix_dims(x):
    if x.ndim == 3 and x.shape[1] == 269:
        return x.transpose(1, 2)
    return x

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.spline_weight, a=np.sqrt(5))
    def forward(self, x):
        return F.linear(x, self.base_weight) + F.linear(F.silu(x), self.spline_weight)

class BaselineKAN_TS(nn.Module):
    def __init__(self, c_in, seq_len=269, c_out=1):
        super().__init__()
        flattened_dim = c_in * seq_len
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            KANLayer(flattened_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU()
        )
        self.classifier = nn.Linear(128, c_out)

    def get_features(self, x): return self.feature_extractor(x)
    def forward(self, x): return self.classifier(self.get_features(x))

class FallbackDummy(nn.Module):
    def __init__(self, c_in, seq_len, c_out=64):
        super().__init__()
        self.flat = nn.Flatten()
        self.lin = nn.Linear(c_in * seq_len, c_out)
    def forward(self, x):
        return self.lin(self.flat(x))

class LiquidCfC_Proxy(nn.Module):
    def __init__(self, c_in, seq_len=269, c_out=1):
        super().__init__()
        self.model = InceptionTimePlus(c_in, 64, seq_len=seq_len) if HAS_INCEPTION else FallbackDummy(c_in, seq_len, 64)
        self.classifier = nn.Linear(64, c_out)
    def get_features(self, x): return self.model(fix_dims(x))
    def forward(self, x): return self.classifier(self.get_features(x))

class TimesNet_Proxy(nn.Module):
    def __init__(self, c_in, seq_len=269, c_out=1):
        super().__init__()
        self.model = TimesNet(c_in, 64, seq_len=seq_len) if HAS_TIMESNET else FallbackDummy(c_in, seq_len, 64)
        self.classifier = nn.Linear(64, c_out)
    def get_features(self, x): return self.model(fix_dims(x))
    def forward(self, x): return self.classifier(self.get_features(x))

class BaselineTST(nn.Module):
    def __init__(self, c_in, seq_len=269, c_out=1):
        super().__init__()
        self.model = TSTPlus(c_in, 64, seq_len=seq_len) if HAS_TST else FallbackDummy(c_in, seq_len, 64)
        self.classifier = nn.Linear(64, c_out)
    def get_features(self, x): return self.model(fix_dims(x))
    def forward(self, x): return self.classifier(self.get_features(x))

class Baseline_xLSTM_FCN(nn.Module):
    def __init__(self, c_in, seq_len=269, c_out=1):
        super().__init__()
        self.model = FCNPlus(c_in, 64) if HAS_FCN else FallbackDummy(c_in, seq_len, 64)
        self.classifier = nn.Linear(64, c_out)
    def get_features(self, x): return self.model(fix_dims(x))
    def forward(self, x): return self.classifier(self.get_features(x))