"""Eğitilmiş modeli ONNX'e çevirir, int8'e sıkıştırır ve sonuçların aynı kaldığını kontrol eder.

Çalıştırma (proje klasöründen):  python scripts/export_onnx.py [model klasörü]
Varsayılan klasör: models/electra-sentiment-v2
Çıktı: <model klasörü>/onnx/model.onnx           (float32)
       <model klasörü>/onnx/model_quantized.onnx (int8)
"""

import os
import sys
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "models/electra-sentiment-v2"
ONNX_DIR = f"{MODEL_DIR}/onnx"
FP32 = f"{ONNX_DIR}/model.onnx"
INT8 = f"{ONNX_DIR}/model_quantized.onnx"
os.makedirs(ONNX_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

# ---------------- 1. PyTorch -> ONNX ----------------
# ONNX, modeli "hesap grafiği" olarak kaydeder: hangi işlem hangi sırayla yapılıyor.
# Bu grafiği PyTorch olmadan da (ör. tarayıcıda) çalıştırabiliriz.
ornek = tokenizer(["örnek bir cümle"], return_tensors="pt")
torch.onnx.export(
    model,
    (ornek["input_ids"], ornek["attention_mask"], ornek["token_type_ids"]),
    FP32,
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    # Cümle sayısı (batch) ve uzunluğu (sequence) sabit değil, değişebilir:
    dynamic_axes={name: {0: "batch", 1: "sequence"} for name in ["input_ids", "attention_mask", "token_type_ids"]}
    | {"logits": {0: "batch"}},
    opset_version=17,
    dynamo=False,
)

# ---------------- 2. float32 -> int8 ----------------
# Ağırlıkları 32 bitlik ondalık sayılar yerine 8 bitlik tam sayılarla saklar: ~4 kat küçük.
quantize_dynamic(FP32, INT8, weight_type=QuantType.QInt8)

for yol in [FP32, INT8]:
    print(f"{yol}: {os.path.getsize(yol) / 1e6:.1f} MB")

# ---------------- 3. Doğrulama: aynı cümleye aynı cevabı veriyorlar mı? ----------------
test = pd.read_csv("data/raw/test.csv").sample(500, random_state=0)
test["text"] = test.text.str.replace(r"\s+([.,!?;:])", r"\1", regex=True).str.strip()
enc = tokenizer(test.text.tolist(), truncation=True, max_length=64, padding=True, return_tensors="np")
girdi = {k: enc[k].astype(np.int64) for k in ["input_ids", "attention_mask", "token_type_ids"]}

with torch.no_grad():
    t = time.time()
    pt_logits = model(**{k: torch.from_numpy(v) for k, v in girdi.items()}).logits.numpy()
    sureler = {"PyTorch": time.time() - t}

pt_tahmin = pt_logits.argmax(-1)
print(f"\n{'model':<10}{'en büyük fark':>15}{'aynı tahmin':>14}{'doğruluk':>11}{'süre (500 cümle)':>19}")
for ad, yol in [("fp32", FP32), ("int8", INT8)]:
    oturum = ort.InferenceSession(yol, providers=["CPUExecutionProvider"])
    t = time.time()
    logits = oturum.run(["logits"], girdi)[0]
    sureler[ad] = time.time() - t
    tahmin = logits.argmax(-1)
    fark = np.abs(logits - pt_logits).max()
    ayni = (tahmin == pt_tahmin).mean()
    dogru = (np.array(labels)[tahmin] == test.label.values).mean()
    print(f"{ad:<10}{fark:>15.5f}{ayni:>14.1%}{dogru:>11.1%}{sureler[ad]:>18.2f}s")

dogru_pt = (np.array(labels)[pt_tahmin] == test.label.values).mean()
print(f"{'PyTorch':<10}{'-':>15}{'-':>14}{dogru_pt:>11.1%}{sureler['PyTorch']:>18.2f}s")
