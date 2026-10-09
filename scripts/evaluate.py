"""Modelleri aynı testlerle yan yana karşılaştırır.

Çalıştırma (proje klasöründen):
  python scripts/evaluate.py                                   # v1 ve v2
  python scripts/evaluate.py models/electra-sentiment-v2       # tek model
"""

import sys

import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import GUNLUK_NOTR, ilk_harf, test_seti

MODELLER = sys.argv[1:] or ["models/electra-sentiment", "models/electra-sentiment-v2"]
BATCH = 64

test = test_seti()
# Aynı test seti üç halde: olduğu gibi, tüm ilk harfler küçük, tüm ilk harfler büyük.
varyantlar = {
    "olduğu gibi": test.text.tolist(),
    "ilk harf küçük": [ilk_harf(m, buyuk=False) for m in test.text],
    "ilk harf büyük": [ilk_harf(m, buyuk=True) for m in test.text],
}


@torch.no_grad()
def tahmin_et(model, tokenizer, metinler):
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    sonuc = []
    for i in range(0, len(metinler), BATCH):
        enc = tokenizer(metinler[i : i + BATCH], truncation=True, max_length=64, padding=True, return_tensors="pt")
        sonuc += [labels[p] for p in model(**enc).logits.argmax(-1).tolist()]
    return sonuc


satirlar = []
for yol in MODELLER:
    print(f"değerlendiriliyor: {yol} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(yol)
    model = AutoModelForSequenceClassification.from_pretrained(yol).eval()
    satir = {"model": yol.split("/")[-1]}

    for ad, metinler in varyantlar.items():
        tahmin = tahmin_et(model, tokenizer, metinler)
        satir[f"F1 ({ad})"] = f1_score(test.label, tahmin, average="macro")
        if ad == "olduğu gibi":
            satir["Negative precision"] = precision_score(test.label, tahmin, labels=["Negative"], average="macro")

    # Günlük nötr cümleler, hem küçük hem büyük harfle başlayarak
    for buyuk in [False, True]:
        metinler = [ilk_harf(m, buyuk) for m in GUNLUK_NOTR]
        dogru = sum(t == "Notr" for t in tahmin_et(model, tokenizer, metinler))
        satir[f"günlük nötr ({'büyük' if buyuk else 'küçük'} harf)"] = f"{dogru}/{len(GUNLUK_NOTR)}"
    satirlar.append(satir)

pd.set_option("display.width", 200)
print()
print(pd.DataFrame(satirlar).set_index("model").T.to_string(float_format=lambda x: f"{x:.3f}"))
