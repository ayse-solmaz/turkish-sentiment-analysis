"""Türkçe ELECTRA-small modelini duygu analizi için fine-tune eder.

Çalıştırma (proje klasöründen):  python scripts/train.py
Çıktı: models/electra-sentiment/  (model + tokenizer)
"""

import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------- Ayarlar ----------------
MODEL_NAME = "dbmdz/electra-small-turkish-cased-discriminator"
OUT_DIR = "models/electra-sentiment"
LABELS = ["Negative", "Notr", "Positive"]
PER_CLASS = 10_000  # her sınıftan kaç örnekle eğiteceğiz (dengeli veri)
VAL_SIZE = 2_000    # eğitim sırasında ilerlemeyi ölçmek için ayrılan örnek
TEST_SIZE = 6_000   # en sonda bir kez ölçülen test örneği
MAX_LEN = 64        # metinlerin %90'ı 64 token'dan kısa
BATCH = 32
EPOCHS = 2
LR = 1e-4
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
label2id = {l: i for i, l in enumerate(LABELS)}


def bosluk_duzelt(metinler):
    # Aşama 4'teki temizlik: "geçti ." -> "geçti."
    return metinler.str.replace(r"\s+([.,!?;:])", r"\1", regex=True).str.strip()


# ---------------- 1. Veri ----------------
train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")
train["text"] = bosluk_duzelt(train.text)
test["text"] = bosluk_duzelt(test.text)

# Tekrarları ve çelişkili etiketli metinleri at, test'te de görünen metinleri train'den çıkar (sızıntı).
celiskili = train.groupby("text").label.nunique()
celiskili = set(celiskili[celiskili > 1].index)
train = train[~train.text.isin(celiskili)].drop_duplicates("text")
train = train[~train.text.isin(set(test.text))]

# Her sınıftan eşit sayıda örnek al, sonra bir kısmını validation'a ayır.
train = train.groupby("label").sample(PER_CLASS, random_state=SEED).sample(frac=1, random_state=SEED)
val, train = train.iloc[:VAL_SIZE], train.iloc[VAL_SIZE:]
test = test.groupby("label").sample(frac=TEST_SIZE / len(test), random_state=SEED)
print(f"train={len(train)}  val={len(val)}  test={len(test)}")

# ---------------- 2. Model ve tokenizer ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=dict(enumerate(LABELS)),
    label2id=label2id,
)


def batches(df, shuffle):
    """Veriyi BATCH büyüklüğünde parçalara bölüp token'lara çevirir."""
    idx = np.random.permutation(len(df)) if shuffle else np.arange(len(df))
    for i in range(0, len(df), BATCH):
        part = df.iloc[idx[i : i + BATCH]]
        enc = tokenizer(
            part.text.tolist(), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt"
        )
        enc["labels"] = torch.tensor(part.label.map(label2id).values)
        yield enc


@torch.no_grad()
def predict(df):
    model.eval()
    preds = []
    for b in batches(df, shuffle=False):
        b.pop("labels")
        preds += model(**b).logits.argmax(-1).tolist()
    return [LABELS[p] for p in preds]


# ---------------- 3. Eğitim döngüsü ----------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
steps_per_epoch = (len(train) + BATCH - 1) // BATCH
total_steps = steps_per_epoch * EPOCHS
# Öğrenme hızını eğitim boyunca doğrusal olarak sıfıra indir.
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1 - s / total_steps)

for epoch in range(EPOCHS):
    model.train()
    start, loss_sum = time.time(), 0.0
    for step, b in enumerate(batches(train, shuffle=True), 1):
        loss = model(**b).loss   # 1) tahmin et ve hatayı (loss) hesapla
        loss.backward()          # 2) her ağırlığın hataya katkısını (gradyan) hesapla
        optimizer.step()         # 3) ağırlıkları hatayı azaltacak yönde güncelle
        scheduler.step()
        optimizer.zero_grad()    # 4) gradyanları sıfırla, sonraki adıma temiz başla
        loss_sum += loss.item()
        if step % 50 == 0 or step == steps_per_epoch:
            gecen = time.time() - start
            kalan = gecen / step * (steps_per_epoch - step)
            print(f"epoch {epoch + 1}/{EPOCHS}  adım {step}/{steps_per_epoch}  "
                  f"loss {loss_sum / step:.4f}  kalan ~{kalan / 60:.1f} dk", flush=True)

    val_f1 = f1_score(val.label, predict(val), average="macro")
    print(f"==> epoch {epoch + 1} bitti: validation macro F1 = {val_f1:.4f}", flush=True)

# ---------------- 4. Test (sadece bir kez, en sonda) ----------------
print("\n===== TEST =====")
print(classification_report(test.label, predict(test), digits=3))

cumleler = pd.DataFrame({"text": [
    "Kargo bugün geldi.",
    "Toplantı saat üçte başlıyor.",
    "Ürün mavi renkte.",
    "Bu ürünü dün aldım.",
    "Fiyatı 200 TL.",
    "Paket kapıya bırakıldı.",
], "label": "Notr"})
for metin, tahmin in zip(cumleler.text, predict(cumleler)):
    print(f"{tahmin:<9} {metin}")

model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("\nmodel kaydedildi:", OUT_DIR)
