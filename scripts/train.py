"""Türkçe ELECTRA-small modelini duygu analizi için fine-tune eder (v2).

v2'deki değişiklikler:
  - İlk harf augmentation: model "büyük harfle başlıyorsa nötr" kısayolunu öğrenemesin.
  - Prior düzeltmesi: dengeli eğitimin yarattığı "olumsuzu fazla tahmin etme" eğilimi düzeltilir.

Çalıştırma (proje klasöründen):  python scripts/train.py
Çıktı: models/electra-sentiment-v2/  (model + tokenizer)
Değerlendirme:  python scripts/evaluate.py
"""

import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import LABELS, SEED, bosluk_duzelt, ilk_harf

# ---------------- Ayarlar ----------------
MODEL_NAME = "dbmdz/electra-small-turkish-cased-discriminator"
OUT_DIR = "models/electra-sentiment-v2"
PER_CLASS = 10_000  # her sınıftan kaç örnekle eğiteceğiz (dengeli veri)
VAL_SIZE = 2_000    # eğitim sırasında ilerlemeyi ölçmek için ayrılan örnek
MAX_LEN = 64        # metinlerin %90'ı 64 token'dan kısa
BATCH = 32
EPOCHS = 2
LR = 1e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
label2id = {l: i for i, l in enumerate(LABELS)}

# ---------------- 1. Veri ----------------
train = pd.read_csv("data/raw/train.csv")
train["text"] = bosluk_duzelt(train.text)
# Gerçek sınıf oranları (prior düzeltmesi için, dengelemeden ÖNCE ölçülür)
gercek_oran = train.label.value_counts(normalize=True)

# Tekrarları ve çelişkili etiketli metinleri at, test'te de görünen metinleri train'den çıkar (sızıntı).
celiskili = train.groupby("text").label.nunique()
celiskili = set(celiskili[celiskili > 1].index)
train = train[~train.text.isin(celiskili)].drop_duplicates("text")
test_metinleri = set(bosluk_duzelt(pd.read_csv("data/raw/test.csv").text))
train = train[~train.text.isin(test_metinleri)]

# Her sınıftan eşit sayıda örnek al, sonra bir kısmını validation'a ayır.
train = train.groupby("label").sample(PER_CLASS, random_state=SEED).sample(frac=1, random_state=SEED)
val, train = train.iloc[:VAL_SIZE], train.iloc[VAL_SIZE:]
print(f"train={len(train)}  val={len(val)}")

# ---------------- 2. Model ve tokenizer ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=dict(enumerate(LABELS)),
    label2id=label2id,
)


def batches(df, egitim):
    """Veriyi BATCH büyüklüğünde parçalara bölüp token'lara çevirir."""
    idx = np.random.permutation(len(df)) if egitim else np.arange(len(df))
    for i in range(0, len(df), BATCH):
        part = df.iloc[idx[i : i + BATCH]]
        metinler = part.text.tolist()
        if egitim:
            # v2: ilk harfi yazı-tura ile büyük ya da küçük yap. Böylece harf tipi
            # sınıf hakkında hiçbir bilgi taşımaz ve model ona güvenmeyi öğrenmez.
            metinler = [ilk_harf(m, buyuk=random.random() < 0.5) for m in metinler]
        enc = tokenizer(metinler, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(part.label.map(label2id).values)
        yield enc


@torch.no_grad()
def predict(df):
    model.eval()
    preds = []
    for b in batches(df, egitim=False):
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
    for step, b in enumerate(batches(train, egitim=True), 1):
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

# ---------------- 4. Prior düzeltmesi ----------------
# Model her sınıfı eşit sıklıkta görerek eğitildi, yani "her sınıf 1/3 ihtimalle gelir" sanıyor.
# Gerçekte olumsuzlar çok daha az. Her sınıfın skoruna log(gerçek oran / eğitimdeki oran) ekleyerek
# bu varsayımı düzeltiyoruz. Düzeltmeyi son katmanın bias'ına yazdığımız için ONNX'e de
# tarayıcıya da kendiliğinden geçer, ekstra kod gerekmez.
egitim_orani = 1 / len(LABELS)
duzeltme = torch.tensor([np.log(gercek_oran[l] / egitim_orani) for l in LABELS], dtype=torch.float32)
print("\nprior düzeltmesi:", {l: round(d, 3) for l, d in zip(LABELS, duzeltme.tolist())})
with torch.no_grad():
    model.classifier.out_proj.bias += duzeltme

model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("model kaydedildi:", OUT_DIR)
print("Şimdi karşılaştırma için:  python scripts/evaluate.py")
