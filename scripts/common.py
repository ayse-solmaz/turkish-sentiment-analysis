"""Eğitim ve değerlendirme script'lerinin ortak kullandığı yardımcılar."""

import pandas as pd

LABELS = ["Negative", "Notr", "Positive"]
SEED = 42

# Duygu içermeyen günlük cümleler: model bunları "Notr" bilmeli.
GUNLUK_NOTR = [
    "Kargo bugün geldi.",
    "Toplantı saat üçte başlıyor.",
    "Ürün mavi renkte.",
    "Bu ürünü dün aldım.",
    "Fiyatı 200 TL.",
    "Paket kapıya bırakıldı.",
    "Bugün hava çok bulutluydu.",
    "Yarın toplantı var.",
]


def bosluk_duzelt(metinler):
    """Kısayol 1: noktalamadan önceki boşluğu sil ("geçti ." -> "geçti.")."""
    return metinler.str.replace(r"\s+([.,!?;:])", r"\1", regex=True).str.strip()


def ilk_harf(metin, buyuk):
    """Kısayol 2: ilk harfi büyüt/küçült. Türkçe i/İ ve ı/I harflerini doğru çevirir."""
    if not metin:
        return metin
    c = metin[0]
    if buyuk:
        c = {"i": "İ", "ı": "I"}.get(c, c.upper())
    else:
        c = {"I": "ı", "İ": "i"}.get(c, c.lower())
    return c + metin[1:]


def test_seti(n=6_000):
    """Temizlenmiş test setinden, sınıf oranlarını koruyarak sabit bir alt küme."""
    test = pd.read_csv("data/raw/test.csv")
    test["text"] = bosluk_duzelt(test.text)
    return test.groupby("label").sample(frac=n / len(test), random_state=SEED)
