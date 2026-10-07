from huggingface_hub import hf_hub_download

# Hugging Face'teki veri setinin adı
REPO = "winvoker/turkish-sentiment-analysis-dataset"

for split in ["train", "test"]:
    path = hf_hub_download(
        repo_id=REPO,
        filename=f"{split}.csv",
        repo_type="dataset",   # model değil, veri seti indiriyoruz
        local_dir="data/raw",  # nereye kaydedilecek
    )
    print("indirildi:", path)
