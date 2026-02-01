from pathlib import Path
import json
import yaml
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# -----------------------------
# 1) Yollar
# -----------------------------
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

db_dir = Path(cfg["db_dir"])
emb_path = db_dir / "face_embeddings.npy"
meta_path = db_dir / "face_meta_fixed.jsonl"
out_path = db_dir / "face_clusters.jsonl"

# -----------------------------
# 2) Embedding'leri yükle
# -----------------------------
X = np.load(emb_path)  # (N,512)
print("Embeddings:", X.shape)

# Güvenlik: L2 normalize (zaten normalize ama tekrar stabil)
X = normalize(X)

# -----------------------------
# 3) PCA (opsiyonel ama önerilir)
# -----------------------------
# 512 -> 128 (hız + daha az gürültü)
pca = PCA(n_components=128, random_state=42)
Xr = pca.fit_transform(X)
print("PCA çıktı:", Xr.shape)

# -----------------------------
# 4) DBSCAN ile cluster
# -----------------------------
# cosine yerine euclidean kullanıyoruz çünkü normalize+PCA sonrası iyi çalışır
# eps: 0.55-0.75 arası denenir. İlk deneme 0.65
eps = 0.80
min_samples = 4

clu = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
labels = clu.fit_predict(Xr)

n_noise = int(np.sum(labels == -1))
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print("Cluster sayısı:", n_clusters)
print("Noise (tek kalan):", n_noise)

# -----------------------------
# 5) Meta ile labels eşleme
# meta dosyasında sadece has_face=True olanların emb_index'i var.
# emb_index ile embedding satırını eşleştiriyoruz.
# -----------------------------
# labels_by_emb_index: {0: label, 1: label, ...}
labels_by_idx = {i: int(lab) for i, lab in enumerate(labels)}

with open(meta_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
    for line in f_in:
        obj = json.loads(line)
        if obj.get("has_face") and obj.get("emb_index") is not None:
            idx = obj["emb_index"]
            obj["cluster_id"] = labels_by_idx.get(idx, -1)
        else:
            obj["cluster_id"] = None
        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("Yazıldı:", out_path)
print("Kullanılan params -> eps:", eps, "min_samples:", min_samples)
