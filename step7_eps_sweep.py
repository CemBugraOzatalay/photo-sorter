from pathlib import Path
import yaml
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

db_dir = Path(cfg["db_dir"])
X = np.load(db_dir / "face_embeddings.npy")

X = normalize(X)

pca = PCA(n_components=128, random_state=42)
Xr = pca.fit_transform(X)

min_samples = 4
eps_list = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

print("N:", Xr.shape[0], "dim:", Xr.shape[1])
print("min_samples:", min_samples)
print("-" * 50)

for eps in eps_list:
    clu = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = clu.fit_predict(Xr)

    n_noise = int(np.sum(labels == -1))
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print(f"eps={eps:.2f} | clusters={n_clusters:3d} | noise={n_noise:4d} | clustered={Xr.shape[0]-n_noise:4d}")
