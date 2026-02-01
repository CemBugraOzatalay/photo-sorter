from pathlib import Path
import yaml, json
import numpy as np
import cv2
import onnxruntime as ort
from tqdm import tqdm

# -----------------------------
# 0) Unicode-safe imread
# -----------------------------
def imread_unicode(path_str):
    data = np.fromfile(path_str, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

# -----------------------------
# 1) Config
# -----------------------------
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

db_dir = Path(cfg["db_dir"])
index_path = db_dir / "index.jsonl"
model_dir = Path("models")

# Output paths
meta_path = db_dir / "face_meta.jsonl"
emb_path = db_dir / "face_embeddings.npy"

# -----------------------------
# 2) Models
# -----------------------------
# Face detector (OpenCV DNN - Caffe)
proto = str(model_dir / "deploy.prototxt")
caffemodel = str(model_dir / "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNetFromCaffe(proto, caffemodel)

# ArcFace ONNX embedding session
arc_path = str(model_dir / "arcface.onnx")
sess = ort.InferenceSession(arc_path, providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name

def detect_largest_face_bgr(img_bgr, conf_th=0.6):
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        img_bgr, 1.0, (300, 300),
        (104.0, 177.0, 123.0),
        swapRB=False, crop=False
    )
    net.setInput(blob)
    det = net.forward()

    best = None
    best_area = 0
    for i in range(det.shape[2]):
        conf = float(det[0, 0, i, 2])
        if conf < conf_th:
            continue

        x1 = int(det[0, 0, i, 3] * w)
        y1 = int(det[0, 0, i, 4] * h)
        x2 = int(det[0, 0, i, 5] * w)
        y2 = int(det[0, 0, i, 6] * h)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2, conf)

    return best

def arcface_embed(face_bgr):
    # NHWC, RGB, normalize
    face = cv2.resize(face_bgr, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32)
    face = (face - 127.5) / 128.0
    face = face[None, ...]  # (1,112,112,3)

    emb = sess.run([out_name], {inp_name: face})[0][0]
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb.astype(np.float32)

# -----------------------------
# 3) Resume: daha önce yazılmış meta varsa kaç satır var?
# -----------------------------
done = 0
if meta_path.exists():
    with open(meta_path, "r", encoding="utf-8") as f:
        for _ in f:
            done += 1
print("Daha önce işlenen kayıt:", done)

# -----------------------------
# 4) Index'i oku (tüm kayıtlar)
# -----------------------------
records = []
with open(index_path, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

total = len(records)
print("Toplam index kaydı:", total)

# Eğer daha önce bir kısmı işlendi ise, devam et
records = records[done:]

# Embeddingleri biriktireceğiz (sadece yüzlüler)
emb_list = []

# Eğer embedding dosyası zaten varsa, üstüne eklemek için önce yükleyelim
if emb_path.exists():
    prev = np.load(emb_path)
    emb_list = [prev]  # sonra concatenate edeceğiz

# -----------------------------
# 5) İşleme döngüsü
# -----------------------------
with open(meta_path, "a", encoding="utf-8") as meta_f:
    for obj in tqdm(records, desc="FaceDB", unit="img"):
        p = obj["path"]
        md5 = obj["md5"]

        out = {
            "path": p,
            "md5": md5,
            "has_face": False,
            "det_conf": None,
            "bbox": None,
            "emb_index": None  # embedding listesinde kaçıncı satır
        }

        try:
            img = imread_unicode(p)
            if img is None:
                meta_f.write(json.dumps(out, ensure_ascii=False) + "\n")
                continue

            det = detect_largest_face_bgr(img, conf_th=0.6)
            if det is None:
                meta_f.write(json.dumps(out, ensure_ascii=False) + "\n")
                continue

            x1, y1, x2, y2, conf = det

            # margin
            h, w = img.shape[:2]
            mx = int(0.10 * (x2 - x1))
            my = int(0.10 * (y2 - y1))
            X1 = max(0, x1 - mx)
            Y1 = max(0, y1 - my)
            X2 = min(w - 1, x2 + mx)
            Y2 = min(h - 1, y2 + my)

            face = img[Y1:Y2, X1:X2]
            if face.size == 0:
                meta_f.write(json.dumps(out, ensure_ascii=False) + "\n")
                continue

            emb = arcface_embed(face)

            # embedding index: mevcut embedding sayısı
            # emb_list[0] varsa prev array; yoksa boş
            current_count = 0
            if len(emb_list) == 1 and isinstance(emb_list[0], np.ndarray):
                current_count = emb_list[0].shape[0]
            else:
                # gelecekte genişletirsek
                current_count = 0

            # Yeni embedding'i geçici listede tut
            # (disk yazmayı arada yapacağız)
            if "new_embs" not in out:
                pass

            # out'u güncelle
            out["has_face"] = True
            out["det_conf"] = float(conf)
            out["bbox"] = [int(X1), int(Y1), int(X2), int(Y2)]
            out["emb_index"] = current_count  # şimdilik prev sonuna eklenecek

            # Embedding'i ayrı listede biriktiriyoruz
            if "new_embs_list" not in globals():
                globals()["new_embs_list"] = []
            globals()["new_embs_list"].append(emb)

            meta_f.write(json.dumps(out, ensure_ascii=False) + "\n")

        except Exception:
            meta_f.write(json.dumps(out, ensure_ascii=False) + "\n")
            continue

# -----------------------------
# 6) Embeddingleri diske yaz
# -----------------------------
new_embs = globals().get("new_embs_list", [])
if len(new_embs) > 0:
    new_embs = np.stack(new_embs, axis=0)
    if emb_path.exists():
        prev = np.load(emb_path)
        all_embs = np.concatenate([prev, new_embs], axis=0)
    else:
        all_embs = new_embs
    np.save(emb_path, all_embs)
    print("Embedding kaydedildi:", emb_path, "shape:", all_embs.shape)
else:
    # hiç yeni yüz yoksa
    if not emb_path.exists():
        np.save(emb_path, np.zeros((0, 512), dtype=np.float32))
    print("Yeni embedding yok. Dosya:", emb_path)

print("Bitti ✅ meta:", meta_path)
