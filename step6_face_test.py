from pathlib import Path
import yaml, json
import numpy as np
import cv2
import onnxruntime as ort

# -----------------------------
# 0) Unicode-safe imread (Turkce karakterli path sorunu icin)
# -----------------------------
def imread_unicode(path_str):
    """
    OpenCV'nin Windows'ta Turkce karakterli path'lerde cv2.imread ile
    acamama sorununu cozer.
    Mantik: dosyayi byte olarak oku -> imdecode ile image'a cevir.
    """
    p = Path(path_str)
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

# -----------------------------
# 1) Config + index'ten ilk N path
# -----------------------------
N = 50

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

index_path = Path(cfg["db_dir"]) / "index.jsonl"
model_dir = Path("models")

paths = []
with open(index_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= N:
            break
        obj = json.loads(line)
        paths.append(obj["path"])

print("Test edilecek foto:", len(paths))

# -----------------------------
# 2) Face detector (OpenCV DNN - Caffe)
# -----------------------------
proto = str(model_dir / "deploy.prototxt")
caffemodel = str(model_dir / "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNetFromCaffe(proto, caffemodel)

# -----------------------------
# 3) ArcFace ONNX embedding session
# -----------------------------
arc_path = str(model_dir / "arcface.onnx")
sess = ort.InferenceSession(arc_path, providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name


def detect_largest_face_bgr(img_bgr, conf_th=0.6):
    """En büyük yüzü (x1,y1,x2,y2,conf) döndürür. Bulamazsa None."""
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        img_bgr, 1.0, (300, 300),
        (104.0, 177.0, 123.0),
        swapRB=False, crop=False
    )
    net.setInput(blob)
    det = net.forward()  # shape: (1,1,N,7)

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

        # clamp
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
    """
    Bu model NHWC bekliyor:
    - 112x112
    - (1,112,112,3)
    - RGB
    - normalize: (img - 127.5)/128
    - L2 normalize output
    """
    face = cv2.resize(face_bgr, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32)
    face = (face - 127.5) / 128.0
    face = face[None, ...]  # (1,112,112,3)

    emb = sess.run([out_name], {inp_name: face})[0][0]  # (D,)
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb


face_count = 0
no_face_count = 0
error_count = 0

for p in paths:
    try:
        img = imread_unicode(p)   # <-- DEGISTI: cv2.imread yerine unicode-safe
        if img is None:
            error_count += 1
            continue

        det = detect_largest_face_bgr(img, conf_th=0.6)
        if det is None:
            no_face_count += 1
            continue

        x1, y1, x2, y2, conf = det

        # biraz margin ekleyelim (yüz kenarı kesilmesin)
        h, w = img.shape[:2]
        mx = int(0.10 * (x2 - x1))
        my = int(0.10 * (y2 - y1))
        X1 = max(0, x1 - mx)
        Y1 = max(0, y1 - my)
        X2 = min(w - 1, x2 + mx)
        Y2 = min(h - 1, y2 + my)

        face = img[Y1:Y2, X1:X2]
        if face.size == 0:
            no_face_count += 1
            continue

        _emb = arcface_embed(face)  # embedding üretildi mi test
        face_count += 1

    except Exception as e:
        # İstersen hata mesajını görmek için:
        # print("HATA:", p, type(e).__name__, e)
        error_count += 1

print("Yüz var:", face_count)
print("Yüz yok:", no_face_count)
print("Hata:", error_count)
