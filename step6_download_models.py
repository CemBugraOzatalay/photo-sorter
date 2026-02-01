from pathlib import Path
import requests

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

files = [
    # OpenCV DNN face detector (Caffe)
    ("deploy.prototxt",
     "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"),
    ("res10_300x300_ssd_iter_140000.caffemodel",
     "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"),

    # ArcFace embedding (ONNX) - ALTERNATIF 1 (InsightFace releases mirror)
     ("arcface.onnx",
      "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx?download=true"),


]


def download(url, path: Path):
    if path.exists() and path.stat().st_size > 0:
        print("Zaten var:", path.name)
        return
    print("İndiriliyor:", path.name)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    print("OK:", path.name, "->", path.stat().st_size, "bytes")

for name, url in files:
    download(url, MODELS_DIR / name)

print("Bitti ✅")
