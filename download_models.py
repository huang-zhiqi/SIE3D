import os
import shutil
import urllib.request
from huggingface_hub import hf_hub_download

base = "./models"
os.makedirs(os.path.join(base, "antelopev2"), exist_ok=True)
os.makedirs(os.path.join(base, "arc2face"), exist_ok=True)
os.makedirs(os.path.join(base, "encoder"), exist_ok=True)
os.makedirs(os.path.join(base, "deepface_emotion"), exist_ok=True)

for f in ["scrfd_10g_bnkps.onnx", "2d106det.onnx", "1k3d68.onnx", "genderage.onnx", "glintr100.onnx"]:
    src = hf_hub_download("lithiumice/insightface", f"models/antelopev2/{f}", token=False)
    shutil.copy(src, os.path.join(base, "antelopev2", f))

src = hf_hub_download("FoivosPar/Arc2Face", "arcface.onnx", token=False)
shutil.copy(src, os.path.join(base, "antelopev2", "arcface.onnx"))

for f in ["config.json", "diffusion_pytorch_model.safetensors"]:
    src = hf_hub_download("FoivosPar/Arc2Face", f"arc2face/{f}", token=False)
    shutil.copy(src, os.path.join(base, "arc2face", f))

for f in ["config.json", "pytorch_model.bin"]:
    src = hf_hub_download("FoivosPar/Arc2Face", f"encoder/{f}", token=False)
    shutil.copy(src, os.path.join(base, "encoder", f))

emotion_weight_url = (
    "https://github.com/serengil/deepface_models/releases/download/v1.0/"
    "facial_expression_model_weights.h5"
)
emotion_weight_path = os.path.join(base, "deepface_emotion", "facial_expression_model_weights.h5")
if not os.path.exists(emotion_weight_path):
    urllib.request.urlretrieve(emotion_weight_url, emotion_weight_path)
