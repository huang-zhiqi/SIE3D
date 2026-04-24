import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DEEPFACE_EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]
DEEPFACE_EMOTION_WEIGHTS_URL = (
    "https://github.com/serengil/deepface_models/releases/download/v1.0/"
    "facial_expression_model_weights.h5"
)
DEFAULT_DEEPFACE_EMOTION_DIR = Path("models") / "deepface_emotion"
DEFAULT_DEEPFACE_EMOTION_H5 = DEFAULT_DEEPFACE_EMOTION_DIR / "facial_expression_model_weights.h5"
DEFAULT_DEEPFACE_EMOTION_PTH = DEFAULT_DEEPFACE_EMOTION_DIR / "facial_expression_model_weights.pth"
def _ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _download_file(url, target_path):
    _ensure_parent_dir(target_path)
    urllib.request.urlretrieve(url, target_path)


def _extract_layer_name(dataset_path):
    candidates = [part for part in dataset_path.split("/") if part.startswith(("conv2d", "dense"))]
    if not candidates:
        return None
    return candidates[-1]


def _load_h5_layer_arrays(weight_file):
    try:
        import h5py
    except ImportError as e:
        raise RuntimeError(
            "h5py is required to load the official DeepFace emotion weights. "
            "Please install it before enabling the expression loss."
        ) from e

    with h5py.File(weight_file, "r") as f:
        root = f["model_weights"] if "model_weights" in f else f
        grouped = {}

        def _visitor(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return

            if name.endswith("kernel:0"):
                kind = "kernel"
            elif name.endswith("bias:0"):
                kind = "bias"
            else:
                return

            layer_name = _extract_layer_name(name)
            if layer_name is None:
                return

            grouped.setdefault(layer_name, {})[kind] = obj[()]

        root.visititems(_visitor)

    return grouped


def _copy_keras_weights(module, kernel, bias):
    kernel_tensor = torch.from_numpy(kernel).float()
    bias_tensor = torch.from_numpy(bias).float()

    with torch.no_grad():
        if isinstance(module, nn.Conv2d):
            module.weight.copy_(kernel_tensor.permute(3, 2, 0, 1))
        elif isinstance(module, nn.Linear):
            module.weight.copy_(kernel_tensor.t())
        else:
            raise TypeError(f"Unsupported module type for DeepFace weight loading: {type(module)}")

        module.bias.copy_(bias_tensor)


class DeepFaceEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=2)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(128, 1024)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(1024, len(DEEPFACE_EMOTION_LABELS))

    def preprocess(self, images):
        # Match DeepFace's emotion preprocessing while keeping everything in torch.
        images = torch.clamp(images, 0.0, 1.0)
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)

        gray = (
            0.299 * images[:, 0:1, :, :]
            + 0.587 * images[:, 1:2, :, :]
            + 0.114 * images[:, 2:3, :, :]
        )
        gray = F.interpolate(gray, size=(48, 48), mode="bilinear", align_corners=False)
        return gray

    def forward(self, images):
        x = self.preprocess(images)

        x = F.relu(self.conv1(x), inplace=True)
        x = self.pool1(x)

        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = self.pool2(x)

        x = F.relu(self.conv4(x), inplace=True)
        x = F.relu(self.conv5(x), inplace=True)
        x = self.pool3(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def load_deepface_emotion_model(weight_dir=None):
    weight_dir = Path(weight_dir) if weight_dir is not None else DEFAULT_DEEPFACE_EMOTION_DIR
    h5_path = weight_dir / DEFAULT_DEEPFACE_EMOTION_H5.name
    pth_path = weight_dir / DEFAULT_DEEPFACE_EMOTION_PTH.name

    model = DeepFaceEmotionModel()

    if pth_path.is_file():
        state_dict = torch.load(pth_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        if not h5_path.is_file():
            _download_file(DEEPFACE_EMOTION_WEIGHTS_URL, h5_path)

        layer_arrays = _load_h5_layer_arrays(str(h5_path))
        layer_mapping = [
            ("conv2d", model.conv1),
            ("conv2d_1", model.conv2),
            ("conv2d_2", model.conv3),
            ("conv2d_3", model.conv4),
            ("conv2d_4", model.conv5),
            ("dense", model.fc1),
            ("dense_1", model.fc2),
            ("dense_2", model.fc3),
        ]

        for layer_name, module in layer_mapping:
            if layer_name not in layer_arrays:
                raise RuntimeError(
                    f"Could not find layer '{layer_name}' in the DeepFace emotion weight file."
                )

            layer_group = layer_arrays[layer_name]
            _copy_keras_weights(module, layer_group["kernel"], layer_group["bias"])

        _ensure_parent_dir(pth_path)
        torch.save(model.state_dict(), pth_path)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model
