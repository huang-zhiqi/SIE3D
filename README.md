# SIE3D [ICASSP 2026]

The environment setup and usage for SIE3D are as follows.

## Installation

Tested environment:

- Python 3.9.16
- CUDA 11.8
- PyTorch 2.0.1

To set up from scratch, follow the steps below.

### 1. Clone the CUDA submodules

```bash
cd submodules
git clone --recursive https://github.com/YixunLiang/diff-gaussian-rasterization.git
git clone --recursive https://github.com/YixunLiang/simple-knn.git
cd ..
```

### 2. Create the conda environment

```bash
conda create -n sie3d python=3.9.16 cudatoolkit=11.8 -y
conda activate sie3d
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Build the local CUDA extensions

```bash
pip install submodules/diff-gaussian-rasterization/
pip install submodules/simple-knn/
```

### 5. Download the pretrained weights

```bash
python download_models.py
```

Notes:

- `download_models.py` downloads the Arc2Face and InsightFace weights to `./models/`.
- `DeepFace` may automatically download its own weight files the first time it runs.

## Data Preparation

- `--subject` should be the path to a subject folder, not a single image path.
- Put one or more face images in `jpg`, `jpeg`, or `png` format under `subjects/<name>/`.
- During training, the script automatically scans the directory and computes identity features.
- Use `configs/<name>/config.yaml` as the config file.
- The script automatically updates `subject_id`, `workspace`, and `C_batch_size` in the config, so you do not need to edit these fields manually.

Example directory structure:

```text
subjects/
  DenzelWashington/
    013_5928728c.jpg
```

If you want to train on your own data, you can copy `configs/config.yaml` and then modify `text`, `target_expression`, and `target_accessory` as needed.

## Run

Example command:

```bash
python train.py --opt ./configs/DenzelWashington/config.yaml --subject ./subjects/DenzelWashington --batch_size 4
```

Training outputs are saved to `subjects/DenzelWashington/splat/` by default.

Note: re-running the same subject will clear the existing `splat/` directory and generate it again.
