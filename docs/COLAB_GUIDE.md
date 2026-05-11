# RVC Training on Google Colab (L4 GPU) — Complete Guide

This guide explains how to train RVC v2 speaker models on Google Colab using
an L4 GPU, your dataset tarballs on Google Drive, and your custom
`rvc_pipeline.py` CLI.

---

## Prerequisites

Before opening Colab:

1. **Push latest repo changes** to GitHub from your local machine:

   ```bash
   cd BengDiDa_preparation/rvc_reference_audio/Retrieval-based-Voice-Conversion-WebUI
   git add -A
   git commit -m "update pipeline"
   git push origin main
   ```

   The Colab notebook clones from `github.com/zakaria-kabir/Retrieval-based-Voice-Conversion-WebUI`.

2. **Upload dataset tarballs** to Google Drive at a path like:

   ```
   MyDrive/RVC_Datasets/sp021.tar.xz
   MyDrive/RVC_Datasets/sp022.tar.xz
   ...
   ```

   The tarball must extract to a folder **named exactly the same as `model_name`**
   (e.g., `sp021.tar.xz` → extracts to `sp021/`).

3. **Runtime type**: In Colab → Runtime → Change runtime type → **L4 GPU**.

---

## Step-by-Step Colab Session

> Run each cell in order. The entire pipeline (setup → train → backup)
> takes ~1–3 hours depending on dataset size and epoch count.

---

### Cell 1 — Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

---

### Cell 2 — System Setup

```python
import os, subprocess

# Install system deps
subprocess.run("apt-get -y install -q build-essential python3-dev ffmpeg aria2", shell=True)

# Check GPU
subprocess.run("nvidia-smi", shell=True)
print(f"CPU cores: {os.cpu_count()}")
```

---

### Cell 3 — Clone Your Repo

```python
GITHUB_REPO = "zakaria-kabir/Retrieval-based-Voice-Conversion-WebUI"
RVC_REPO = "/content/Retrieval-based-Voice-Conversion-WebUI"

if not os.path.exists(RVC_REPO):
    subprocess.run(f"git clone https://github.com/{GITHUB_REPO}.git {RVC_REPO}", shell=True, check=True)
else:
    print("Repo already cloned, pulling latest...")
    subprocess.run("git pull", shell=True, cwd=RVC_REPO)

%cd {RVC_REPO}
```

---

### Cell 4 — Install Python Dependencies

> This takes ~3-5 minutes. Only needs to run once per Colab session.

```python
# Uninstall conflicting packages first
subprocess.run("pip uninstall -y jax jaxlib tensorboard tensorflow -q", shell=True)

# Install RVC requirements
subprocess.run(f"pip install -r {RVC_REPO}/requirements-py311_updated.txt -q", shell=True, check=True)

# Extra packages needed by rvc_pipeline.py
subprocess.run("pip install pyyaml tqdm matplotlib -q", shell=True)

print("✅ Dependencies installed")
```

---

### Cell 5 — Download Pretrained Models

> Uses `aria2c` for fast parallel downloads. Skip if models are already
> cached from a previous session (check with `ls assets/pretrained_v2/`).

```python
import os

os.makedirs(f"{RVC_REPO}/assets/pretrained_v2", exist_ok=True)
os.makedirs(f"{RVC_REPO}/assets/hubert", exist_ok=True)
os.makedirs(f"{RVC_REPO}/assets/rmvpe", exist_ok=True)

HF_BASE = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"

# Download only what is missing
def dl(url, dest_dir, out_name):
    dest = f"{dest_dir}/{out_name}"
    if not os.path.exists(dest):
        subprocess.run(
            f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "{url}" -d "{dest_dir}" -o "{out_name}"',
            shell=True, check=True
        )
    else:
        print(f"⏩ Skipping {out_name} (already exists)")

# v2 pretrained (48k only — change if using different sample rate)
for f in ["f0G48k.pth", "f0D48k.pth"]:
    dl(f"{HF_BASE}/pretrained_v2/{f}", f"{RVC_REPO}/assets/pretrained_v2", f)

# Hubert + RMVPE
dl(f"{HF_BASE}/hubert_base.pt", f"{RVC_REPO}/assets/hubert", "hubert_base.pt")
dl(f"{HF_BASE}/rmvpe.pt", f"{RVC_REPO}/assets/rmvpe", "rmvpe.pt")

print("✅ Pretrained models ready")
```

---

### Cell 6 — Configure Speaker

> **Edit this cell for every new speaker you train.**

```python
# ══════════════════════════════════════════════
# EDIT THESE FOR EACH SPEAKER
SPEAKER_ID   = "sp021"
DRIVE_TAR    = f"/content/drive/MyDrive/RVC_Datasets/{SPEAKER_ID}.tar.xz"
DRIVE_BACKUP = "/content/drive/MyDrive/RVC_Backups"
# ══════════════════════════════════════════════

DATASET_DIR  = "/content/dataset"
CONFIG_PATH  = f"/content/{SPEAKER_ID}_config.yaml"

# Write YAML config for this speaker
import yaml

config = {
    "rvc_repo_root":     RVC_REPO,
    "backup_root":       DRIVE_BACKUP,
    "dataset_tar":       f"{DATASET_DIR}/{SPEAKER_ID}.tar.xz",
    "dataset_extract_dir": DATASET_DIR,
    "model_name":        SPEAKER_ID,
    "speaker_id":        0,
    "sample_rate":       "48k",
    "f0_method":         "rmvpe_gpu",   # L4: faster than cpu rmvpe
    "gpu":               "0",
    "batch_size":        32,            # L4 24GB: safe; try 40 if no OOM
    "total_epoch":       300,
    "save_every":        20,
    "version":           "v2",
    "use_f0":            True,
    "cache_data_in_gpu": True,          # L4 VRAM is large enough
    "save_latest_only":  False,         # keep all checkpoints on Drive
    "save_every_weights": True,
    "preprocess_per":    3.7,
    "no_parallel":       False,
    "threads":           os.cpu_count() or 4,
    "inference": {
        "model_weight": "", "index_file": "",
        "input_audio": "", "output_dir": "/content/inference_output",
        "output_format": "wav", "f0_up_key": 0, "f0_method": "rmvpe",
        "index_rate": 0.75, "filter_radius": 3, "resample_sr": 0,
        "rms_mix_rate": 0.25, "protect": 0.33, "f0_file": "",
        "randomize_params": False, "random_ranges": {},
    }
}

with open(CONFIG_PATH, "w") as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"✅ Config written to {CONFIG_PATH}")
print(f"   Speaker:    {SPEAKER_ID}")
print(f"   Drive tar:  {DRIVE_TAR}")
print(f"   Backup dir: {DRIVE_BACKUP}/{SPEAKER_ID}")
```

---

### Cell 7 — Copy Dataset from Drive to Colab SSD

> Copying to Colab's local SSD `/content/` is **critical** for training
> speed. Training directly from Drive is ~10x slower.

```python
import os, shutil

os.makedirs(DATASET_DIR, exist_ok=True)
local_tar = f"{DATASET_DIR}/{SPEAKER_ID}.tar.xz"

if not os.path.exists(local_tar):
    print(f"Copying {DRIVE_TAR} → {local_tar}  (this may take a few minutes)...")
    subprocess.run(
        f'rsync -ah --info=progress2 "{DRIVE_TAR}" "{local_tar}"',
        shell=True, check=True
    )
    print("✅ Copy complete")
else:
    print(f"✅ Tar already on local SSD: {local_tar}")
```

---

### Cell 8 — Preflight Check

```python
subprocess.run(
    f"python {RVC_REPO}/rvc_pipeline.py --config {CONFIG_PATH} train --dry-run",
    shell=True, cwd=RVC_REPO
)
```

---

### Cell 9 — Run Full Training Pipeline

> This runs all stages in sequence:
> Extract dataset → Preprocess → Extract F0 → Extract Features →
> Prepare filelist → Train → Build FAISS Index → Analyze log → Backup to Drive → Cleanup

```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

result = subprocess.run(
    f"python {RVC_REPO}/rvc_pipeline.py --config {CONFIG_PATH} train",
    shell=True, cwd=RVC_REPO
)
if result.returncode != 0:
    print("❌ Pipeline failed. Check output above.")
else:
    print("🎉 Training pipeline complete!")
```

---

### Cell 10 — (Optional) Plot Training Metrics

> Metrics are auto-saved to Drive by the pipeline, but you can plot
> interactively here too.

```python
subprocess.run(
    f"python {RVC_REPO}/rvc_pipeline.py --config {CONFIG_PATH} analyze",
    shell=True, cwd=RVC_REPO
)
```

---

### Cell 11 — (Optional) Manual Backup

> The pipeline runs backup automatically at the end. Use this if you
> stopped early or want to re-backup manually.

```python
subprocess.run(
    f"python {RVC_REPO}/rvc_pipeline.py --config {CONFIG_PATH} backup",
    shell=True, cwd=RVC_REPO
)
```

---

## Resuming Interrupted Training

Colab sessions can disconnect. The pipeline handles this gracefully:

1. Rerun **Cells 1–6** (setup).
2. Rerun **Cell 7** only if `/content/dataset/{SPEAKER_ID}/` was deleted.
3. **Do NOT rerun Cell 9 from scratch** — instead, run the individual
   stages manually. The RVC train script auto-detects the latest G*/D*
   checkpoint and resumes:

```python
# Resume training only (skip preprocessing/f0/features if already done)
import subprocess, os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

RVC_REPO    = "/content/Retrieval-based-Voice-Conversion-WebUI"
SPEAKER_ID  = "sp021"
SAMPLE_RATE = "48k"
GPU         = "0"
BATCH_SIZE  = 32
TOTAL_EPOCH = 300
SAVE_EVERY  = 20

subprocess.run(
    f"python infer/modules/train/train.py "
    f"-e {SPEAKER_ID} -sr {SAMPLE_RATE} -f0 1 "
    f"-bs {BATCH_SIZE} -g {GPU} -te {TOTAL_EPOCH} -se {SAVE_EVERY} "
    f"-pg assets/pretrained_v2/f0G{SAMPLE_RATE}.pth "
    f"-pd assets/pretrained_v2/f0D{SAMPLE_RATE}.pth "
    f"-l 0 -c 1 -sw 1 -v v2",
    shell=True, cwd=RVC_REPO, check=True
)
```

---

## Restoring from Drive (New Session, Already Trained)

```python
subprocess.run(
    f"python {RVC_REPO}/rvc_pipeline.py --config {CONFIG_PATH} restore",
    shell=True, cwd=RVC_REPO
)
```

---

## Training Multiple Speakers (Batch Loop)

```python
SPEAKERS = ["sp021", "sp022", "sp023"]  # edit as needed
DRIVE_DATASET_ROOT = "/content/drive/MyDrive/RVC_Datasets"
DRIVE_BACKUP       = "/content/drive/MyDrive/RVC_Backups"

for spk in SPEAKERS:
    print(f"\n{'='*60}\nTraining: {spk}\n{'='*60}")

    # Write config
    cfg = config.copy()
    cfg["model_name"]      = spk
    cfg["dataset_tar"]     = f"/content/dataset/{spk}.tar.xz"

    cfg_path = f"/content/{spk}_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    # Copy tar from Drive
    local_tar = f"/content/dataset/{spk}.tar.xz"
    if not os.path.exists(local_tar):
        subprocess.run(
            f'rsync -ah --info=progress2 "{DRIVE_DATASET_ROOT}/{spk}.tar.xz" "{local_tar}"',
            shell=True, check=True
        )

    # Train
    ret = subprocess.run(
        f"python {RVC_REPO}/rvc_pipeline.py --config {cfg_path} train",
        shell=True, cwd=RVC_REPO
    )
    if ret.returncode != 0:
        print(f"❌ {spk} failed — continuing to next speaker")
    else:
        # Free Colab SSD space before next speaker
        import shutil, glob
        shutil.rmtree(f"/content/dataset/{spk}", ignore_errors=True)
        os.remove(local_tar)
        for f in glob.glob(f"{RVC_REPO}/logs/{spk}/0_gt_wavs"):
            shutil.rmtree(f, ignore_errors=True)
        print(f"✅ {spk} done and cleaned up")
```

---

## L4 vs Local RTX 5060 — Setting Differences

| Setting                   | Local RTX 5060 (8GB)     | Colab L4 (24GB)          |
| ------------------------- | ------------------------ | ------------------------ |
| `batch_size`              | 8                        | **32–40**                |
| `cache_data_in_gpu`       | false                    | **true**                 |
| `f0_method`               | rmvpe                    | **rmvpe_gpu**            |
| `save_latest_only`        | true                     | **false**                |
| `threads`                 | 8                        | `os.cpu_count()`         |
| `PYTORCH_CUDA_ALLOC_CONF` | expandable_segments:True | expandable_segments:True |

---

## Drive Folder Structure After Training

```
MyDrive/RVC_Backups/
└── sp021/
    ├── sp021_e20.pth          ← inference weights (every save_every epoch)
    ├── sp021_e40.pth
    ├── ...
    ├── G_300.pth              ← latest G checkpoint (for resume)
    ├── D_300.pth              ← latest D checkpoint (for resume)
    ├── added_IVF*.index       ← FAISS retrieval index
    ├── config.json
    ├── filelist.txt
    ├── train_stdout.log
    ├── training_metrics.csv
    ├── training_metrics.png   ← loss curves plot
    └── best_checkpoints.json  ← top-3 epoch candidates
```

---

## Common Issues

### OOM during training

- Lower `batch_size` to 24 or 16.
- Set `cache_data_in_gpu: false`.
- Add to the cell before training: `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"`

### "Repo not found" preflight error

- Rerun Cell 3 (clone) — the session may have reset.

### f0 extraction hangs with `rmvpe_gpu`

- Switch `f0_method` to `rmvpe` (CPU) in the config — more stable on some Colab runtimes.

### Drive copy is slow

- Make sure you are using `rsync` not `shutil.copy` — rsync resumes partial transfers.

### Training doesn't resume after disconnect

- The G*/D* checkpoints must exist in `/content/Retrieval-based-Voice-Conversion-WebUI/logs/{SPEAKER_ID}/`.
- Run the restore cell first: `python rvc_pipeline.py --config ... restore`
- Then rerun just the training subprocess (Cell in "Resuming" section above).
