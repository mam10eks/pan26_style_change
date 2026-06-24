# PAN 2026 Multi-Author Writing Style Analysis

Inference code for our submission to the [PAN 2026 Multi-Author Writing Style
Analysis](https://pan.webis.de/clef26/pan26-web/style-change-detection.html)
shared task. A cross-encoder built on DeBERTa-v3 with length, domain, and
language conditioning, and an inference-time refinement step that aggregates
evidence from same-author neighbors.

## Quick start

```bash
# 1. Clone the repository
git clone <repo-url>
cd <repo-name>

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate           # Linux / macOS
# .\venv\Scripts\Activate.ps1      # Windows PowerShell

# 3. Install dependencies (includes PyTorch with CUDA support — see notes)
pip install -r requirements.txt

# 4. Verify CUDA is working (skip if you're running CPU-only):
python check_cuda.py

# 5. Download trained model weights from Google Drive
#    (URL provided separately by the repository owner)
python download_weights.py --url "<GOOGLE_DRIVE_LINK>"

# 6. Run inference on PAN-format input
python pan26_runner.py -i /path/to/input -o /path/to/output
```

The first inference run downloads the DeBERTa-v3 backbone from HuggingFace
(~280 MB for base, ~720 MB for large) and caches it under
`~/.cache/huggingface/`. Subsequent runs use the cache.

### CUDA version selection

`requirements.txt` defaults to **CUDA 12.6** wheels (`cu126`), which work
on most modern NVIDIA GPUs (Ampere, Ada Lovelace). To change:

- **Blackwell GPUs** (RTX Pro 6000, RTX 5090, etc.) → edit the
  `--extra-index-url` line in `requirements.txt` to use `cu128`.
- **Older drivers** (CUDA 11.x) → use `cu118`.
- **CPU-only** (no GPU) → change to `https://download.pytorch.org/whl/cpu`.

Check your driver's CUDA version with `nvidia-smi` (top-right corner).

If `check_cuda.py` reports CPU-only PyTorch despite having a GPU,
reinstall torch explicitly:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## Repository layout

| File | Purpose |
|------|---------|
| `pan26_runner.py` | Main entry point (CLI: `-i INPUT -o OUTPUT`) |
| `pan26_predict.py` | Inference engine (chunking + cross-pair evidence aggregation) |
| `load_model.py` | Constructs the `AVClassifier` and loads weights |
| `av_classifier.py` | Model architecture (must be filled in — see below) |
| `download_weights.py` | Downloads the trained weights from Google Drive |
| `check_cuda.py` | Verifies PyTorch was installed with CUDA support |
| `requirements.txt` | Python dependencies |

## Before first use: provide `av_classifier.py`

The repository ships `av_classifier.py` as a placeholder. Replace it with
the actual `AVClassifier` class definition used at training time (along
with `MultiLayerGatedFusion` and any other helper modules referenced in
its `forward()`). The interface expected by `load_model.py` is documented
in the placeholder's docstring.

## Input format

`pan26_runner.py` accepts two input layouts:

**1. Subfolders for each difficulty** (canonical PAN layout):

```
INPUT/
├── easy/
│   └── test/ (or train/, or validation/)
│       ├── problem-1.txt
│       └── ...
├── medium/
│   └── test/
│       └── ...
└── hard/
    └── test/
        └── ...
```

Outputs go to matching subfolders under `OUTPUT/`:

```
OUTPUT/
├── easy/solution-problem-1.json
├── medium/solution-problem-1.json
└── hard/solution-problem-1.json
```

**2. Single dataset directly under INPUT:**

```
INPUT/
├── problem-1.txt
├── problem-2.txt
└── ...
```

In this case all outputs go directly under `OUTPUT/`. The runner tries to
infer the difficulty from the path (looking for `easy`/`medium`/`hard` as
a path component); use `--dataset easy|medium|hard` to override.

Each `problem-{id}.txt` should contain one sentence per line. Output JSON
files have the format `{"changes": [0, 1, 0, ...]}`.

## Configuration

### Model checkpoint location

The default checkpoint path is `weights/av_classifier.pt`. Override with:

```bash
CHECKPOINT_PATH=/path/to/weights.pt python pan26_runner.py -i ... -o ...
```

### Backbone choice

Defaults to `microsoft/mdeberta-v3-base`. To use a different backbone:

```bash
BACKBONE_HF_NAME=microsoft/deberta-v3-large python pan26_runner.py -i ... -o ...
```

The backbone identifier must match the one used at training time — the
state_dict's parameter shapes are tied to the backbone's hidden size and
layer count. If they mismatch, `load_state_dict` will raise.

### Inference hyperparameters

The defaults at the top of `pan26_runner.py` control all behavior:

```python
INFERENCE_DEFAULTS = dict(
    max_length=250,                  # per-side cap on chunk tokenization
    batch_size=32,
    threshold=0.5,                   # final-output decision threshold
    merging_threshold=0.2,           # stricter threshold for same-author detection
    concat_left_budget=493,          # token budget per side in concat-evidence
    concat_right_budget=493,
    concat_iterations=2,
    concat_direction="both",
    concat_shared_budget=509,        # total budget (fits a 512-token model)
    concat_shared_anchor="auto",
    concat_shared_min_per_side=16,
)
```

To switch from concat-evidence to the lighter evidence-window refinement,
replace the `concat_*` keys with:

```python
evidence_window=5,
evidence_iterations=2,
evidence_aggregate="mean",         # mean / max / min / median / logodds
evidence_direction="both",
```

The two refinement modes are mutually exclusive.

## Output verification

The official PAN verifier checks that outputs are formatted correctly:

```bash
git clone https://github.com/pan-webis-de/pan-code.git
python pan-code/clef25/multi-author-analysis/output_verifier/output_verifier.py \
    --input  /path/to/input \
    --output /path/to/output
```

All problems should report `OK`. `INVALID_LENGTH` errors indicate a
mismatch between predicted output length and the verifier's expected
`raw.count("\n") + 1 - 1`; `pan26_predict.py` pads/truncates internally
to handle this, but check the data layout if it appears.

## Hardware

CUDA-capable GPU is recommended but not required. CPU inference works,
just slower:

- mDeBERTa-v3-base on 1000 problems: ~5 min on A100, ~45 min on CPU
- DeBERTa-v3-large on 1000 problems: ~12 min on A100, ~2 hr on CPU

## Submission to TIRA

```bash
tira-cli --dry-run code-submission --path . --task multi-author-writing-style-analysis-2026 --dataset smoketest-20260330-training --command 'python3 pan26_runner.py -i $inputDataset -o $outputDir' --mount-hf-model microsoft/deberta-v3-large
```

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{your_paper_2026,
  author    = {...},
  title     = {...},
  booktitle = {Working Notes of CLEF 2026},
  year      = {2026},
}
```

## License

See the LICENSE file.
