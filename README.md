# Beyond Memorization: Training-Free Style Mixing for Variability in Handwritten Text Generation Using Writer Embedding Injection in Pretrained Diffusion Models

**ICDAR 2025** — Published in *Document Analysis and Recognition – ICDAR 2025*, Springer Nature Switzerland, pp. 465–484.

<p align='center'>
  <b>
    <a href="https://doi.org/10.1007/978-3-032-04627-7_27">Paper (Springer)</a>
  </b>
</p>

> **Abstract:**
> Recent advancements in handwritten text generation using diffusion models have achieved high-quality and realistic handwriting synthesis. However, existing models often suffer from limited style variability, which reduces their effectiveness for downstream tasks like handwriting recognition and writer identification, which rely on diverse handwriting samples to ensure model generalization and robustness. Without sufficient variability, models trained on synthetic data risk overfitting to a narrow set of styles, limiting their applicability in real-world scenarios.

---

## Attention Map Visualization

Per-character attention maps showing how the model localizes each character in the generated word image:

![Attention Maps Preview](attentionMaps_preview.gif)

Each frame shows the attention map for one character overlaid on the generated word image. These maps are used to spatially inject a different writer's style at the character level.

---

## Method Overview

- **Base model:** [WordStylist](https://github.com/koninik/WordStylist) — pretrained latent diffusion model for styled handwritten word generation (ICDAR 2023)
- **Our contribution:** Training-free style mixing using character-level attention map localization and writer embedding injection
- **No retraining required** — runs purely at inference time on the pretrained WordStylist model

---

## Requirements

```bash
pip install -r requirements.txt
```

Key packages: `torch>=2.4.1`, `diffusers>=0.35.1`, `transformers>=4.46.3`, `einops`, `timm`, `scikit-image`

---

## Setup: Paths to Change Before Running

> ⚠️ **This is the most important step.** The code has several paths you must update for your system.

### 1. `config.py` — Primary configuration (edit these 3 paths)

| Variable | Line | What to set |
|----------|------|-------------|
| `iam_path` | ~36 | Path to preprocessed IAM word images (64×256 PNG crops) |
| `authorBasePath` | ~44 | Directory containing the WordStylist pretrained model |
| `save_path` | ~105 | Directory where generated images and attention maps will be saved |

Example:
```python
# config.py
iam_path = [
    "/your/path/to/allCrops_preprocess/"   # ← change this
][dataIndx]

authorBasePath = "/your/path/to/wordStylist/models/"  # ← change this

save_path = ["/your/path/to/output/results/"][allInOneIndx]  # ← change this
```

### 2. `regFrmTrnVariStyleMixOcr.py` — Two hardcoded model paths (edit these)

| Line | Variable | What to set |
|------|----------|-------------|
| ~1380 | `--loadPrevPath` | Path to the pretrained HTR (OCR) model `.pt` file |
| ~1492 | `modelPath` | Path to the WordStylist EMA model (`ema_ckpt.pt`) |

```python
# Line ~1380
parser.add_argument('--loadPrevPath', default="/your/path/to/htr_model.pt")

# Line ~1492
modelPath = "/your/path/to/wordStylist/models/ema_ckpt.pt"
```

### 3. `regFrmTrnVariStyleMixOcr.py` — Command-line argument (can also pass at runtime)

```bash
python regFrmTrnVariStyleMixOcr.py \
  --iam_path /your/path/to/allCrops_preprocess/ \
  --batch_size 4 --epochs 1
```

---

## Pre-trained Models

| Model | Download | Place at |
|-------|----------|----------|
| WordStylist (EMA) | [Google Drive](https://drive.google.com/file/d/1XVRUXSJw0PaNgrtFH_mNHceFO-Ouf_xz/view?usp=share_link) | `authorBasePath` in `config.py` |
| HTR/OCR model | From [HTR best practices](https://github.com/georgeretsi/HTR-best-practices) | `--loadPrevPath` argument |

---

## IAM Dataset Preprocessing

Download IAM `data/words.tgz` from [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database), then preprocess:

```bash
python prepare_images.py
```
*(sets `iam_path` to the output folder)*

---

## Running Inference

```bash
python regFrmTrnVariStyleMixOcr.py --batch_size 4 --epochs 1
```

To limit to 200 images (fast test):
> The script runs 50 batches by default (`if i >= 50: break` in the loader loop). Adjust in `regFrmTrnVariStyleMixOcr.py` line ~980.

---

## Output: Where to Find Generated Images and Attention Maps

After running, all outputs are saved under `save_path` (set in `config.py`):

```
save_path/
└── noChange/                        ← Generated word images (original writer style)
    │   a03-034-01-03_049_116_New__called_0.png
    │   b06-019-00-07_128_085_New__Herr_0.png
    │   ...
    └── attentionMaps/               ← Per-character attention map visualizations
            a03-034-01-03_049_116_New__called_0_c_0_char_att0_val2_rollMins4.png
            a03-034-01-03_049_116_New__called_1_a_0_char_att0_val2_rollMins4.png
            a03-034-01-03_049_116_New__called_2_l_0_char_att0_val2_rollMins4.png
            ...
```

**Filename format for word images:**
```
{imageID}_{originalWriterID}_{shuffledWriterID}_New__{word}_{epoch}.png
```

**Filename format for attention maps:**
```
{imageID}_{writerIDs}_New__{word}_{charIndex}_{charLetter}_char_att0_val2_rollMins4.png
```

For each generated word, you get one attention map PNG per character (e.g. 6 maps for a 6-letter word).

---

## Citation

```bibtex
@InProceedings{10.1007/978-3-032-04627-7_27,
  author="Gurav, Aniket and Chanda, Sukalpa and Krishnan, Narayanan C.",
  editor="Yin, Xu-Cheng and Karatzas, Dimosthenis and Lopresti, Daniel",
  title="Beyond Memorization: Training-Free Style Mixing for Variability in Handwritten Text Generation Using Writer Embedding Injection in Pretrained Diffusion Models",
  booktitle="Document Analysis and Recognition -- ICDAR 2025",
  year="2026",
  publisher="Springer Nature Switzerland",
  address="Cham",
  pages="465--484",
  isbn="978-3-032-04627-7"
}
```

---

## Code Credits

Built on top of **WordStylist** ([koninik/WordStylist](https://github.com/koninik/WordStylist)). The base diffusion model, U-Net architecture, and dataset pipeline are from WordStylist. We extend it with training-free style mixing at inference time.

```bibtex
@article{nikolaidou2023wordstylist,
  title={{WordStylist: Styled Verbatim Handwritten Text Generation with Latent Diffusion Models}},
  author={Nikolaidou, Konstantina and Retsinas, George and Christlein, Vincent and Seuret, Mathias and Sfikas, Giorgos and Smith, Elisa Barney and Mokayed, Hamam and Liwicki, Marcus},
  journal={arXiv preprint arXiv:2303.16576},
  year={2023}
}
```

Also thanks to [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [HTR best practices](https://github.com/georgeretsi/HTR-best-practices), and [GANwriting](https://github.com/omni-us/research-GANwriting).
