# Beyond Memorization: Training-Free Style Mixing for Variability in Handwritten Text Generation Using Writer Embedding Injection in Pretrained Diffusion Models

<p align='center'>
  <b>
    <a href="#">Paper (coming soon)</a>
  </b>
</p>

> **Abstract:**
> We propose a training-free style mixing approach for handwritten text generation that injects multiple writer embeddings at the character level into a pretrained diffusion model (WordStylist). At inference time, attention maps from the U-Net middle block spatially localize individual characters, and a randomly selected writer's embedding is blended into the corresponding region — producing variability in generated handwriting styles without any additional training.

---

## Method Overview

- **Base model:** [WordStylist](https://github.com/koninik/WordStylist) — a pretrained latent diffusion model for styled handwritten word generation (ICDAR 2023)
- **Our contribution:** Training-free style mixing at inference time using character-level attention map localization and writer embedding injection
- **Key idea:** For a word with *N* characters, the U-Net attention maps identify each character's spatial region. A different writer's style embedding is injected into selected character regions, producing stylistically varied outputs.

---

## Requirements

```bash
pip install torch torchvision diffusers transformers einops wandb timm scikit-image pandas
```

---

## Pre-trained Model

Download the WordStylist trained model weights:
[trained_model](https://drive.google.com/file/d/1XVRUXSJw0PaNgrtFH_mNHceFO-Ouf_xz/view?usp=share_link)

Place the model under the path specified in `config.py` (`authorBasePath`).

---

## Configuration

Edit `config.py` to set:
- `iam_path` — path to preprocessed IAM word images
- `authorBasePath` — path to the pretrained WordStylist model
- `save_path` — where to save generated images and attention maps
- `MAX_CHARS` — maximum word length (default: 25)

---

## Running Inference

```bash
python regFrmTrnVariStyleMixOcr.py --batch_size 4 --epochs 1
```

**Output structure:**
```
save_path/
  noChange/                  ← generated word images (original writer style)
    attentionMaps/           ← per-character attention map PNGs
```

Each attention map filename encodes: `imagename_charIndex_charLetter_....png`

---

## Key Files

| File | Description |
|------|-------------|
| `regFrmTrnVariStyleMixOcr.py` | Main inference script |
| `unetVarStleMixExp4.py` | Modified U-Net with attention visualization and style injection |
| `config.py` | All configuration (paths, MAX_CHARS, model names) |
| `utils/saveAttentionMaps.py` | Attention map visualization and saving utilities |
| `utils/dataset.py` | IAM dataset loader |
| `htr/` | HTR OCR model for word-level filtering |
| `ResPhoSCNetZSL/` | PHOSC character embedding module |
| `gt/gany.filter27` | IAM training word list |

---

## Code Credits

This work builds directly on top of **WordStylist** ([koninik/WordStylist](https://github.com/koninik/WordStylist)). The base diffusion model, U-Net architecture, training pipeline, and dataset preprocessing are from WordStylist. We extend it with training-free style mixing at inference time.

> Nikolaidou, K., Retsinas, G., Christlein, V., Seuret, M., Sfikas, G., Smith, E. B., Mokayed, H., & Liwicki, M. (2023). *WordStylist: Styled Verbatim Handwritten Text Generation with Latent Diffusion Models*. ICDAR 2023.

```bibtex
@article{nikolaidou2023wordstylist,
  title={{WordStylist: Styled Verbatim Handwritten Text Generation with Latent Diffusion Models}},
  author={Nikolaidou, Konstantina and Retsinas, George and Christlein, Vincent and Seuret, Mathias and Sfikas, Giorgos and Smith, Elisa Barney and Mokayed, Hamam and Liwicki, Marcus},
  journal={arXiv preprint arXiv:2303.16576},
  year={2023}
}
```

We also thank the authors of [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [HTR best practices](https://github.com/georgeretsi/HTR-best-practices), and [GANwriting](https://github.com/omni-us/research-GANwriting) for their open-source contributions.

---

## Citation

If you use this code, please cite:
```bibtex
@article{gurav2025beyondmemorization,
  title={{Beyond Memorization: Training-Free Style Mixing for Variability in Handwritten Text Generation Using Writer Embedding Injection in Pretrained Diffusion Models}},
  author={Gurav, Aniket and others},
  year={2025}
}
```
