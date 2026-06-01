# Beyond Memorization: Training-Free Style Mixing for Variability in Handwritten Text Generation Using Writer Embedding Injection in Pretrained Diffusion Models

**ICDAR 2025** — Published in *Document Analysis and Recognition – ICDAR 2025*, Springer Nature Switzerland, pp. 465–484.

<p align='center'>
  <b>
    <a href="https://doi.org/10.1007/978-3-032-04627-7_27">Paper (Springer)</a>
  </b>
</p>

> **Abstract:**
> Recent advancements in handwritten text generation using diffusion models have achieved high-quality and realistic handwriting synthesis. However, existing models often suffer from limited style variability, which reduces their effectiveness for downstream tasks like handwriting recognition and writer identification, which rely on diverse handwriting samples to ensure model generalization and robustness. Without sufficient variability, models trained on synthetic data risk overfitting to a narrow set of styles, limiting their applicability in real-world scenarios. Diffusion models typically require large-scale datasets to generalize effectively, but in the domain of handwriting generation, comparatively limited training data is available, leading to strong memorization tendencies. As a result, generated handwriting samples often replicate training data instead of introducing novel variations, further restricting their usefulness for downstream applications.

---

## Method Overview

- **Base model:** [WordStylist](https://github.com/koninik/WordStylist) — a pretrained latent diffusion model for styled handwritten word generation (ICDAR 2023)
- **Our contribution:** Training-free style mixing at inference time using character-level attention map localization and writer embedding injection
- **Key idea:** For a word with *N* characters, the U-Net attention maps identify each character's spatial region. A different writer's style embedding is injected into selected character regions, producing stylistically varied outputs without any additional training.

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

## Citation

If you use this code, please cite our ICDAR 2025 paper:

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
