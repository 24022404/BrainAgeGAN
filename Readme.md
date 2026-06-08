# Latent Manipulation of Brain MRI using Volume-Preserving GANs

Simulating brain aging through MRI images using conditional GANs, combining volume-preserving preprocessing with latent space analysis conditioned on age.

---

## Objectives

- Design a **volume-preserving preprocessing pipeline** that decouples inter-subject brain size variation from age-related change, enabling more stable GAN training
- Train **2D and 3D conditional GANs** (U-Net + PatchGAN + WGAN-GP) to synthesise the same brain at different target ages
- Evaluate models on **SSIM, PSNR, and VPS** (Volume Preservation Score)
- Analyse the learned latent space via **PCA and per-dimension Pearson correlation** to identify which dimensions encode age
- Construct **per-dimension activation maps** to localise which brain regions each age-sensitive dimension controls

---

## Directory Structure

```
📁 latent_examination_imgs/     # PCA scatter, correlation bar chart, activation maps (2D & 3D)
📁 latent_manipulation_imgs/    # Conditional generation & Real/Synthetic/Difference triplets
📁 models_eval_res/             # SSIM, PSNR, VPS results + comparison charts
```

---

## Notebooks

| # | File | Description |
|---|------|-------------|
| 1 | `00a_preprocessing_2d.ipynb` | 2D preprocessing: skull stripping, intensity normalisation, 1mm resample → axial PNG |
| 2 | `00b_preprocessing_3d.ipynb` | 3D preprocessing: skull stripping, intensity normalisation, 1mm resample → NIfTI |
| 3 | `01_model2d_normalized.ipynb` | Train 2D GAN on volume-preserving data (WGAN-GP + L1 + age regression) |
| 4 | `02_model2d_unnormalized.ipynb` | Train 2D GAN on unnormalised data (ablation) |
| 5 | `03_compare_2d_models.ipynb` | Compare 2D models by SSIM; select winner |
| 6 | `04_model3d_normalized.ipynb` | Train 3D GAN on volume-preserving data |
| 7 | `05_model3d_unnormalized.ipynb` | Train 3D GAN on unnormalised data (ablation) |
| 8 | `06_compare_3d_models.ipynb` | Compare 3D models by SSIM; select winner |
| 9 | `07_latent_manipulation.ipynb` | Conditional age synthesis: generate same brain at ages 20–80 |
| 10 | `08_model_evaluation.ipynb` | Full evaluation: SSIM, PSNR, VPS for all four model variants |
| 11 | `09_latent_examination.ipynb` | Latent space analysis: PCA, per-dimension correlation, activation maps |

---

## Model Architecture

**Generator** — Conditional U-Net with age injection at the bottleneck:

- **2D**: 8 encoder levels, Conv2d, input 256×256 PNG, output 256×256 synthetic slice
- **3D**: 4 encoder levels, Conv3d, input 64×64×64 NIfTI, output 64×64×64 synthetic volume
- **Age injection**: age scalar → 2-layer MLP (1→128→512) → concatenated with bottleneck features → fused via Conv1×1. Concatenation (not addition) allows the model to learn its own balance between anatomical and age signals.
- **Decoder**: skip connections from encoder + BatchNorm + ReLU + ConvTranspose

**Discriminator** — PatchGAN; receives image concatenated with age map as a 2-channel input; outputs patch-level realness score (no sigmoid).

**Loss function** — WGAN-GP + λ₁·L1 reconstruction (λ=10) + λ_age·age regression (λ=5)

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **SSIM** | Structural Similarity Index — perceptual fidelity (luminance, contrast, structure). Range 0→1, higher is better. |
| **PSNR** | Peak Signal-to-Noise Ratio — pixel-level reconstruction accuracy in dB. Higher is better. |
| **VPS** | Volume Preservation Score — ratio of synthesised to real grey-matter volume. Closer to 1.0 is better. |

---

## Dataset

**12,497 open-source T1-weighted brain MRI scans** (ages 18–97, median 25), drawn from 214 studies on OpenNeuro, ADNI, OASIS-3, IXI, CamCAN, and others. Each scan processed through: skull stripping (ANTsPyNet) → percentile intensity normalisation → 1mm isotropic resampling.

---

## Results

### Primary models (volume-preserving)

| Model | SSIM ↑ | PSNR ↑ (dB) | VPS ↑ |
|-------|--------|-------------|-------|
| 2D — Volume-Preserving | **0.9801** | **35.11** | **0.9984** |
| 3D — Volume-Preserving | 0.9734 | 30.95 | 0.9915 |

### Ablation (unnormalised, excluded from downstream analysis)

| Model | SSIM | PSNR (dB) | VPS |
|-------|------|-----------|-----|
| 2D — Unnormalised | 0.9750 | 36.30 | 0.9982 |
| 3D — Unnormalised | 0.9805 | 30.95 | 0.9974 |

> The 3D unnormalised model systematically inflates synthesised grey-matter volume by +0.26%, confirming that volume-preserving resampling is a necessary preprocessing step.

### Latent space

| | 2D model | 3D model |
|-|----------|----------|
| PCA PC1 variance | 93.5% | 80.5% |
| Max \|r\| (age) | 0.997 at dim 235 | 0.994 at dim 69 |
| Top-5 dims | 235, 151, 164, 226, 41 | 69, 160, 191, 180, 129 |
| Activation region | Posterior parietal cortex + ventricular margins | Same |

---

## Key Finding

The per-dimension activation maps align with real-world brain ageing anatomy — the model learned to associate its age-sensitive latent dimensions with the posterior parietal cortex and ventricular margins, which are well-established sites of age-related structural change (cortical thinning and ventricular enlargement). This was achieved **without any anatomical supervision**, validating that the GAN captured genuine biological ageing patterns.

Future models could restrict attention specifically to these identified regions rather than processing the entire brain, potentially improving training efficiency and diagnostic sensitivity for Alzheimer's disease and multiple sclerosis.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- `nibabel`, `antspynet`, `scikit-image`, `scikit-learn`, `matplotlib`, `pandas`
- Kaggle GPU (T4 or P100 recommended for training)

---

## Running on Kaggle

1. Upload the dataset to Kaggle Datasets
2. Run notebooks sequentially: `00a` → `00b` → `01` → .... → `09`
3. Checkpoints are saved to `/kaggle/working/` after each best epoch
4. `03_compare` and `06_compare` select the better model and write the result to JSON for downstream notebooks
5. `07_latent_manipulation` and `09_latent_examination` require the checkpoints from steps 3–4
