# Latent Manipulation of Brain MRI using Volume-Preserving GANs

Simulating brain aging through MRI images using a GAN model, combining volume-preserving preprocessing and latent space manipulation conditioned on age.

---

## Objectives

- Develop a GAN capable of manipulating the latent space to simulate age-related structural changes in the brain
- Apply preprocessing that preserves the original brain volume (resample to 1mm isotropic) prior to training
- Systematically modify the age attribute in the latent vector to investigate how brain structures evolve over time, while keeping other anatomical features unchanged

---

## Directory Structure

```
📁 latent_examination_imgs/     # Latent space analysis outputs (PCA, correlation, activation maps)
📁 latent_manipulation_imgs/    # Latent space manipulation outputs (conditional generation, latent directions)
📁 models_eval_res/             # Model evaluation results (SSIM, PSNR, VPS, comparison charts)
```

---

## Notebooks

| # | File | Description |
|---|------|-------------|
| 1 | `00a_preprocessing_2d.ipynb` | 2D MRI preprocessing: conform, skull-stripping, resample 1mm, save axial slice as PNG |
| 2 | `00b_preprocessing_3d.ipynb` | 3D MRI preprocessing: conform, skull-stripping, resample 1mm, save as NIfTI |
| 3 | `01_model2d_normalized.ipynb` | Train 2D GAN on normalized data (WGAN-GP + L1 + age regression) |
| 4 | `02_model2d_unnormalized.ipynb` | Train 2D GAN on unnormalized data |
| 5 | `03_compare_2d_models.ipynb` | Compare both 2D models, select the better one by SSIM |
| 6 | `04_model3d_normalized.ipynb` | Train 3D GAN on normalized data |
| 7 | `05_model3d_unnormalized.ipynb` | Train 3D GAN on unnormalized data |
| 8 | `06_compare_3d_models.ipynb` | Compare both 3D models, select the better one by SSIM |
| 9 | `07_latent_manipulation.ipynb` | Latent space manipulation: conditional generation and latent direction traversal |
| 10 | `08_model_evaluation.ipynb` | Comprehensive evaluation: SSIM, PSNR, VPS (Volume Preservation Score) |
| 11 | `09_latent_examination.ipynb` | Latent space analysis: PCA, correlation, activation maps, per-dimension analysis |

---

## Model Architecture

**Generator** — U-Net encoder-decoder with age conditioning at the bottleneck:
- Encoder: 8 layers (2D) / 4 layers (3D) with LeakyReLU
- Age injection: `z = age_fuse(concat(e_bottleneck, age_feat))` — concatenation rather than addition, allowing the model to learn its own balance between image features and age features
- Decoder: skip connections from encoder, BatchNorm + ReLU

**Discriminator** — PatchGAN with age map concatenation

**Loss function** — WGAN-GP + L1 reconstruction + age regression on generated images

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **SSIM** | Structural Similarity — measures how well anatomical brain structures are preserved |
| **PSNR** | Peak Signal-to-Noise Ratio — measures pixel-level reconstruction quality |
| **VPS** | Volume Preservation Score — measures brain volume preservation (closer to 1.0 = better) |

---

## Dataset

**12,000 open-source brain MRI scans** (ages 5–97), processed through the pipeline: Conform → Skull-stripping → Alignment → Resample 1mm isotropic. Split into normalized (size-standardized) and unnormalized subsets to compare the effect of preprocessing on GAN quality.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- nibabel, scikit-image, scikit-learn, matplotlib, pandas
- Kaggle GPU (T4 or P100) for training

---

## Running on Kaggle

1. Upload the dataset to Kaggle Datasets
2. Run notebooks sequentially from `01` through `09`
3. Checkpoints are automatically saved to the Kaggle Dataset after each best epoch
4. `03_compare` and `06_compare` automatically select the better model and save the result as JSON for downstream notebooks to read

---

## Results

| Model | SSIM |
|-------|------|
| GAN 2D Normalized | 0.9839 |
| GAN 3D Normalized | 0.9907 |
