# Latent Manipulation of Brain MRI using Volume-Preserving GANs

Mô phỏng quá trình lão hóa não bộ qua ảnh MRI bằng mô hình GAN, kết hợp kỹ thuật tiền xử lý bảo toàn thể tích não và thao túng không gian tiềm ẩn (latent space) theo thuộc tính tuổi.

---

## Mục tiêu

- Phát triển mô hình GAN có khả năng thao túng latent space để mô phỏng các thay đổi não bộ theo tuổi
- Áp dụng tiền xử lý bảo toàn thể tích não gốc (resample 1mm isotropic) trước khi huấn luyện
- Thay đổi có hệ thống thuộc tính tuổi trong latent vector để khảo sát sự tiến hóa cấu trúc não theo thời gian, trong khi giữ nguyên các đặc trưng giải phẫu khác

---

## Cấu trúc thư mục

```
📁 latent_examination_imgs/     # Kết quả phân tích latent space (PCA, correlation, activation map)
📁 latent_manipulation_imgs/    # Kết quả thao túng latent space (conditional generation, latent direction)
📁 models_eval_res/             # Kết quả đánh giá mô hình (SSIM, PSNR, VPS, comparison charts)
```

---

## Danh sách Notebook

| # | File | Mô tả |
|---|------|--------|
| 1 | `00a_preprocessing_2d.ipynb` | Tiền xử lý ảnh MRI 2D: conform, skull-stripping, resample 1mm, lưu axial slice PNG |
| 2 | `00b_preprocessing_3d.ipynb` | Tiền xử lý ảnh MRI 3D: conform, skull-stripping, resample 1mm, lưu NIfTI |
| 3 | `01_model2d_normalized.ipynb` | Huấn luyện GAN 2D trên dữ liệu đã normalized (WGAN-GP + L1 + age regression) |
| 4 | `02_model2d_unnormalized.ipynb` | Huấn luyện GAN 2D trên dữ liệu chưa normalized |
| 5 | `03_compare_2d_models.ipynb` | So sánh 2 model 2D, chọn model tốt hơn theo SSIM |
| 6 | `04_model3d_normalized.ipynb` | Huấn luyện GAN 3D trên dữ liệu đã normalized |
| 7 | `05_model3d_unnormalized.ipynb` | Huấn luyện GAN 3D trên dữ liệu chưa normalized |
| 8 | `06_compare_3d_models.ipynb` | So sánh 2 model 3D, chọn model tốt hơn theo SSIM |
| 9 | `07_latent_manipulation.ipynb` | Thao túng latent space: Conditional generation và Latent Direction manipulation |
| 10 | `08_model_evaluation.ipynb` | Đánh giá toàn diện: SSIM, PSNR, VPS (Volume Preservation Score) |
| 11 | `09_latent_examination.ipynb` | Phân tích latent space: PCA, correlation, activation map, per-dimension analysis |

---

## Kiến trúc mô hình

**Generator** — U-Net encoder-decoder với age conditioning tại bottleneck:
- Encoder: 8 layer (2D) / 4 layer (3D) với LeakyReLU
- Age injection: `z = age_fuse(concat(e_bottleneck, age_feat))` — concat thay vì cộng để model tự học cân bằng giữa image features và age features
- Decoder: skip connections từ encoder, BatchNorm + ReLU

**Discriminator** — PatchGAN với age map concatenation

**Loss function** — WGAN-GP + L1 reconstruction + age regression trên ảnh fake

---

## Metrics đánh giá

| Metric | Ý nghĩa |
|--------|---------|
| **SSIM** | Structural Similarity — đo mức độ giữ nguyên cấu trúc giải phẫu não |
| **PSNR** | Peak Signal-to-Noise Ratio — đo chất lượng pixel-level |
| **VPS** | Volume Preservation Score — đo mức độ bảo toàn thể tích não (gần 1.0 = tốt) |

---

## Dataset

Hơn 12000 ảnh MRI não open-source, xử lý qua pipeline: Conform → Skull-stripping → Alignment → Resample 1mm isotropic. Chia thành normalized (chuẩn hóa kích thước) và unnormalized để so sánh ảnh hưởng của tiền xử lý lên chất lượng GAN.

---

## Yêu cầu

- Python 3.10+
- PyTorch 2.0+
- nibabel, scikit-image, scikit-learn, matplotlib, pandas
- Kaggle GPU (T4 hoặc P100) để huấn luyện

---

## Chạy trên Kaggle

1. Upload dataset lên Kaggle Datasets
2. Chạy lần lượt từ `00a` → `09` theo thứ tự
3. Checkpoint được tự động lưu lên Kaggle Dataset sau mỗi epoch tốt nhất
4. Các file `03_compare` và `06_compare` chọn tự động model tốt hơn và lưu kết quả JSON để các file sau đọc

---

## Kết quả

| Model | SSIM |
|-------|------|
| GAN 2D Normalized | 0.9839 |
| GAN 3D Normalized | 0.9907 |