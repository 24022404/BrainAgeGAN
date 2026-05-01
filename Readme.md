# Latent Manipulation of Brain MRI using Volume-Preserving GANs

## 📋 Phân công công việc

Xem chi tiết phân công tại: [Google Docs](https://docs.google.com/document/d/1RU2WGWDfILGBAOdBGp-vDyFSmVqaHril0cuCsTkOCV8/edit?hl=vi&tab=t.0)

# Latent Manipulation of Brain MRI using Volume-Preserving GANs

Mô phỏng quá trình lão hóa não bộ bằng cách thao tác trên không gian tiềm ẩn (latent space) của GAN, trong khi giữ nguyên các đặc điểm giải phẫu.

---

## Mô tả

Dự án xây dựng mô hình GAN có khả năng:
- Học phân phối ảnh MRI não theo tuổi
- Thay đổi tuổi trong latent space → sinh ra ảnh não ở các mốc tuổi khác nhau
- Giữ nguyên cấu trúc giải phẫu, chỉ thay đổi đặc điểm lão hóa

---

## Dataset

- **Nguồn:** Tổng hợp từ nhiều dataset mã nguồn mở (ABIDE, ADHD, ADNI, OpenNeuro,...)
- **Số lượng:** ~12,499 subjects
- **Định dạng:** NIfTI (`.nii.gz`), T1-weighted MRI
- **Tuổi:** 5 – 97 tuổi

---

## Cấu trúc project

```
├── 00a_preprocessing_2d.ipynb   # Raw MRI → PNG 2D (normalized + unnormalized)
├── 00b_preprocessing_3d.ipynb   # Raw MRI → NIfTI 3D (normalized + unnormalized)
├── 01_model2d_normalized.ipynb  # Train GAN 2D trên data normalized
├── 02_model2d_unnormalized.ipynb# Train GAN 2D trên data unnormalized
├── 03_compare_2d.ipynb          # So sánh 2 model 2D, chọn model tốt hơn
├── 04_model3d_normalized.ipynb  # Train GAN 3D trên data normalized
├── 05_model3d_unnormalized.ipynb# Train GAN 3D trên data unnormalized
├── 06_compare_3d.ipynb          # So sánh 2 model 3D, chọn model tốt hơn
├── 07_latent_manipulation.ipynb # Simulate aging với model tốt nhất
└── 08_evaluation.ipynb          # Đánh giá hiệu suất toàn bộ pipeline
```

---

## Pipeline

```
Raw .nii.gz
    │
    ├── 00a ──► PNG 2D (normalized / unnormalized)
    │               │
    │           01, 02 ──► GAN 2D × 2
    │               │
    │            03 ──► So sánh → chọn model 2D tốt nhất
    │
    ├── 00b ──► NIfTI 3D (normalized / unnormalized)
    │               │
    │           04, 05 ──► GAN 3D × 2
    │               │
    │            06 ──► So sánh → chọn model 3D tốt nhất
    │
    ├── 07 ──► Latent Manipulation (simulate aging)
    └── 08 ──► Evaluation (SSIM, Loss_G)
```

---

## Kiến trúc Model

**Generator:** U-Net với Age Embedding inject vào bottleneck
- 2D: `Conv2d`, input `(B, 1, 256, 256)`
- 3D: `Conv3d`, input `(B, 1, 64, 64, 64)`

**Discriminator:** PatchGAN với Age Conditioning
- Age được broadcast thành channel và concat vào input

**Loss Functions:**
- `Adversarial loss` — đánh lừa Discriminator
- `L1 loss` — giữ nguyên cấu trúc giải phẫu
- `Age regression loss` — đảm bảo ảnh sinh ra đúng tuổi

---

## Preprocessing

Dùng **ANTsPy**:
- `antspynet.brain_extraction()` — skull-stripping (loại bỏ hộp sọ)
- `ants.resample_image()` — chuẩn hóa về 1mm isotropic (normalized output)

**2 loại output:**
- `normalized` — resample về 1mm isotropic, tất cả não cùng scale
- `unnormalized` — giữ nguyên kích thước gốc

---

## Metrics đánh giá

| Metric | Mô tả |
|--------|-------|
| **SSIM** | Structural Similarity — độ giữ nguyên cấu trúc giải phẫu |
| **Loss_G** | Generator loss — độ thuyết phục của ảnh sinh ra |

---

## Môi trường

- **Platform:** Kaggle Notebooks (GPU T4)
- **Python:** 3.10
- **Thư viện chính:** PyTorch, ANTsPy, ANTsPyNet, nibabel, scikit-image

---

## Cách chạy

1. Upload dataset lên Kaggle
2. Chạy lần lượt từ `00a` → `08`
3. Mỗi file đọc output từ file trước làm input
4. Kết quả cuối tại `evaluation/metrics_summary.json`