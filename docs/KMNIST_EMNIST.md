# KMNIST & EMNIST (grayscale — tùy chọn)

> **Khuyến nghị:** Nếu mục tiêu là **đa dạng ảnh màu**, dùng **[CIFAR-10 + SVHN](COLOR_DATASETS.md)** thay vì thêm grayscale.

Hai dataset grayscale bổ sung (cùng 28×28):

| Dataset | CLI key | Classes |
|---------|---------|---------|
| KMNIST | `kmnist` | 10 |
| EMNIST Letters | `emnist_letters` | 26 |

Chạy giống Fashion-MNIST:

```bash
bash run_dataset_pipeline_parallel.sh kmnist
bash run_dataset_pipeline_parallel.sh emnist_letters
```

EMNIST: labels đã remap 1–26 → 0–25; dùng `--skip-advanced-viz`.
