# CIFAR-10 & SVHN — Hướng dẫn chạy (ảnh màu)

Hai dataset **RGB 32×32** bổ sung cho MNIST / Fashion-MNIST (grayscale 28×28).

| Dataset | CLI key | Kênh | Kích thước | Classes | Mô tả |
|---------|---------|------|------------|---------|--------|
| **CIFAR-10** | `cifar10` | 3 | 32×32 | 10 | Ảnh tự nhiên (động vật, phương tiện…) |
| **SVHN** | `svhn` | 3 | 32×32 | 10 | Số nhà trên Google Street View (màu) |

CNN tự động dùng **3 input channels** — không cần sửa code khi đổi dataset.

**Backbone (CIFAR-10 / SVHN):** pipeline mặc định dùng **ResNet-18** (`--backbone resnet18`), output lưu tại `runs/cifar10_resnet18/`. CNN nhẹ (`--backbone cnn`) vẫn dùng được cho so sánh.

---

## 1. Tải dữ liệu

```bash
source .venv/bin/activate
python -c "
from src.datasets.loaders import get_loaders, SUPPORTED_DATASETS
print('Supported:', SUPPORTED_DATASETS)
for ds in ('cifar10', 'svhn'):
    tr, te = get_loaders(dataset=ds, seed=0, num_workers=0)
    x, y = next(iter(tr))
    print(f'{ds}: shape={tuple(x.shape)} labels={y[:3].tolist()}')
"
```

Dữ liệu lưu tại `./data/` (torchvision auto-download).

---

## 2. Train nhanh — 1 seed

```bash
# CIFAR-10 với ResNet-18 backbone (~3–5h/seed với 50 epochs + FID)
python train_cs_wae.py --dataset cifar10 --backbone resnet18 --seed 0 --device cuda:0 \
  --output-dir runs/cifar10_resnet18/seed_0 --skip-viz

# CNN nhẹ (baseline cũ)
python train_cs_wae.py --dataset cifar10 --backbone cnn --seed 0 --device cuda:0 \
  --output-dir runs/cifar10/seed_0 --skip-viz

# SVHN (ResNet-18 mặc định)
python train_cs_wae.py --dataset svhn --backbone resnet18 --seed 0 --device cuda:0 \
  --output-dir runs/svhn_resnet18/seed_0 --skip-viz
```

`--skip-viz`: visualization hiện tối ưu cho grayscale; metrics (ACC, FID, …) vẫn đầy đủ.

Smoke test (2 epochs):

```bash
python train_cs_wae.py --dataset cifar10 --seed 0 --device cuda:0 \
  --epochs 2 --skip-viz --output-dir runs/cifar10/debug
```

---

## 3. Pipeline đầy đủ (3 seeds + ablation + baselines)

```bash
chmod +x run_dataset_pipeline_parallel.sh

# CIFAR-10 (ResNet-18 mặc định → runs/cifar10_resnet18/)
nohup bash run_dataset_pipeline_parallel.sh cifar10 \
  > logs/pipeline_cifar10_resnet18_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# CNN nhẹ (giữ kết quả cũ)
BACKBONE=cnn nohup bash run_dataset_pipeline_parallel.sh cifar10 \
  > logs/pipeline_cifar10_cnn_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# SVHN (ResNet-18 mặc định → runs/svhn_resnet18/)
nohup bash run_dataset_pipeline_parallel.sh svhn \
  > logs/pipeline_svhn_resnet18_$(date +%Y%m%d_%H%M%S).log 2>&1 &

tail -f logs/pipeline_cifar10_resnet18_*.log
```

Ước lượng: **~2–4 ngày GPU** mỗi dataset (FID 10k ảnh × nhiều runs).

---

## 4. Output

```
runs/cifar10_resnet18/   # ResNet-18 (mặc định pipeline)
runs/cifar10/            # CNN nhẹ (BACKBONE=cnn)
├── seed_{0,1,2}/metrics.json
├── aggregated_metrics.json
├── ablation_<timestamp>/ablation_results.csv
└── baselines/seed_0/comparison_results.csv
```

Aggregate:

```bash
python aggregate_results.py --runs-dir runs/cifar10_resnet18
```

---

## 5. So sánh portfolio datasets

| | MNIST | Fashion-MNIST | **CIFAR-10** | **SVHN** |
|---|-------|---------------|--------------|----------|
| Màu | ❌ | ❌ | ✅ | ✅ |
| Size | 28² | 28² | 32² | 32² |
| Domain | Chữ số | Quần áo | Object natural | Street digits |

Story paper: *CS-WAE generalizes from grayscale digits/fashion to **RGB natural and street-view** benchmarks.*

---

## 6. Lưu ý

- **KMNIST / EMNIST** (grayscale khác) vẫn có trong code nhưng **không cần** nếu đã có CIFAR + SVHN cho đa dạng.
- CIFAR-10 thường **khó hơn** MNIST (ACC/FID kém hơn là bình thường).
- SVHN train set lớn (~73k) → train lâu hơn Fashion-MNIST.
- Sau khi có kết quả, thêm paths vào `scripts/plot_paper_figures.py` để export bảng/hình.
