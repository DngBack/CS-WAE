# Fashion-MNIST — Hướng dẫn chạy CS-WAE

Fashion-MNIST dùng **cùng kiến trúc** với MNIST (28×28 grayscale, 10 classes). Chỉ khác bộ dữ liệu và thư mục output.

---

## 1. Chuẩn bị

```bash
cd /home/bachdx2/CS-WAE
source .venv/bin/activate
```

Lần đầu chạy sẽ tự download Fashion-MNIST vào `./data/FashionMNIST/`.

---

## 2. Chạy full pipeline (khuyến nghị — 2 GPU)

Tương tự MNIST: 3 seeds + ablation + baselines + aggregate (~5–7 giờ).

```bash
chmod +x run_fashion_mnist_pipeline_parallel.sh
nohup bash run_fashion_mnist_pipeline_parallel.sh > logs/fashion_mnist_$(date +%Y%m%d_%H%M%S).log 2>&1 &
tail -f logs/fashion_mnist_*.log
```

**Output:**

```
runs/fashion_mnist/
├── seed_0/, seed_1/, seed_2/
├── aggregated_metrics.json
├── ablation_<timestamp>/
└── baselines/seed_0/
```

---

## 3. Chạy từng bước thủ công

### Main training (1 seed)

```bash
python train_cs_wae.py \
  --dataset fashion_mnist \
  --seed 0 \
  --device cuda:0 \
  --output-dir runs/fashion_mnist/seed_0
```

### Multi-seed (3 seeds, 2 GPU)

```bash
python train_cs_wae.py --dataset fashion_mnist --seed 0 --device cuda:0 --output-dir runs/fashion_mnist/seed_0 &
python train_cs_wae.py --dataset fashion_mnist --seed 1 --device cuda:1 --output-dir runs/fashion_mnist/seed_1 --skip-advanced-viz &
wait
python train_cs_wae.py --dataset fashion_mnist --seed 2 --device cuda:0 --output-dir runs/fashion_mnist/seed_2 --skip-advanced-viz

python aggregate_results.py --runs-dir runs/fashion_mnist
```

### Ablation (song song 2 GPU)

```bash
ABL=runs/fashion_mnist/ablation_manual
mkdir -p "$ABL"

python run_ablation_study.py --dataset fashion_mnist --seed 0 --device cuda:0 \
  --variants baseline no_sup_mmd minimal --results-dir "$ABL" --skip-aggregate &
python run_ablation_study.py --dataset fashion_mnist --seed 0 --device cuda:1 \
  --variants euclidean vmf_prior --results-dir "$ABL" --skip-aggregate &
wait
python run_ablation_study.py --summarize-only --results-dir "$ABL"
```

### Baselines (song song 2 GPU)

```bash
BL=runs/fashion_mnist/baselines/seed_0
mkdir -p "$BL"

python compare_baselines.py --dataset fashion_mnist --seed 0 --device cuda:0 --epochs 50 \
  --output-dir "$BL" --models VAE WAE-MMD --skip-summary &
python compare_baselines.py --dataset fashion_mnist --seed 0 --device cuda:1 --epochs 50 \
  --output-dir "$BL" --models VaDE CS-WAE --skip-summary &
wait
python compare_baselines.py --dataset fashion_mnist --summarize-only --output-dir "$BL"
```

---

## 4. CLI `--dataset` trên mọi script

| Script | Flag |
|--------|------|
| `train_cs_wae.py` | `--dataset fashion_mnist` |
| `compare_baselines.py` | `--dataset fashion_mnist` |
| `run_ablation_study.py` | `--dataset fashion_mnist` |
| `aggregate_results.py` | `--runs-dir runs/fashion_mnist` |

Giá trị hỗ trợ: `mnist`, `fashion_mnist`.

---

## 5. So sánh với MNIST

| | MNIST | Fashion-MNIST |
|---|-------|---------------|
| Input | 1×28×28 | 1×28×28 |
| Classes | 10 | 10 |
| Model / loss | Giống nhau | Giống nhau |
| Hyperparams | `src/config.py` | Cùng config |
| Output | `runs/mnist/` | `runs/fashion_mnist/` |
| FID reference | MNIST test set | Fashion-MNIST test set |

Kỳ vọng: Fashion-MNIST **khó hơn MNIST** (ACC/FID thường kém hơn một chút) — đó là bằng chứng generalization cho paper.

---

## 6. Files đã thêm/sửa

- `src/datasets/fashion_mnist.py` — loader
- `src/datasets/loaders.py` — registry + `build_test_dataset()` cho FID
- `src/metrics/evaluation.py` — FID dùng đúng test set theo dataset
- `run_fashion_mnist_pipeline_parallel.sh` — pipeline 2 GPU
