# PML - Climate Attribution (OLS, BLR, CVAE)

A comparison pipeline for climate extreme event attribution using Ordinary Least Squares (OLS), Bayesian Linear Regression (BLR), and Conditional Variational Autoencoders (CVAE).

---

## 1. Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .PML_env
source .PML_env/bin/activate

# Install dependencies
pip install -r requirement.txt
```

---

## 2. Download Raw Datasets

Download the required large raw datasets (`tg.nc`, `pp.nc`, `anomalies.nc`) into the `dataset/` folder using `gdown`:

```bash
mkdir -p dataset
gdown "https://drive.google.com/uc?id=1K7PIv-1ODnHIUoEzhO9bQk6BjTVaBz_1" -O dataset/tg.nc
gdown "https://drive.google.com/uc?id=1uaqjBF3lvKlif-paDpJVUCh0HifUrrDL" -O dataset/pp.nc
gdown "https://drive.google.com/uc?id=1ZcQCutrMGOtRm2G89xmPSmJOw73k2h_Z" -O dataset/anomalies.nc
```

---

## 3. Pipeline Steps

For each step, you can either **run the code** or **download the precomputed file**.

### Step 1: Data Preparation (FVG Region)
* **Run from scratch**:
  ```bash
  python3 ols/prepare_data.py 12 47 14 45 dataset/vae_dataset.nc
  ```
* **Or download preprocessed dataset**:
  ```bash
  gdown "https://drive.google.com/uc?id=1TymZu4Gat8os0n7fsE4g092gt7zIkxdv" -O dataset/vae_dataset.nc
  ```
---

### Step 2: Bayesian Linear Regression (BLR)
* **Run training**:
  ```bash
  python3 blr/train.py
  ```
* **Or download precomputed posterior**:
  ```bash
  gdown "https://drive.google.com/uc?id=16p3GzeNkpU5HpeUJdRqjBTGLUd3K_xFX" -O dataset/blr_posterior.nc
  ```
---

### Step 3: Train CVAE
* **Run training**:
  ```bash
  python3 cvae/train_cvae.py
  ```
* **Or download trained model weights**:
  ```bash
  gdown "https://drive.google.com/uc?id=1PGF3s-tsG555_dx2EDDelX9ddRsnKZj7" -O cvae/cvae_model.pt
  ```
---

### Step 4: Run Evaluation & Generate Plots
* **Run full comparison** (generates all figures in `graphs/`):
  ```bash
  python3 main.py
  ```
* **Or download the generated plots directly**:
  ```bash
  gdown "https://drive.google.com/uc?id=1MDYpjjB5Vb9monnLaIInwrNJvIxQVxDK" -O graphs.zip
  unzip graphs.zip -d .
  ```
