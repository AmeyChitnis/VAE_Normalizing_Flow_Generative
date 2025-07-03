# VAE & Normalizing Flow for Fashion‑MNIST 🎨

This repository demonstrates two foundational likelihood-based generative models implemented in PyTorch:

- **Variational Autoencoder (VAE)** — encoder/decoder architecture with learned latent distribution  
- **Normalizing Flow (NF)** — invertible mapping modeling exact data likelihood using affine coupling layers

---


## ⚙️ Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/AmeyChitnis/VAE_Normalizing_Flow_Generative.git
   cd VAE_Normalizing_Flow_Generative
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install torch torchvision matplotlib
   ```

---

## 🚀 Usage

### 🧪 Interactive (Notebook)
Open and run:
```
Generative Models.ipynb
```
This notebook uses the `models/` folder and walks through training, generating, and comparing both models.

### 🖥️ Scripts

#### VAE:
```bash
python scripts/train_vae.py
```


#### Normalizing Flow:
```bash
python scripts/train_flow.py
```


---

## 🖼️ Sample Comparison

| FashionMNIST Samples                 | VAE Generated                         | NF Generated                          | VAE + NF Generated                             |
|--------------------------------------|---------------------------------------|---------------------------------------|------------------------------------------------|
| ![](images/fashionmnist_samples.png) | ![](images/vae_generated_samples.png) | ![](images/nf__generated_samples.png) | ![](images/vae_and_flow_generated_samples.png) |

- **VAE**: captures global structure well, but images are smoother and blurrier.
- **NF**: inconsistent and noisy due to latent mismatch or unbalanced flow design.
- **VAE + NF (after fix)**: still somewhat blurry like VAE, requires tuning (more flows, normalization, etc.)

---

## 💡 Insights

- **VAE**: Optimizes ELBO (approximate likelihood) — tends to average input patches  
- **NF**: Learns invertible transforms, enabling exact likelihood but requires capacity and normalization  
- **Tips**:
  - Use alternating coupling parity and normalization layers for NF
  - Experiment with learning rates, schedule, and longer training

---


---

## 📝 License & Contact

Licensed under [MIT](LICENSE).  
Feel free to open issues or reach out if you'd like to collaborate!
