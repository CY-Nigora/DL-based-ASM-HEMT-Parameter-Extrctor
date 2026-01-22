# DL-based ASM-HEMT Parameter Extractor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

A deep learning-based framework for automatic parameter extraction of GaN ASM-HEMT physical models. This project utilizes neural networks to map electrical characteristics ($I-V$) to physical model parameters, overcoming the limitations of non-differentiable simulation environments, in meanwhile solving the key-feature of input sensitivity and non-uniqueness of ASM-HEMT parameter extraction.

---

## 📖 Table of Contents
- [DL-based ASM-HEMT Parameter Extractor](#dl-based-asm-hemt-parameter-extractor)
  - [📖 Table of Contents](#-table-of-contents)
  - [🧠 Background \& Methodology](#-background--methodology)
    - [Comparison of Different Methods](#comparison-of-different-methods)
  - [🏗 System Architecture](#-system-architecture)
  - [📂 Project Structure](#-project-structure)
  - [🛠 Installation](#-installation)
  - [🚀 Workflow \& Training](#-workflow--training)
  - [🛠 Future Work](#-future-work)
  - [⚖️ License](#️-license)
  - [👥 Contact](#-contact)

---

## 🧠 Background & Methodology

In parameter extraction for ASM-HEMT models, the simulation environment (e.g., ADS) often acts as a **black box** because it doesn't support differential operations for backpropagation. 

### Comparison of Different Methods

| Method | Workflow | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Online Training (EA)** | CMA-ES / Genetic Algorithm | Easy to implement; no gradient needed. | Slow, no parallel support, prone to local minima. |
| **Reinforcement Learning** | Agent-Environment Interaction | Adaptive; no gradient needed. | High computational cost for parallel training. |
| **DL(Our Choice, CNN + CVAE)** | Large-scale Monte Carlo Data | **Fast inference; high accuracy via gradient descent, perfectly suits non-uniqueness and input-sensitivity.** | Requires large amounts of pre-simulated data. |

---

## 🏗 System Architecture

Basic workflow:
1. **Data Generation**: log-uniform based data generation connecting with ADS external Python Interface. Support multi-processing, process-monitor, matrix simulation.
2. **Data Pre-processing**: add specific noise in genrated smooth data, soft filter applied to remove outlier (non-monotonic, max/min constrains ...)
3. **Model Training**: Proxy model + Main model trainnig, including pure MLP model as baseline and CNN + CVAE based DL model, and advanced 2-stage fine-tuning strategy, also advanced infenrence strategy (BoK).

showed in figure:
<p align = "center">    
<img src="./img/work_flow_2channel.png" alt='ss' width=80%/>
</p> 

---
## 📂 Project Structure

```bash
├── NN_training/           # Core Neural Network architectures (CNN, CVAE)
├── data_gen/              # Scripts for generating data via ADS/Simulation interface
├── data_gen_pro/          # Advanced data generation with physical constraints
├── data_pre_processing/   # Data cleaning, normalization, and H5 packaging
├── data_viewer/           # GUI/Scripts for Model performance diagnostic tools
└── README.md              # Project documentation
```
Best Model location，navigate to path:
1. CNN + CVAE based unidirectional GaN HEMTs : `NN_training\model\cvae_Unidi_14param_2channel\version_2_4`
2. Baseline of unidirectional GaN HEMTs : `NN_training\model\Unidi_pureMLP_2channel_baseline`
3. CNN + CVAE based bi-directional GaN HEMTs : `NN_training\model\cvae_Bidi_11param_2channel\version_1_3`
4. Baseline of bidirectional GaN HEMTs : `NN_training\model\Bidi_pureMLP_2channel_baseline`

---
## 🛠 Installation

1. Clone the repository:
```bash
git clone https://github.com/CY-Nigora/DL-based-ASM-HEMT-Parameter-Extrctor.git
```
2. Install Dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Workflow & Training
**Model: CNN + CVAE based Uni/Bi-directional GaN HEMTs**
The model estimates 11/14 key parameters simultaneously.
Code files locate in path `NN_training\code\training\cvae_Bidi_CNN\main.py` or `NN_training\code\training\cvae_Unidi_CNN\main.py`. The path `pure_MLP_Bidi` and `pure_MLP_Bidi` are pure MLP models working as baseline. Specific paramters in following commands only work as example.

**Step 1: Proxy Training**
To train the proxy model (mapping parameters to curves):

```Bash
python code_file_path --data h5_dataset_file_path ^
  --train-proxy-only  --outdir output_path --proxy-hidden 512,512,512,512 --proxy-batch-size 1024^
  --proxy-lr 2.0e-4 --proxy-wd 5e-5 --proxy-patience 25 --proxy-epochs 180 
```

**Step 2: Parameter Extraction Training**
To train pure MLP based extractor (mapping curves back to parameters):

```Bash
python code_file_path --data h5_dataset_file_path ^
  --outdir output_path ^
  --hidden 1280,640,320 --batch-size 256 --lr 2.0e-4 ^
  --dropout 0.1 --weight-decay 1e-4 --max-epochs 300 --onecycle-epochs 300 --patience 100
```

To train CNN + CVAE based extractor (mapping curves back to parameters):

For 1-stage training:
```bash
python code_file_path --data h5_dataset_file_path ^
  --outdir output_path ^
  --proxy-run proxy_path ^
  --meas-h5 reference_measurement_small_h5_dataset_path ^
  --hidden 1280,640,320 --batch-size 256 --lr 2.5e-4 ^
  --feat-dim 256 ^
  --dropout 0.1 --weight-decay 1e-4 --max-epochs 300 --onecycle-epochs 300 --patience 100 ^
  --lambda-cyc-sim 10.0 --lambda-cyc-meas 0.1 --weight-iv 5.0 --weight-gm 1.0 --cyc-warmup-epochs 150 ^
  --sup-weight 1.0 --prior-l2 1.0e-2 --prior-bound 1e-2 --prior-bound-margin 0.05 ^
  --es-metric val_cyc_meas --es-min-delta 1e-5 ^
  --trust-alpha 1.0 --trust-alpha-meas 0.1 --trust-tau 1.0 ^
  --cyc-meas-knn-weight --cyc-meas-knn-gamma 0.8 ^
  --latent-dim 32 --kl-beta 0.5 ^
  --aug-prob 0.5 --aug-noise-std 0.015 ^
  --best-of-k 0 --cnn-dropout 0.1 ^
  --diag --z-sample-mode mean --phys-loss
```

For 2-stage fine-turning:
```bash
python code_file_path --data h5_dataset_file_path ^
  --outdir output_path ^
  --proxy-run proxy_path ^
  --meas-h5 reference_measurement_small_h5_dataset_path ^
  --resume 1stage_result_named::best_model.pt ^
  --hidden 1280,640,320 --batch-size 256 --lr 1.5e-5 ^
  --feat-dim 256 ^
  --dropout 0.0 --weight-decay 1e-4 --max-epochs 300 --onecycle-epochs 300 --patience 100 ^
  --lambda-cyc-sim 0.1 --lambda-cyc-meas 10.0 --weight-iv 5.0 --weight-gm 1.0 --cyc-warmup-epochs 150 ^
  --sup-weight 1.0 --prior-l2 1.0e-2 --prior-bound 1e-2 --prior-bound-margin 0.05 ^
  --es-metric val_cyc_meas --es-min-delta 1e-5 ^
  --trust-alpha 1.0 --trust-alpha-meas 0.1 --trust-tau 1.0 ^
  --cyc-meas-knn-weight --cyc-meas-knn-gamma 0.8 ^
  --latent-dim 32 --kl-beta 0.05 ^
  --aug-prob 0.0 --aug-noise-std 0.015 ^
  --best-of-k 0 --cnn-dropout 0.0 ^
  --diag --z-sample-mode mean --phys-loss
```

**Inference Command:**

Inference of Proxy model:
```bash
python code_file_path ^
  --infer-proxy-run proxy_path ^
  --proxy-input-h5 h5_dataset_file_path ^
  --proxy-index index_number ^
  --save-xhat-npy xhat.npy
```

Inference of main model:
```bash
python code_file_path ^
--infer-run model_father_folder_path,
--input-h5 inference_data_path,
--index inference_data_index, # if ignored, then inference all data in input data file
--save-csv output_csv_path,
--sample-mode cvae_mode, # `rand` / `mean`
--num-samples 1000 # number of sampling from latent space
--dropout-infer # if used, then use same dropout rate during training, which is saved in model`s local configuration files
```

TTO based Inference of CNN + CVAE model:
```bash
python code_file_path ^
  --cvae-run model_father_folder_path ^
  --proxy-run proxy_path ^
  --meas-h5 inference_h5_data_path ^
  --save-to output_csv_path ^
  --steps 1000 --lr 0.05
```
---

## 🛠 Future Work

1. 14 -> 15 ASM-HEMT, take self-heating effect also into consideration
2. Besides static I-V chracteristic also dynamic C-V chractersitic into consideration : more parameters and higher input dimensions.
3. Correcte an advanced bi-directional GaN HEMTs model, updated based on ASM-HEMT model, considering more parasistic components and coupling impacts. In meanwhile supplement the influence of leakage current in common substrate during both measuremnt and simulation. 
4. Try to simulate and verify unsymmetric rather than symmetric configurations in bi-directional GaN HEMTs.


---

## ⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Contact
Created by CY-Nigora (github).