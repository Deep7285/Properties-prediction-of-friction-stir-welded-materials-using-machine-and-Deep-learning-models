# Physics-Informed Surrogate Modelling for Dissimilar FSW

### Process–Structure–Property Design Space via Gaussian Process Regression

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-GPR-f89939?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Active%20Research-blueviolet)](.)
[![Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

> **Note:** This notebook presents the methodology and analytical framework for an ongoing research project. All numerical demonstrations use representative FSW trial data to match the physical characteristics of the problem.

---

## 📌 Overview

This project develops a **physics-informed, uncertainty-aware GPR surrogate model** for dissimilar material Friction Stir Welding (FSW). Rather than treating the process as a black box, the work encodes established physical mechanisms directly into the modelling pipeline.

The central hypothesis is the **Process → Structure → Property (PSP) linkage**:

```
Process Parameters          →    Microstructure         →    Mechanical Property
(ω, v, plunge depth)             IMC layer thickness          Joint shear strength
     ↕                                  ↕                               ↕
Heat input: Q ∝ ω²/v          Arrhenius diffusion           UTS (MPa)
```

The analysis demonstrates that Bayesian Optimisation, had it been applied from the start, could have identified the optimal process window in significantly fewer experimental trials than a traditional trial-and-error approach.

---

## 🧠 The Physics Behind It

### Heat Input as the Governing Variable

The FSW thermal model (Arbegast & Hartley, 1998) relates peak temperature to process parameters:

$$\frac{T}{T_m} = C \left(\frac{\omega^2}{v \cdot 10^4}\right)^{\alpha}$$

where $T_m$ is the melting point of the softer material (Al), and empirical constants $C \in [0.65, 0.75]$, $\alpha \in [0.04, 0.06]$.

### Why the Log Transform Is Not Just a Trick

Intermetallic compound (IMC) growth at the Al–Steel interface follows Arrhenius diffusion kinetics:

$$\delta_{\text{IMC}} \propto D_0 \exp\left(\frac{-E_a}{RT}\right) \cdot \sqrt{t}$$

Since temperature scales with $\ln(\omega^2/v)$, IMC thickness — and by extension joint strength — is approximately **linear in log-space**. This physical motivation is confirmed empirically: `LogHeatInput` consistently shows stronger Pearson correlation with UTS than raw `HeatInput` or individual parameters alone.

---

## 📂 Repository Structure

```
.
├── FSW_GPR_DesignSpace.ipynb    # Main analysis notebook
├── README.md
└── figures/                     # Auto-generated output figures
    ├── 01_EDA.png
    ├── 02_LOOCV_analysis.png
    ├── 03_DesignSpace.png
    ├── 04_BayesianOptimisation.png
    └── 05_PSP_Chain.png
```

---

## 🗂️ Notebook Structure

| # | Section | Description |
|---|---------|-------------|
| 1 | **Background & Motivation** | PSP linkage and research rationale |
| 2 | **Physical Properties Correlation** | Arbegast–Hartley model, Arrhenius kinetics |
| 3 | **Data Loading** | Plug-in for real `.xlsx` data or representative synthetic dataset |
| 4 | **Exploratory Data Analysis** | Correlation maps, design space coverage, heat input distribution |
| 5 | **Physics-Informed Feature Engineering** | Log heat input, RPM/traverse ratio — each with physical justification |
| 6 | **GPR Modelling** | Matérn ν=2.5 kernel, WhiteKernel noise, marginal likelihood optimisation |
| 7 | **LOOCV Validation** | Leave-One-Out CV suited for small-n datasets; RMSE vs noise floor |
| 8 | **Design Space Characterisation** | Probabilistic map: $P(\text{UTS} \geq \tau^*)$ over the full parameter grid |
| 9 | **Bayesian Optimisation** | Expected Improvement acquisition; convergence vs trial-and-error |
| 10 | **PSP Chain Validation** | Physical consistency check against IMC, temperature, force, hardness, EBSD |
| 11 | **Discussion & Next Steps** | Limitations, extensions, broader applicability |

---

## 🔬 Modelling Approach

### Gaussian Process Regression

GPR was selected over Response Surface Methodology (RSM) for three reasons:

- **Non-parametric** — no fixed polynomial form assumed; the response surface is learned from data
- **Uncertainty-aware** — every prediction comes with a calibrated confidence interval, essential when $n < 50$
- **Physics-compatible kernel** — the Matérn $\nu = 2.5$ kernel produces twice-differentiable functions, appropriate for physical processes that are smooth but not infinitely so

The kernel is defined as:

$$k(\mathbf{x}, \mathbf{x}') = \sigma^2 \cdot \left(1 + \frac{\sqrt{5}r}{l} + \frac{5r^2}{3l^2}\right)\exp\left(-\frac{\sqrt{5}r}{l}\right) + \sigma_n^2 \,\delta(\mathbf{x}, \mathbf{x}')$$

where $r$ is the Euclidean distance between input points, $l$ is the learned length scale, $\sigma^2$ the signal variance, and $\sigma_n^2$ the noise variance absorbed by the `WhiteKernel`.

### Design Space Definition

Following ICH Q8 pharmaceutical analogy, the probabilistic design space is formally defined as:

$$\mathcal{DS}_{p^*} = \left\{(\omega, v, d) : P\!\left(\tau(\omega, v, d) \geq \tau^*\right) \geq p*\right\}$$

where $\tau^*$ is the target shear strength and the probability is evaluated from the GPR predictive distribution $\mathcal{N}(\mu(\mathbf{x}),\, \sigma^2(\mathbf{x}))$.

### Bayesian Optimisation

The Expected Improvement (EI) acquisition function drives the sequential experiment selection:

$$\text{EI}(\mathbf{x}) = (\mu(\mathbf{x}) - y^* - \xi)\,\Phi(Z) + \sigma(\mathbf{x})\,\phi(Z), \quad Z = \frac{\mu(\mathbf{x}) - y^* - \xi}{\sigma(\mathbf{x})}$$

---

## ⚙️ Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy openpyxl
```

### Running on Google Colab

The notebook is designed for Colab. Just open it and run all cells — it will mount your Google Drive if you supply your own data, or run on the built-in representative dataset otherwise.

### Using Your Own Data

In **Section 3**, set the flag and file path:

```python
Exp_data = True
FILE_PATH = 'your_data.xlsx'
```

Your Excel file should contain columns for rotation speed (RPM), traverse speed (mm/min), plunge depth (mm), and measured shear/UTS strength (MPa). Column names are mapped at the top of that cell — adjust them to match your file headers.

### Adjusting the Design Space Target

In **Section 8**, tune the threshold to your application:

```python
UTS_THRESHOLD = 45.0  # MPa — change to your minimum acceptable strength
CONFIDENCE    = 0.90  # 90% probability level for the design space boundary
PLUNGE_FIXED  = 3.0   # mm — fix plunge depth for the 2D parameter map
```

---

## 📊 Output Figures

| Figure | Description |
|--------|-------------|
| `01_EDA.png` | Six-panel EDA: log(heat input) vs UTS, RPM–traverse interaction, strength distribution, design space coverage, heat input per trial, plunge depth effect |
| `02_LOOCV_analysis.png` | Predicted vs true with 95% CI, residuals vs heat input, per-trial uncertainty |
| `03_DesignSpace.png` | Mean UTS surface, prediction uncertainty map, probabilistic design space with 80/90/95% contours |
| `04_BayesianOptimisation.png` | EI acquisition per BO step, optimisation convergence vs true optimum |
| `05_PSP_Chain.png` | Process → Structure (heat input vs IMC), Structure → Property (IMC vs UTS), GPR uncertainty vs IMC |

---

## 📐 Feature Engineering Summary

| Feature | Formula | Physical Basis |
|---------|---------|----------------|
| Heat Input Proxy | $Q = \omega^2 / v$ | Arbegast & Hartley (1998) FSW thermal model |
| **Log Heat Input** | $\ln(\omega^2 / v)$ | Linearises Arrhenius IMC growth kinetics — primary model input |
| RPM/Traverse Ratio | $\omega / v$ | Simplified heat proxy; easier to interpret for process engineers |

---

## 📖 References

- Arbegast, W.J. & Hartley, P.J. (1998). *FSW Technology*. TMS.
- Rasmussen, C.E. & Williams, C.K.I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Arlot, S. & Celisse, A. (2010). A survey of cross-validation procedures for model selection. *Statistics Surveys*, 4, 40–79.
- Shahriari, B. et al. (2016). Taking the human out of the loop: A review of Bayesian optimization. *Proceedings of the IEEE*, 104(1), 148–175.
- Shi, S. et al. (2023). Gaussian process regression for materials property prediction. *npj Computational Materials*.

---

## 🤝 Contributing

Issues and suggestions are welcome — particularly around kernel selection, alternative acquisition functions for BO, or extension to 3D design space visualisation with plunge depth as a free variable.

---

<p align="center">
  Made with 🔥 and a lot of thermocouple data
</p>
