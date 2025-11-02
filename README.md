# 🌾 AquaCrop Global Sensitivity Analysis (GSA) Framework
**Project:** Parameter Sensitivity and Adaptability Assessment of the AquaCrop Model in Northeast China  
**Author:** Yage Song  
**Version:** 1.4  
**Date:** November 2025  

---

## 1. Overview
This repository provides a complete workflow for conducting **global sensitivity analysis (GSA)** and **adaptability assessment** of the FAO AquaCrop model under different climatic year types in Northeast China.  
The framework integrates:

- **Year-type classification** based on long-term meteorological data.  
- **Morris screening** for preliminary identification of influential parameters.  
- **eFAST sensitivity analysis** for quantifying first-order and total effects.  
- **Daily time-resolved eFAST** for canopy cover and biomass to reveal the dynamic evolution of parameter influence.  

All modules are designed to be reproducible, transparent, and fully automated.

---

## 2. Repository Structure

```
📂 AquaCrop-GSA/
│
├── year_type_classification.py      # Year-type classification based on precipitation quantiles
├── Morris_Yield.py                  # Parameter screening using the Morris method
├── eFAST_Yield.py                   # eFAST sensitivity analysis on simulated yield
├── eFAST_CC_daily.py                # Daily eFAST analysis for canopy cover
├── eFAST_Biomass_daily.py           # Daily eFAST analysis for biomass
├── run_all_GSA.py                   # Main controller script for the full pipeline
└── results/
    ├── logs/                        # Runtime logs
    ├── markers/                     # Step completion flags
    └── outputs/                     # CSV, PNG, and report files
```

---

## 3. Workflow Description

### **Step 1 — Year-type Classification**
Classifies representative dry, normal, and wet years from long-term meteorological data and produces climatological summaries.

### **Step 2 — Morris Screening**
Conducts global parameter screening on crop yield to identify influential variables and reduce parameter dimensionality.

### **Step 3 — eFAST Sensitivity Analysis (Yield)**
Performs extended Fourier amplitude sensitivity testing on yield outputs across different climatic year types, calculating both first-order (S₁) and total (ST) indices.

### **Step 4 — Daily eFAST (Canopy Cover & Biomass)**
Performs time-resolved eFAST analysis over a 130-day growing window, quantifying the temporal evolution of parameter influence during canopy development and biomass accumulation.

### **Automated Workflow Execution**
The entire workflow can be managed automatically using `run_all_GSA.py`.  
This script coordinates each analysis stage — including year-type classification, Morris screening, and eFAST computations — and handles:
- Sequential execution of modules  
- Logging and progress tracking  
- Step resumption after interruption  

Researchers can flexibly run the full pipeline or selectively execute individual components.

---

## 4. Input and Output

### **Inputs**
- Long-term meteorological data (daily precipitation, ET₀, Tmax, Tmin).  
- Parameter boundaries defined according to AquaCrop guidelines.  
- Representative years for dry, normal, and wet conditions.

### **Outputs**
| Category | Description | File Format |
|-----------|--------------|--------------|
| Year-type classification | Annual classification summary and climatological visualization | `.csv`, `.png` |
| Morris analysis | Screening indices and ranking plots | `.csv`, `.png` |
| eFAST (Yield) | Sensitivity indices by climate year | `.csv`, `.png`, `.txt` |
| Daily eFAST (CC & Biomass) | Time-resolved S₁ and ST indices | `.csv`, `.png`, `.txt` |
| Summary report | Consolidated summary of key parameters | `.txt` |

All results are stored in the `results/outputs/` directory.

---

## 5. Execution

### **Run the Entire Pipeline**
```bash
python run_all_GSA.py
```

### **Run Specific Steps**
```bash
python run_all_GSA.py --only morris
python run_all_GSA.py --no-efast-cc
```

### **Force Re-run All Analyses**
```bash
python run_all_GSA.py --force
```

Each script may also be executed independently.

---

## 6. Software Environment

### **Base Environment**
- **Python ≥ 3.9**  
- **Recommended setup:**
  ```bash
  python -m venv venv
  source venv/bin/activate  # (Linux/Mac)
  venv\Scripts\activate   # (Windows)
  pip install -U pip wheel
  ```

### **Dependencies**
```bash
pip install aquacrop SALib numpy pandas matplotlib seaborn
```

### **References**
- **AquaCrop-OSPy (PyPI):** https://pypi.org/project/aquacrop/  
- **AquaCrop-OSPy Source (GitHub):** https://github.com/aquacropos/aquacrop  
- **FAO AquaCrop Official Model:** https://www.fao.org/aquacrop/zh/  

> The AquaCrop model is developed and maintained by the Food and Agriculture Organization (FAO) of the United Nations.  
> The Python interface (`aquacrop` on PyPI) provides an open-source implementation of the FAO model logic for research and educational use.

---

## 7. Performance and Automation

- Fully automated workflow with built-in step markers and log tracking.  
- Daily eFAST runs each parameter set once, then computes daily sensitivity indices from stored outputs.  
- Supports parallel computing and high-performance cluster deployment.

---

## 8. Contact

- **Email:** syg@stu.syau.edu.cn  
- **Facebook Group:** [https://www.facebook.com/share/g/17hrqMVN14/](https://www.facebook.com/share/g/17hrqMVN14/)

---

## 9. License

This project is released for academic and research use.  
Please acknowledge the author and reference the FAO AquaCrop framework when extending or redistributing.

---

## 10. Sensitivity Analysis Parameter List

The following parameters were included in the global sensitivity analysis.  
They are organized by functional category according to the **FAO AquaCrop** documentation and **AquaCrop-OSPy** parameter structure.

### 🟤 Soil Parameters
- CN — Curve Number for runoff estimation  
- th_s — Saturation water content  
- th_fc — Field capacity  
- th_wp — Permanent wilting point  
- Ksat — Saturated hydraulic conductivity  
- REW — Readily evaporable water

### 🟢 Canopy and Phenology
- Emergence — Thermal time to emergence  
- HIstart — Start of harvest index build-up  
- Senescence — Onset of canopy senescence  
- Maturity — Physiological maturity  
- YldForm — Grain filling duration  
- Flowering — Flowering duration  
- SeedSize — Average seed size  
- PlantPop — Plant population density  
- CGC — Canopy growth coefficient  
- CCx — Maximum canopy cover  
- CDC — Canopy decline coefficient  
- Tbase — Base temperature for growth  
- Tupp — Upper temperature threshold

### 🔵 Root Growth
- MaxRooting — Maximum rooting depth (GDD or days)  
- Zmax — Maximum soil depth explored by roots (m)  
- Zmin — Minimum root depth (m)  
- fshape_r — Shape factor for root expansion  
- SxTopQ — Fraction of roots in top soil  
- SxBotQ — Fraction of roots in bottom soil

### 🟣 Transpiration and Water Uptake
- Kcb — Basal crop coefficient  
- fage — Ageing reduction factor  
- fsink — Fraction of assimilate partitioned to sink organs

### 🟠 Production and Yield Formation
- WP* — Water productivity normalized for ET₀  
- HI0 — Reference harvest index  
- exc — Excess biomass reduction factor  
- WPy — Water productivity under non-optimal conditions

### 🔴 Water and Temperature Stresses
- Aer — Aeration stress threshold  
- fshape_w4 — Shape factor for water stress curve (stage 4)  
- p_up1, p_lo1, fshape_w1 — Water stress parameters for canopy expansion  
- p_up2, fshape_w2 — Water stress parameters for stomatal closure  
- p_up3, fshape_w3 — Water stress parameters for canopy senescence  
- p_up4 — Water stress parameter for yield formation  
- dHI_pre, a_HI, b_HI, dHI0 — Parameters controlling stress impact on harvest index  
- Tmin_up, Tmax_up, Tmin_lo, Tmax_lo — Temperature thresholds for growth stress  
- GDD_up — Upper limit of daily growing degree days

---

*Parameter groupings and terminology follow the FAO AquaCrop model structure (Steduto et al., 2009; Raes et al., 2023) and the open-source Python implementation [AquaCrop-OSPy](https://aquacropos.github.io/aquacrop/notebooks/AquaCrop_OSPy_Notebook_1/).*
