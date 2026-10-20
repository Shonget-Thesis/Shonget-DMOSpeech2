# Repeated-Split Stability Analysis (100 Random 50/50 Partitions)

**Total Pooled Evaluation Cohort:** 400 independent speakers (Calibration + Test)  
**Per-Split Sizes:** 200 calibration speakers / 200 test speakers  
**Number of Random Splits ($B$):** 100  
**Target Risk Parameters:** $\alpha = 0.05$, $\delta = 0.1$  
**Reference-Only Fallback Rate:** 0.0%  

---

## 1. Selected Threshold Distribution ($\hat{\lambda}$)

| Certified Threshold ($\hat{\lambda}$) | Frequency Across 100 Splits (%) |
|---|---:|
| `0.3` | 1.0% |
| `0.4` | 9.0% |
| `0.45` | 7.0% |
| `0.5` | 33.0% |
| `0.55` | 19.0% |
| `0.6` | 2.0% |
| `0.65` | 11.0% |
| `0.85` | 9.0% |
| `1.2` | 9.0% |

---

## 2. Test Split Risk and Compute Stability Across Splits

| Metric | Mean ± Std | 5th Percentile | Median | 95th Percentile |
|---|:---:|:---:|:---:|:---:|
| **Held-Out Test Robust Risk** | 2.89% ± 1.27% | 0.50% | 3.25% | 4.50% |
| **Held-Out Test Mean NFE** | 17.92 ± 0.59 | 17.02 | 18.00 | 18.93 |

---

## 3. Thesis Methodology Verification

- The empirical test risk across 100 random splits consistently respects the nominal PAC bound $\le 5.0\%$, demonstrating that the fixed-sequence LTT calibration procedure is robust to finite-sample sampling fluctuations across independent speaker cohorts.