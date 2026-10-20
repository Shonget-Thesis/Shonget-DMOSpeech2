# Held-Out Test Split Verification Report

**Certified Threshold ($\hat{\lambda}$):** `0.5`  
**Test Sample Size ($n$):** 200 independent speakers / utterances  
**Reference-Only Fallback:** NO  

---

## 1. Realized Failure Rates & Statistical Guarantees

| Metric | Failures / Total | Point Estimate | Exact 95% Clopper-Pearson CI |
|---|---|---|---|
| **Robust Added Failure Rate** | 5 / 200 | 2.50% | [0.82%, 5.74%] |
| `clean` Condition Failure Rate | 5 / 200 | 2.50% | — |
| `noisy15` Condition Failure Rate | 0 / 200 | 0.00% | — |

---

## 2. Computational Efficiency

| Reference NFE (Step 8) | Adaptive Policy Mean NFE | NFE Savings (%) |
|---|---|---|
| 21.00 | 18.15 | **13.6%** |

---

## 3. Switching Distribution

| Candidate Step | Solver Steps | Forward NFE | Selection Frequency (%) |
|---|---|---|---|
| Step 4 | 8 | 13 | 9.8% |
| Step 5 | 9 | 15 | 24.0% |
| Step 6 | 10 | 17 | 15.8% |
| Step 8 | 12 | 21 | 50.5% |