# Learn-Then-Test (LTT) Risk Calibration Report

**Configuration SHA-256:** `d3644f989f09925ffa1e8fbf502a3dfb5781ed447fc77fcbd1d22985f6830d73`  
**Calibration Unit:** Independent Speakers ($n = 200$)  
**Primary Certification Level:** $\alpha = 0.05$, $\delta = 0.1$  
**Primary Selected Threshold ($\hat{\lambda}$):** `0.5`  
**Secondary Selected Threshold ($\alpha = 0.1$):** `1.2`  

---

## 1. Primary Fixed-Sequence Hypothesis Testing Table ($\alpha = 0.05$, $\delta = 0.10$)

| $\lambda$ | Robust Failures ($k$) | Total Speakers ($n$) | Empirical Risk ($\hat{R}$) | Binomial $p$-value | Decision Status | Mean NFE | Clean Risk | Noisy15 Risk |
|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| 0.20 | 0 | 200 | 0.00% | 3.5053e-05 | **ACCEPTED** | 20.83 | 0.00% | 0.00% |
| 0.25 | 2 | 200 | 1.00% | 2.3363e-03 | **ACCEPTED** | 20.56 | 1.00% | 0.00% |
| 0.30 | 3 | 200 | 1.50% | 9.0484e-03 | **ACCEPTED** | 20.07 | 1.50% | 0.00% |
| 0.35 | 3 | 200 | 1.50% | 9.0484e-03 | **ACCEPTED** | 19.55 | 1.50% | 0.00% |
| 0.40 | 3 | 200 | 1.50% | 9.0484e-03 | **ACCEPTED** | 18.90 | 1.50% | 0.00% |
| 0.45 | 5 | 200 | 2.50% | 6.2342e-02 | **ACCEPTED** | 18.55 | 2.50% | 0.00% |
| 0.50 | 4 | 200 | 2.00% | 2.6447e-02 | **ACCEPTED** | 18.19 | 2.00% | 0.00% |
| 0.55 | 6 | 200 | 3.00% | 1.2374e-01 | **REJECTED_STOP** | 17.82 | 3.00% | 0.00% |
| 0.60 | 3 | 200 | 1.50% | 9.0484e-03 | **UNTESTED_STOPPED** | 17.59 | 1.50% | 0.00% |
| 0.65 | 3 | 200 | 1.50% | 9.0484e-03 | **UNTESTED_STOPPED** | 17.43 | 1.50% | 0.00% |
| 0.70 | 3 | 200 | 1.50% | 9.0484e-03 | **UNTESTED_STOPPED** | 17.27 | 1.50% | 0.00% |
| 0.75 | 3 | 200 | 1.50% | 9.0484e-03 | **UNTESTED_STOPPED** | 17.13 | 1.50% | 0.00% |
| 0.80 | 3 | 200 | 1.50% | 9.0484e-03 | **UNTESTED_STOPPED** | 17.06 | 1.50% | 0.00% |
| 0.85 | 3 | 200 | 1.50% | 9.0484e-03 | **UNTESTED_STOPPED** | 17.04 | 1.50% | 0.00% |
| 0.90 | 4 | 200 | 2.00% | 2.6447e-02 | **UNTESTED_STOPPED** | 17.03 | 2.00% | 0.00% |
| 0.95 | 3 | 200 | 1.50% | 9.0484e-03 | **UNTESTED_STOPPED** | 17.02 | 1.50% | 0.00% |
| 1.00 | 3 | 200 | 1.50% | 9.0484e-03 | **UNTESTED_STOPPED** | 17.02 | 1.50% | 0.00% |
| 1.05 | 3 | 200 | 1.50% | 9.0484e-03 | **UNTESTED_STOPPED** | 17.02 | 1.50% | 0.00% |
| 1.10 | 3 | 200 | 1.50% | 9.0484e-03 | **UNTESTED_STOPPED** | 17.02 | 1.50% | 0.00% |
| 1.15 | 3 | 200 | 1.50% | 9.0484e-03 | **UNTESTED_STOPPED** | 17.02 | 1.50% | 0.00% |
| 1.20 | 3 | 200 | 1.50% | 9.0484e-03 | **UNTESTED_STOPPED** | 17.02 | 1.50% | 0.00% |

---

## 2. Secondary Fixed-Sequence Hypothesis Testing Table ($\alpha = 0.10$, $\delta = 0.10$)

| $\lambda$ | Robust Failures ($k$) | Total Speakers ($n$) | Empirical Risk ($\hat{R}$) | Binomial $p$-value | Decision Status | Mean NFE |
|---:|---:|---:|---:|---:|:---:|---:|
| 0.20 | 0 | 200 | 0.00% | 7.0551e-10 | **ACCEPTED** | 20.83 |
| 0.25 | 2 | 200 | 1.00% | 1.8971e-07 | **ACCEPTED** | 20.56 |
| 0.30 | 3 | 200 | 1.50% | 1.4608e-06 | **ACCEPTED** | 20.07 |
| 0.35 | 3 | 200 | 1.50% | 1.4608e-06 | **ACCEPTED** | 19.55 |
| 0.40 | 3 | 200 | 1.50% | 1.4608e-06 | **ACCEPTED** | 18.90 |
| 0.45 | 5 | 200 | 2.50% | 3.8712e-05 | **ACCEPTED** | 18.55 |
| 0.50 | 4 | 200 | 2.00% | 8.4164e-06 | **ACCEPTED** | 18.19 |
| 0.55 | 6 | 200 | 3.00% | 1.4811e-04 | **ACCEPTED** | 17.82 |
| 0.60 | 3 | 200 | 1.50% | 1.4608e-06 | **ACCEPTED** | 17.59 |
| 0.65 | 3 | 200 | 1.50% | 1.4608e-06 | **ACCEPTED** | 17.43 |
| 0.70 | 3 | 200 | 1.50% | 1.4608e-06 | **ACCEPTED** | 17.27 |
| 0.75 | 3 | 200 | 1.50% | 1.4608e-06 | **ACCEPTED** | 17.13 |
| 0.80 | 3 | 200 | 1.50% | 1.4608e-06 | **ACCEPTED** | 17.06 |
| 0.85 | 3 | 200 | 1.50% | 1.4608e-06 | **ACCEPTED** | 17.04 |
| 0.90 | 4 | 200 | 2.00% | 8.4164e-06 | **ACCEPTED** | 17.03 |
| 0.95 | 3 | 200 | 1.50% | 1.4608e-06 | **ACCEPTED** | 17.02 |
| 1.00 | 3 | 200 | 1.50% | 1.4608e-06 | **ACCEPTED** | 17.02 |
| 1.05 | 3 | 200 | 1.50% | 1.4608e-06 | **ACCEPTED** | 17.02 |
| 1.10 | 3 | 200 | 1.50% | 1.4608e-06 | **ACCEPTED** | 17.02 |
| 1.15 | 3 | 200 | 1.50% | 1.4608e-06 | **ACCEPTED** | 17.02 |
| 1.20 | 3 | 200 | 1.50% | 1.4608e-06 | **ACCEPTED** | 17.02 |

---

## 3. Mathematical Interpretation

- The fixed-sequence testing controls family-wise error rate without Bonferroni conservatism.
- Stopping occurs at the first threshold where the null $H_g: R > \alpha$ cannot be rejected at level $\delta$.
- Selected threshold $\hat{\lambda} = 0.5$ guarantees that the true robust failure risk satisfies $P(R(\pi_{\hat{\lambda}}) > \alpha) \le \delta$ under i.i.d. sampling.