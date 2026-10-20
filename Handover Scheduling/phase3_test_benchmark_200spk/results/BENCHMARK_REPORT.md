# Chapter 3 Master Benchmark Results (Held-Out Test Split)

| Handover Policy | Mean NFE | NFE 95% CI | Savings vs Ref (%) | WER (%) | SIM vs GT | UTMOS | Robust Failure Rate (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Fixed Step 4** | 13.00 | [13.00, 13.00] | **38.1%** | 25.96% | 0.8163 | 2.5716 | 69.50% |
| **Fixed Step 5** | 15.00 | [15.00, 15.00] | **28.6%** | 12.92% | 0.8177 | 2.5701 | 56.50% |
| **Fixed Step 6** | 17.00 | [17.00, 17.00] | **19.0%** | 12.37% | 0.8209 | 2.5694 | 53.50% |
| **Fixed Step 8** | 21.00 | [21.00, 21.00] | **0.0%** | 48.29% | 0.8254 | 2.5663 | 0.00% |
| **SNR-Only Policy** | 17.04 | [17.00, 17.10] | **18.9%** | 48.54% | 0.8252 | 2.5697 | 5.50% |
| **Proposed Adaptive (λ=0.50)** | 18.15 | [18.05, 18.27] | **13.6%** | 48.49% | 0.8252 | 2.5692 | 2.50% |
| **Proposed Adaptive (No SNR Gate, λ=0.50)** | 14.12 | [14.03, 14.23] | **32.7%** | 25.91% | 0.8164 | 2.5711 | 70.00% |
| **Oracle Policy** | 14.94 | [14.71, 15.17] | **28.9%** | 9.33% | 0.8320 | 2.5706 | 0.00% |

## Statistical Significance Tests against Step-8 Reference (Wilcoxon Signed-Rank with Holm-Bonferroni Correction)

| Policy | NFE Adj. p-value | WER Adj. p-value | SIM Adj. p-value | UTMOS Adj. p-value |
|---|---:|---:|---:|---:|
| **Fixed Step 4** | 3.3043e-88 | 0.0000 | 0.0198 | 0.4462 |
| **Fixed Step 5** | 3.3043e-88 | 1.0000 | 0.0122 | 1.0000 |
| **Fixed Step 6** | 3.3043e-88 | 1.0000 | 0.5267 | 1.0000 |
| **SNR-Only Policy** | 1.1411e-44 | 1.0000 | 1.0000 | 1.0000 |
| **Proposed Adaptive (λ=0.50)** | 1.5635e-35 | 1.0000 | 1.0000 | 1.0000 |
| **Proposed Adaptive (No SNR Gate, λ=0.50)** | 8.0648e-71 | 0.0000 | 0.0227 | 0.1014 |
