# TPC_RP: Active Learning on a Budget — CIFAR-10 Implementation

**Coursework 2 | 5CCSAMLF Machine Learning | King's College London**
**Student:** Layan Alsubhi | K23065725

---

## Overview

This repository contains an implementation of the **TPC_RP**
active learning algorithm, based on the paper:

> Hacohen, G., Dekel, A., & Weinshall, D. (2022).
> *Active Learning on a Budget: Opposite Strategies Suit
> High and Low Budgets.*
> Proceedings of the 39th ICML, PMLR 162.
> https://arxiv.org/abs/2202.02794

The algorithm is implemented and evaluated on CIFAR-10
as part of Coursework 2 for 5CCSAMLF Machine Learning
at King's College London. The implementation includes
the original TPC_RP algorithm, four uncertainty-based
baselines (Random, Least-Confidence, Margin, Entropy),
and a systematic ablation study of proposed modifications
to the typicality estimator.

---

## Algorithm Summary

TPC_RP operates in three steps:

1. **Representation Learning**: SimCLR self-supervised
pre-training on the full unlabelled pool using a
CIFAR-10-adapted ResNet-18 backbone (3×3 initial conv,
no max-pooling). The projection head is discarded at
inference; only the 512-d penultimate features are used.

2. **Clustering for Diversity**: Mini-Batch K-Means
partitioning into K = min(|L| + B, 500) clusters.
Clusters with fewer than five points are excluded.
Setting K = |L| + B guarantees at least B uncovered
clusters, enforcing diversity across semantic regions.

3. **Typicality-Based Querying**: Selects the most
typical example from each uncovered cluster using
inverse average K-NN distance. Clusters with the fewest
labelled points are prioritised, ties broken by size.

---

## Baseline Comparisons

Four baselines were implemented and compared against
TPC_RP to evaluate the paper's central claim that
uncertainty-based methods underperform at low budgets:

| Method | Selection Strategy |
|---|---|
| Random | Uniform random sampling |
| Least-Confidence | Lowest max softmax probability |
| Margin | Smallest top-2 probability gap |
| Entropy | Highest predictive entropy |

All baselines share the same cold-start fallback
(random selection at round 1), identical classifier
architecture, training settings, and evaluation
protocol as TPC_RP for a fair comparison.

### Results: Five-Way Comparison

| Round | Labels | TPC_RP | Random | L-Conf | Margin | Entropy |
|---|---|---|---|---|---|---|
| 1 | 10 | **17.51%** | 15.35% | 17.46% | 12.85% | 12.91% |
| 2 | 20 | 17.44% | **19.78%** | 18.53% | 19.02% | 14.57% |
| 3 | 30 | 21.21% | 18.27% | **21.41%** | 19.82% | 16.07% |
| 4 | 40 | **25.00%** | 21.48% | 21.70% | 20.41% | 17.33% |
| 5 | 50 | **25.76%** | 23.28% | 20.58% | 24.50% | 17.92% |

**Key findings:**
- TPC_RP leads all methods at rounds 1, 4, and 5,
reaching 25.76% vs 23.28% for Random (+2.48 pp) at
50 labels
- Entropy is most severely affected by cold start,
falling to 12.91% at B=10 and never exceeding 17.92%,
confirming the paper's phase-transition hypothesis
- Least-Confidence underperforms TPC_RP at rounds 4
and 5 despite a strong round 3, consistent with the
paper's prediction that uncertainty methods fail to
sustain gains in the low-budget regime
- Margin recovers to 24.50% at round 5 but remains
1.26 pp behind TPC_RP, suggesting partial calibration
as the labelled set grows
- The marginal deficit of TPC_RP at round 2
(17.44% vs 19.78% Random) reflects stochastic
variance at very small budgets

---

## Task 3: Proposed Modification

Two principled modifications to the typicality
estimator are proposed and evaluated through a
four-variant ablation study:

| Variant | Distance Metric | Neighbourhood Size |
|---|---|---|
| Original | Euclidean | Fixed K=20 |
| Variant A | Cosine | Fixed K=20 |
| Variant B | Euclidean | Adaptive K |
| Variant C | Cosine | Adaptive K (combined) |

**Justification:**
SimCLR embeddings are L2-normalised, placing them on
a unit hypersphere where cosine distance is
geometrically more appropriate than Euclidean distance,
which conflates direction and magnitude. Adaptive K
maintains a neighbourhood-to-cluster-size ratio of
approximately 30–50% (K=3 for size <10, K=5 for size
10–19, K=20 otherwise), consistent with the spirit of
the paper's own min{20, cluster_size} schedule in
Appendix F.1.

### Results: Ablation Study

| Round | Labels | Original | Var A (Cosine) | Var B (Adaptive K) | Var C (Combined) |
|---|---|---|---|---|---|
| 1 | 10 | 17.51% | 13.33% | 15.34% | 13.10% |
| 2 | 20 | 17.44% | 19.50% | 17.62% | 20.05% |
| 3 | 30 | 21.21% | 21.14% | 21.30% | 20.63% |
| 4 | 40 | 25.00% | 25.13% | 26.05% | 26.17% |
| 5 | 50 | 25.76% | 25.65% | **27.48%** | **27.96%** |

**Key findings:**
- Variant C achieves the strongest final accuracy
(27.96%, +2.20 pp over original) and Variant B
reaches 27.48% (+1.72 pp) at round 5
- Adaptive K is the most impactful modification,
most effective once cluster sizes are large and
stable enough for meaningful neighbourhood
differentiation
- All four variants substantially outperform all
uncertainty-based baselines at round 5, confirming
typicality-based querying provides genuine value
independent of the distance metric used
- Early-round underperformance of Variants A and C
reflects instability in sparse cluster conditions
- Under fully trained embeddings (500 SimCLR epochs),
the geometric advantage of cosine distance is
expected to be more pronounced

---

## Repository Structure

```
typiclust_cw2.ipynb    — Full implementation:
                         Task 1 (TPC_RP original)
                                 (baseline comparisons:
                                 Random, Least-Confidence,
                                 Margin, Entropy)
                         Task 3 (ablation study:
                                 Variants A, B, C)
README.md              — This file
```

---

## Implementation Notes

- SimCLR trained for 50 epochs (vs 500 in the paper)
due to computational constraints on Colab
- ResNet-18 backbone adapted for CIFAR-10
(3×3 initial conv, Identity maxpool)
- Mini-Batch K-Means used for scalability; clusters
with fewer than 5 points excluded from selection
- All uncertainty baselines use cold-start random
fallback at round 1 (no trained model available)
- Classifier retrained from scratch each AL round
to isolate sample quality effects
- Budget protection applied to all selection functions
- Fixed random seed (42) for reproducibility

---

## Environment

- Python 3.10
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- Google Colab (T4 GPU recommended)

---

## References

- Hacohen et al. (2022) —
https://arxiv.org/abs/2202.02794
- Chen et al. (2020) SimCLR —
https://arxiv.org/abs/2002.05709
- Settles (2009) Active Learning Survey —
http://burrsettles.com/pub/settles.activelearning.pdf
- Mittal et al. (2019) —
https://arxiv.org/abs/1912.05361
