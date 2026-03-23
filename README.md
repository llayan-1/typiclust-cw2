# TPC_RP: Active Learning on a Budget — CIFAR-10 Implementation

**Coursework 2 | 5CCSAMLF Machine Learning | King's College London**
**Student:** Layan Alsubhi | K23065725

---

## Overview

This repository contains an implementation of the **TPC_RP** (Typical Clustering with Representation and Partition) active learning algorithm, based on the paper:

> Hacohen, G., Dekel, A., & Weinshall, D. (2022).
> *Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets.*
> Proceedings of the 39th ICML, PMLR 162.
> https://arxiv.org/abs/2202.02794

The algorithm is implemented on CIFAR-10 as part of Coursework 2 for 5CCSAMLF Machine Learning at King's College London.

---

## Algorithm Summary

TPC_RP operates in three steps:

1. **Representation Learning** — SimCLR self-supervised pre-training on the full unlabeled pool using a CIFAR-10 adapted ResNet-18 backbone
2. **Clustering for Diversity** — K-Means / Mini-Batch K-Means partitioning of the embedding space
3. **Typicality-Based Querying** — Selects the most typical (highest density) example from each uncovered cluster using K-NN distance

---

## Task 3: Proposed Modification

We propose a refined typicality estimator evaluated through an ablation study with four variants:

| Variant | Distance Metric | Neighbourhood Size |
|---|---|---|
| Original | Euclidean | Fixed K=20 |
| Variant A | Cosine | Fixed K=20 |
| Variant B | Euclidean | Adaptive K |
| Variant C | Cosine | Adaptive K |

**Justification:** SimCLR embeddings are L2-normalised, placing them on a unit hypersphere where cosine distance is geometrically more appropriate. Adaptive K produces more locally sensitive density estimates in small clusters, extending the paper's own use of min{20, cluster_size}.

---

## Results

### Task 1: Original TPC_RP

| Round | Labelled Samples | Test Accuracy |
|---|---|---|
| 1 | 10 | 11.84% |
| 2 | 20 | 17.98% |
| 3 | 30 | 19.17% |
| 4 | 40 | 22.05% |
| 5 | 50 | 22.69% |

### Task 3: Ablation Study

| Round | Original | Variant A (Cosine, K=20) | Variant B (Euclidean, Adaptive K) | Variant C (Cosine + Adaptive K) |
|---|---|---|---|---|
| 1 (10 labels) | 11.84% | 15.91% | 16.20% | 12.46% |
| 2 (20 labels) | 17.98% | 16.85% | 17.53% | 19.52% |
| 3 (30 labels) | 19.17% | 18.98% | 19.26% | 20.15% |
| 4 (40 labels) | 22.05% | 23.06% | 21.23% | 22.13% |
| 5 (50 labels) | 22.69% | 24.44% | 26.63% | 24.81% |

**Key findings:**
- Variant A (cosine distance, fixed K=20) provides the most consistent improvement over the original across rounds, finishing at 24.44% vs 22.69%
- Variant B (adaptive K) achieves the strongest final-round performance at 26.63%, though it is less stable in early rounds
- Variant C (combined) does not additively improve on either individual change, suggesting the two modifications interact non-trivially in the density estimation step
- Overall accuracy is lower than the paper's reported figures due to using 50 SimCLR epochs rather than the 500 used in the original implementation

---

## Repository Structure

```
typiclust_cw2.ipynb    — Full implementation: Task 1 (original) + Task 3 (ablation study)
README.md              — This file
```

---

## Implementation Notes

- SimCLR trained for 50 epochs (vs 500 in the paper) due to computational constraints on Colab
- ResNet-18 backbone adapted for CIFAR-10 (3x3 initial conv, no max-pooling)
- Mini-Batch K-Means used for scalability; clusters with fewer than 5 points excluded from selection
- Classifier retrained from scratch each active learning round to isolate sample quality effects
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

- Hacohen et al. (2022) — https://arxiv.org/abs/2202.02794
- Chen et al. (2020) SimCLR — https://arxiv.org/abs/2002.05709
- Settles (2009) Active Learning Survey — http://burrsettles.com/pub/settles.activelearning.pdf
- Mittal et al. (2019) — https://arxiv.org/abs/1912.05361
