# 【AEI 2025】DKGPL-SmeiDGFD code

---

**This is the source code for "Domain Knowledge Guided Pseudo-Label Generation Framework for Semi-supervised Domain Generalization Fault Diagnosis". You can refer to the following steps to reproduce the smei-supervised domain generalization of the fault diagnosis task.**

# :triangular_flag_on_post:Highlights

----

**A new semi-supervised domain generalization fault diagnosis method is introduced, enhancing the utilization of unlabeled data and learning domain-invariant features for improved generalization.**

**A pseudo-label generation framework is proposed, using domain knowledge-guided weight adjustment to optimize classifier weights and address the imbalance in pseudo-label quality and quantity across diverse working conditions.** 

**A distributed feature expansion strategy is introduced, enhancing feature space search through random perturbations and hybrid channel expansion, improving generalization under varying conditions while reducing pseudo-labeling errors.**

**A domain-aware prototype contrast method is proposed, aligning pseudo-labels and class prototypes to learn domain-invariant knowledge, ensuring stable predictions and more accurate pseudo-label generation under unknown conditions.**

# ⚙️Abstract

----

**Fault diagnosis methods based on domain generalization have gained significant attention. However, obtaining sufficient labeled samples from various source domains is costly and challenging. Therefore, a new semi-supervised domain generalization fault diagnosis method based on a domain knowledge-guided pseudo-label generation framework is proposed. This method efficiently generates high-quality pseudo-labels and improves their robustness for cross-domain generalization. First, a domain knowledge-guided weight adjustment strategy is proposed to modulate the weights of shared classifiers by creating domain-level information vectors and adaptively adjusting class-level confidence thresholds to realize the trade-off between the quantity and quality of pseudo-label. 
Further, a distributed feature expansion strategy is proposed to expand the feature space by enhancing the feature statistics in the channel dimension, improve the model's cross-domain adaptability, thereby reduce the accumulation of errors caused by feature shifts in pseudo-labels. Finally, a domain-aware prototype construct is proposed to maximize the similarity between intra-domain and cross-domain category prototypes, and align the category prototype similarity with the pseudo-labels to achieve domain-invariant knowledge learning and implicitly promote the high-quality generation of pseudo-labels. Experiments show that the proposed method generates high-quality pseudo-labels and achieves superior diagnostic accuracy in semi-supervised domain generalization fault diagnosis.**

# **:sunny:**Proposed Method

---

![pVV1EYd.jpg](https://s21.ax1x.com/2025/06/19/pVV1EYd.jpg)



# 📄Citation

## If you find this paper and repository useful, please cite us! ⭐⭐⭐

----

```
@article{wei2025domain,
  title={Domain knowledge guided pseudo-label generation framework for semi-supervised domain generalization fault diagnosis},
  author={Wei, Jiacheng and Wang, Qibin and Zhang, Guowei and Ma, Hongbo and Wang, Yi},
  journal={Advanced Engineering Informatics},
  volume={67},
  pages={103540},
  year={2025},
  publisher={Elsevier}
}
```

