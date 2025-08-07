# 🧬 ASD Microbiome Explorer

**ASD Microbiome Explorer** is a Python-based tool for analyzing gut microbiome data in individuals with Autism Spectrum Disorder (ASD) and typically developing (TD) individuals. It provides statistical metrics, differential abundance analysis, and machine learning models to extract insights from microbiome abundance data.

---

## 📊 Features

- **Alpha Diversity (Shannon Index)**  
  Measure within-sample microbial diversity.

- **Beta Diversity (Bray-Curtis Distance + MDS)**  
  Visualize between-sample differences using multidimensional scaling.

- **Differential Abundance Analysis**  
  Identify taxa significantly more or less abundant in ASD vs. TD groups (Mann–Whitney U test + FDR correction).

- **Supervised Machine Learning**  
  Train models like Logistic Regression, Random Forest, and SVM to classify ASD vs. TD samples based on microbiome profiles.

- **Visualization (Dash App)**  
  Plot interactive diversity charts and classification results.

- **Unit Testing (Pytest)**  
  Test core functions for correctness and reproducibility.