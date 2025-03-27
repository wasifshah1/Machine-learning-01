# 🍄 Mushroom Classification with SVM (RBF Kernel)

## 🧠 Introduction
Mushrooms can be safe, delicious—or dangerously poisonous. Accurate classification is crucial for foragers, cultivators, and researchers. This project uses a **Support Vector Machine (SVM)** with an **RBF kernel** to classify mushrooms as edible (`0`) or poisonous (`1`) using morphological and habitat characteristics from the [UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom).

This end-to-end machine learning pipeline includes:

- Data loading and exploration
- One-hot encoding of categorical variables
- Binary classification using SVM with RBF kernel
- Evaluation via classification metrics and ROC
- Interpretability with visualizations like t-SNE, radar chart, and partial dependence

---

## 📊 Dataset Overview

- **Source:** [UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom)
- **Samples:** 8124 mushrooms
- **Original Features:** 22 categorical + 1 class column
- **Processed Features:** 95 (after one-hot encoding)
- **Target Variable:** `class` – edible (`e`) or poisonous (`p`) → mapped to 0/1

---

## ⚙️ Preprocessing Steps

1. **Label Transformation:**  
   Converted `class` column to binary (`0` = edible, `1` = poisonous)

2. **One-Hot Encoding:**  
   All 22 categorical features encoded using `pd.get_dummies`, resulting in 95 columns

3. **Train–Test Split:**  
   80% training, 20% testing (`random_state=42`)

---

## ⚙️ Model Overview

### Handles non-linear decision boundaries
We use an SVM with an RBF (Radial Basis Function) kernel to allow for non-linear separation of mushroom classes in high-dimensional space.

### Probability estimates enabled for ROC analysis
Setting `probability=True` in `SVC` enables probability outputs, which are essential for generating ROC curves and AUC metrics.

---

## ✅ Performance

- **Accuracy:** `100%`
- **ROC AUC Score:** `1.00`
- **Precision / Recall:** Perfect scores for both edible and poisonous classes

### Confusion Matrix

|                | Pred: Edible | Pred: Poisonous |
|----------------|--------------|-----------------|
| **Actual: Edible**    | 842          | 0               |
| **Actual: Poisonous** | 0            | 783             |

✅ No false positives or false negatives — the model perfectly separates classes.

---

## 📈 Visualizations

- **t-SNE Plot:** 2D scatter reveals clear, non-overlapping clusters
- **Confusion Matrix:** Plasma colormap highlights classification confidence
- **ROC Curve:** AUC = 1.0 (ideal curve)
- **Radar Chart:** Top 8 features from permutation importance (e.g., `odor_foul`, `gill_size_narrow`)
- **Partial Dependence Plot:** Shows binary impact of key dummy features on predicted probability

---

## 💡 Key Insights

- The dataset is **easily separable**, largely due to highly discriminative features like `odor`
- **Permutation importance** confirms the model relies heavily on `odor` and `gill size`
- **Perfect accuracy** here reflects a curated dataset — not always reproducible in real-world data

---

## 🔭 Future Work

- Introduce **ambiguous mushroom types** for multi-class extension
- Apply **cost-sensitive learning** to penalize false negatives more heavily
- Evaluate with **cross-validation** to confirm generalization
- Compare with **Random Forests**, **Logistic Regression**, or **Neural Networks**
- Use **SHAP / LIME** for local interpretability per mushroom sample

---

## 🧑‍🏫 Teaching Emphasis

- Handling **fully categorical datasets** with one-hot encoding
- Visual tools like **t-SNE** and **radar charts** for interpretability
- High-performance modeling with simple classifiers like **SVM**
- Model evaluation beyond accuracy: **ROC, AUC, Confusion Matrix**

---


---

## 🚀 Getting Started

### Clone the repository:
```bash
git clone https://github.com/wasifshah1/mushroom-svm-classification.git
cd mushroom-svm-classification

