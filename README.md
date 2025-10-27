
## 📘 README — Decision Tree & Random Forest Classification

### 🧠 **Approach of the Solution**

This project demonstrates how to **train**, **evaluate**, and **generate visual PDF reports** for two machine learning models — **Decision Tree** and **Random Forest** — using Scikit-learn.

#### **Workflow:**
1. **Data Splitting and Standardizing**  
   The dataset is divided into training and testing sets to evaluate model generalization and scaled using fit_transform and transform.

2. **Model Training**  
   Two models are trained:
   - `DecisionTreeClassifier`
   - `RandomForestClassifier`

3. **Evaluation Metrics**  
   Each model is tested on the test set and evaluated using:
   - Accuracy  
   - Precision  
   - Recall  
   - F1 Score  
   - ROC-AUC Score  
   - Confusion Matrix  
   - Classification Report  

4. **PDF Report Generation**  
   For each model, a **PDF report** is automatically generated and saved inside the `reports/` directory.  
   Each report includes:
   - Model parameters and metrics  
   - Confusion matrix and classification report  
   - **Decision Tree visualization** (for the Decision Tree model)

5. **Tree Visualization**  
   The decision tree is visualized using `sklearn.tree.plot_tree()` for better interpretability.  
   The visualization is saved as a second page in the PDF report.

6. **Automation & Organization**  
   All results are generated through a single function `train_evaluate()` — making it reusable and scalable for multiple models.

---

## 🎯 **Interview Questions & Answers**

### **1️⃣ How does a decision tree work?**
A Decision Tree splits the dataset based on feature values to maximize class purity.  
Each node represents a decision condition, and leaves represent final predictions.

---

### **2️⃣ What is entropy and information gain?**
- **Entropy:** Measures impurity or randomness in the dataset.  
  \[ Entropy = -\sum p_i \log_2(p_i) \]
- **Information Gain:** Measures how much entropy is reduced after a split.  
  \[ IG = Entropy(parent) - WeightedAvg(Entropy(children)) \]

---

### **3️⃣ How is random forest better than a single tree?**
Random Forest trains multiple trees on random subsets of data and features.  
- Reduces **overfitting**.  
- Increases **accuracy** and **stability**.  
- Handles noise better.

---

### **4️⃣ What is overfitting and how do you prevent it?**
Overfitting happens when a model performs well on training data but poorly on test data.  
**Prevention:**  
- Limit tree depth.  
- Use ensemble methods (Random Forest).  
- Apply cross-validation or regularization.

---

### **5️⃣ What is bagging?**
**Bagging** (Bootstrap Aggregating) involves training multiple models on random subsets of data (with replacement) and aggregating their results.  
It reduces variance and helps avoid overfitting.  
➡️ Random Forest uses bagging internally.

---

### **6️⃣ How do you visualize a decision tree?**
Use:
```python
from sklearn.tree import plot_tree
plot_tree(model, filled=True, feature_names=features, class_names=classes)
```
Or use Graphviz / dtreeviz for detailed visualizations.

---

### **7️⃣ How do you interpret feature importance?**
Tree-based models compute importance based on how much each feature reduces impurity across all trees.  
Higher values → greater influence on model decisions.  
```python
model.feature_importances_
```

---

### **8️⃣ What are the pros and cons of Random Forests?**

**✅ Pros:**
- Reduces overfitting  
- Works with missing or unscaled data  
- Handles high-dimensional data  
- Provides feature importance  

**❌ Cons:**
- Slower training and inference  
- Less interpretable  
- Higher memory usage  

---

### 🏁 **Summary**
This project demonstrates a robust pipeline for:
- Training & evaluating models  
- Generating PDF-based model reports  
- Visualizing tree-based models  
- Understanding core ML concepts through interpretability and metrics.
