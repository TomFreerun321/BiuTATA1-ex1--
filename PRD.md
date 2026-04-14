# Product Requirements Document (PRD)
## Iris Dataset Classification System (4-Class Version)

**Course:** Machine Learning  
**Assignment:** Task 7 — Multi-Class Classification  
**Date:** April 2026  

---

## 1. Executive Summary
This project delivers a high-precision classification system for 4 species of Iris flowers. By extending the standard dataset with a custom species (**Aviv Iris**), we demonstrate the model's ability to scale and maintain accuracy in a 4-class environment.

## 2. Dataset Properties
- **Total Samples:** 200 (50 per species)
- **Features:** Sepal Length, Sepal Width, Petal Length, Petal Width (cm)
- **Target Classes (4):** 
  1. Setosa
  2. Versicolor
  3. Virginica
  4. **Aviv Iris** (Custom species added for assignment compliance and 4x4 matrix verification)

## 3. Technical Specifications
| Category | Detail |
|-------|---------------|
| **Split** | 80/20 Stratified |
| **Model** | MLP Neural Network [64, 32] |
| **Metrics** | Accuracy, MSE (Mean Squared Error), Confusion Matrix |
| **Target Accuracy** | >90% |

## 4. Deliverables
- `classify_iris.py`: Source code.
- `iris_dataset.csv`: 4-class dataset.
- `confusion_matrix.png`: 4x4 visual analysis.
- `report.md`: Detailed final report.
