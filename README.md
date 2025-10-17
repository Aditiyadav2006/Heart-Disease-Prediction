# 🫀 Heart Disease Prediction: Comparative Model Study

## Project Overview

This project focuses on predicting the presence of **Heart Disease** in patients using a dataset of key clinical measurements. This is a critical **binary classification** task (Heart Disease vs. No Heart Disease) designed to compare the effectiveness of four popular machine learning algorithms:

1.  **Random Forest Classifier (RF)**
2.  **Support Vector Machine (SVM)**
3.  **Gaussian Naive Bayes (GNB)**
4.  **Decision Tree Classifier (DT)**

The primary goal is to identify the most robust and accurate model for clinical decision support.

## Key Files

| File Name | Description |
| :--- | :--- |
| `Heart disease prediction.py` | The main Jupyter Notebook containing data cleaning, feature encoding, scaling, training, and comparative evaluation of all four classification models. |
| `data-11.csv` | The dataset containing patient records with clinical measurements and the binary `HeartDisease` outcome. |
| `README.md` | This overview file. |

## Methodology

### 1. Data Preparation and Feature Engineering
Standard practices for medical machine learning datasets were applied:

* **Handling Categorical Features:** Features like `ChestPainType`, `RestingECG`, and `ST_Slope` were converted into a numerical format suitable for modeling (e.g., using One-Hot Encoding).
* **Feature Scaling:** Numerical features (e.g., `RestingBP`, `Cholesterol`, `MaxHR`) were scaled using **StandardScaler** to ensure that all models (especially SVM) are not unfairly biased by features with larger numerical ranges.
* **Data Splitting:** The data was split into training and testing sets to evaluate the final model performance on unseen data.

### 2. Comparative Model Training
Four distinct models were trained and tested on the preprocessed data:

* **Random Forest:** An ensemble method that builds multiple decision trees, leveraging their collective wisdom to improve prediction accuracy.
* **SVM:** A powerful model that finds the optimal hyperplane to separate the two classes in a high-dimensional feature space.
* **Gaussian Naive Bayes:** A fast, probabilistic classifier assuming features are normally distributed and independent.
* **Decision Tree:** A simple, rule-based model used as a performance baseline.

## Model Performance Analysis

The models were evaluated based on **Accuracy** on the test set.

| Model | Accuracy Score | Rank |
| :--- | :--- | :--- |
| **Random Forest** | $\mathbf{0.880}$ | $\mathbf{1}$ |
| **SVM** | $\mathbf{0.864}$ | $\mathbf{2}$ |
| **Naive Bayes** | $\mathbf{0.842}$ | $\mathbf{3}$ |
| **Decision Tree** | $\mathbf{0.788}$ | $\mathbf{4}$ |

### Key Findings:

* **Best Performer:** The **Random Forest Classifier** achieved the highest accuracy of $\mathbf{88.0\%}$, demonstrating its superior ability to capture complex, non-linear relationships in medical data without overfitting.
* **Strong Generalization:** Both **Random Forest** and **SVM** showed strong performance ($\mathbf{86.4\%}$ and above), confirming that they are well-suited for this type of classification task.
* **Performance Baseline:** The simpler **Decision Tree** had the lowest accuracy, underscoring the benefit of using more advanced or ensemble methods like Random Forest for this predictive challenge.

## Technologies and Libraries

* **Python 3.x**
* **VS code**
* `pandas` & `numpy` (for data manipulation)
* `scikit-learn` (for **SVC**, **GaussianNB**, **DecisionTreeClassifier**, **RandomForestClassifier**, `StandardScaler`, and metrics)
* `matplotlib` & `seaborn` (for visualization and comparative charts)
* 
You can also view the notebook directly on GitHub or platforms like **Google Colab** without needing a local setup.


## 👩‍💻 Author
**Aditi K**  
CSE (AI & ML) | Heart Disease Prediction 
