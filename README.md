# Customer Segmentation using KNN

Customer Segmentation is one of the most important applications of supervised machine learning. Using the **K-Nearest Neighbors (KNN)** algorithm along with preprocessing pipelines, companies can classify customers into distinct segments, enabling targeted marketing and personalized services.

![Customer Segmentation](https://user-images.githubusercontent.com/90209933/147927776-948a9af0-18bb-49ac-bbc0-30efd2790649.png)

---

## 📌 Table of Contents

- [Overview](#overview)
- [What is Customer Segmentation?](#what-is-customer-segmentation)
- [What is the KNN Algorithm?](#what-is-the-knn-algorithm)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Steps for Implementation](#steps-for-implementation)
- [Model Pipeline](#model-pipeline)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)

---

## Overview

In this machine learning project, we apply the **K-Nearest Neighbors (KNN)** classification algorithm to classify telecom customers into four service categories based on their demographic and behavioral attributes. The project includes complete preprocessing steps and is wrapped in a **scikit-learn model pipeline** for clean, reproducible predictions.

---

## What is Customer Segmentation?

Customer Segmentation is the process of dividing a customer base into several groups of individuals that share similarities in ways relevant to marketing — such as **gender**, **age**, **annual income**, and **spending habits**.

Companies that deploy customer segmentation recognize that every customer has different requirements and respond better to tailored marketing approaches. By analyzing collected data, businesses can:

- Gain a deeper understanding of customer preferences
- Discover valuable segments that maximize profit
- Strategize marketing techniques more efficiently
- Minimize investment risk

The technique relies on key differentiators such as demographics, geography, economic status, and behavioral patterns to divide customers into actionable groups.

---

## What is the KNN Algorithm?

**K-Nearest Neighbors (KNN)** is a simple, non-parametric supervised learning algorithm used for both classification and regression tasks.

### How KNN Works:
1. Choose the number of neighbors **K**.
2. Calculate the **Euclidean distance** between the query point and all training samples.
3. Select the **K nearest neighbors** based on computed distances.
4. For classification, the output is determined by **majority voting** among the K neighbors.
5. Assign the most frequent class label to the new data point.

### Choosing the Optimal K:
The choice of K significantly affects model performance:
- A **small K** (e.g., K=1) leads to overfitting — the model is too sensitive to noise.
- A **large K** leads to underfitting — decision boundaries become too smooth.
- The optimal K is typically found using **cross-validation** and by plotting the error rate against different K values.

```python
  Ks = 40
    mean_acc = np.zeros((Ks-1))
    std_acc = np.zeros((Ks-1))

  for n in range(1,Ks):
         model = KNeighborsClassifier(n_neighbors = n).fit(X_train_norm,y_train)
         y_pred= model.predict(X_test_norm)
         mean_acc[n-1] = metrics.accuracy_score(y_test, y_pred)
         std_acc[n-1]=np.std(y_pred==y_test)/np.sqrt(y_pred.shape[0])
```

---

## Dataset

The dataset (`teleCust1000t.csv`) is a telecom customer dataset sourced from a GitHub repository, commonly used for customer classification tasks.

- **Total Records:** 1,000 customers
- **Total Features:** 12 columns (11 input features + 1 target)
- **Target Variable:** `custcat` — 4 customer categories (1, 2, 3, 4)
- **Missing Values:** None

| Feature | Type | Description |
|---|---|---|
| `region` | int | Geographic region of the customer (1, 2, or 3) |
| `tenure` | int | Number of years the customer has been with the company (1–72) |
| `age` | int | Age of the customer |
| `marital` | int | Marital status (0 = Single, 1 = Married) |
| `address` | int | Number of years at current address |
| `income` | float | Annual income of the customer (in thousands) |
| `ed` | int | Education level (1 = Below college to 5 = Post-graduate) |
| `employ` | int | Number of years currently employed |
| `retire` | float | Retired status (0 = No, 1 = Yes) |
| `gender` | int | Gender (0 = Female, 1 = Male) |
| `reside` | int | Number of people in the household |
| `custcat` | int | **Target** — Customer category (1: Basic, 2: E-Service, 3: Plus, 4: Total Service) |

### Target Class Distribution

| Category | Label | Count |
|---|---|---|
| 1 | Basic Service | 266 |
| 2 | E-Service | 217 |
| 3 | Plus Service | 281 |
| 4 | Total Service | 236 |

---

## Project Structure

```text
Customer-segmentation-knn/
│
├── src/
│   ├── preprocessing.py
│   └── train_model.py
│
├── notebook.ipynb
│  
│
├── README.md
└── .gitignore
```

---

## Steps for Implementation

### 1. Import Necessary Packages

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

---

### 2. Data Loading & Exploration

```python
df = pd.read_csv('Mall_Customers.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
```

---

### 3. Data Preprocessing

```python
# Encode categorical variable
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Female=0, Male=1

# Define features and target
X = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
y = df['Segment']  # Target customer segment label
```

---

### 4. Visualizations

**Bar Plot – Gender Distribution**
```python
sns.countplot(x='Gender', data=df, palette='Set2')
plt.title('Gender Distribution of Customers')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()
```
![Gender Distribution](assets/Gender DIstribution.png)

**Histogram – Age Distribution**
```python
plt.hist(df['Age'], bins=15, color='steelblue', edgecolor='black')
plt.title('Histogram – Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
```
**Density Plot – Annual Income**
```python
df['Annual Income (k$)'].plot(kind='density', color='green')
plt.title('Density Plot – Annual Income')
plt.xlabel('Annual Income (k$)')
plt.show()
```

### 5. Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=96
)
```

---

## Model Pipeline

The project uses a **scikit-learn Pipeline** to combine preprocessing and model training into a single reproducible workflow.


### Finding Optimal K

```python
 Ks = 40
  mean_acc = np.zeros((Ks-1))
  std_acc = np.zeros((Ks-1))

 for n in range(1,Ks):
   model = KNeighborsClassifier(n_neighbors = n).fit(X_train_norm,y_train)
   y_pred= model.predict(X_test_norm)
   mean_acc[n-1] = metrics.accuracy_score(y_test, y_pred)
   std_acc[n-1]=np.std(y_pred==y_test)/np.sqrt(y_pred.shape[0])

---

## Results

After training the KNN model and tuning the value of K, the model is evaluated using:

```python
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))
```

### Customer Segment Interpretation

| Segment | Description |
|---|---|
| Segment 1 | Young customers with high income and high spending score |
| Segment 2 | Middle-aged customers with medium income and medium spending |
| Segment 3 | Older customers with low income and low spending score |
| Segment 4 | High income but low spending — potential target for upselling |
| Segment 5 | Low income but high spending — brand-loyal customers |

With the help of KNN-based segmentation, companies can understand customer behavior more precisely, enabling them to release targeted products and services based on income, age, and spending patterns.

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.x | Core programming language |
| Jupyter Notebook | Interactive development environment |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Matplotlib / Seaborn | Data visualization |
| Scikit-learn | KNN model, pipeline, preprocessing, evaluation |

---

## How to Run

1. **Clone the repository**
```bash
git clone https://github.com/koushiikk/Customer-segmentation-knn.git
cd Customer-segmentation-knn
```

2. **Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

3. **Launch Jupyter Notebook**
```bash
jupyter notebook Untitled.ipynb
```

4. **Run all cells** to reproduce the full analysis, visualizations, and model results.

---

## 📬 Connect

Feel free to raise an issue or submit a pull request if you'd like to contribute or suggest improvements!
