# 🏠 House Price Prediction using Custom Linear Regression

This project predicts house prices using a **custom-built Linear Regression model from scratch**, without using `sklearn`'s regression implementation.

It is based on the **Ames Housing Dataset** and includes:
- Data cleaning
- Feature engineering
- Encoding
- Standardization
- Gradient Descent with Early Stopping

---

## 📂 Project Structure

House-Price-Prediction/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── testAns.csv
│
├── src/
│   ├── dataCleaner.py
│   ├── featureEnginer.py
│   ├── encodingData.py
│   ├── dataManipulation.py
│   ├── data_splitter.py
│   ├── standardization.py
│   ├── LinearRegression.py
│   └── model.py
│
├── requirements.txt
├── README.md
├── .gitignore

## 🚀 Features

- Custom **Linear Regression implementation**
- Manual **gradient descent**
- **Early stopping** to prevent overfitting
- Feature alignment between train & test
- Log-transform target variable
- Fully modular pipeline

## 🧠 Workflow

1. **Data Cleaning**
   - Missing value handling
   - Neighborhood-wise imputation

2. **Feature Engineering**
   - Total square footage
   - Total bathrooms
   - House age
   - Years since remodeling

3. **Encoding**
   - Ordinal quality mapping
   - One-hot encoding for categorical features

4. **Model Training**
   - Gradient Descent
   - Mean Squared Error loss
   - Early stopping

5. **Prediction**
   - Reverse log transformation
   - CSV submission generation

## ⚙️ How to Run

```bash
>>> pip install -r requirements.txt

>>> python src/model.py

data/testAns.csv will be generated

