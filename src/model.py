import pandas as pd
import numpy as np
import os

import dataManipulation as dm
import data_splitter as ds

from LinearRegression import LinearRegressionCustom
from standardization import Standardizer
from sklearn.metrics import mean_squared_error, r2_score

# ======================
# TRAINING
# ======================

# 1. Get the path of the current script (model.py)
script_dir = os.path.dirname(__file__) 

# 2. Go up one level to the root (House-Price-Prediction)
root_dir = os.path.join(script_dir, '..')

# 3. Point to the data file
data_path = os.path.join(root_dir, 'data', 'train.csv')

# 4. Load the dataframe
df = pd.read_csv(data_path)

# 1 Clean & manipulate FIRST
df = dm.manipulateData(df)

# Save feature structure
FEATURE_COLUMNS = df.drop(['SalePrice', 'Id'], axis=1).columns

# 2️ Split
X_train, y_train, X_val, y_val = ds.split(df)

# 3 Standardize (TRAIN stats only)
scaler = Standardizer()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 4 Add Bias (FAST & SAFE)
X_train = np.c_[np.ones(len(X_train)), X_train.values]
X_val = np.c_[np.ones(len(X_val)), X_val.values]

# 5 Train model
model = LinearRegressionCustom()
model.fit(X_train, y_train, X_val, y_val)

# ======================
# OPTIONAL EVALUATION
# ======================
preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
r2 = r2_score(y_val, preds)
print(f"R2   : {r2:.4f}")
print(f"RMSE : {rmse:.4f}")

# ======================
# TEST / UNKNOWN DATA
# ======================

# df2 = pd.read_csv("data/test.csv")

# # Same preprocessing
# df2 = dm.manipulateData(df2)

# X_test = df2.drop(['Id'], axis=1)

# #  ALIGN FEATURES (MOST IMPORTANT LINE)
# X_test = X_test.reindex(columns=FEATURE_COLUMNS, fill_value=0)

# # Use TRAIN scaler
# X_test = scaler.transform(X_test)

# # Add bias
# X_test = np.c_[np.ones(len(X_test)), X_test.values]

# # Predict
# predictions = model.predict(X_test)

# # Reverse log transform
# predictions = np.expm1(predictions)

# # Save submission
# pd.DataFrame({
#     "Id": df2["Id"],
#     "SalePrice": predictions
# }).to_csv("data/testAns.csv", index=False)

# print("testAns.csv generated successfully")
