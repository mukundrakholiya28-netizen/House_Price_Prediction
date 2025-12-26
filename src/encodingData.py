import pandas as pd

def encode(df):
    qual_map = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0}

    qual_cols = [
        'ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC',
        'KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC'
    ]

    for col in qual_cols:
        df[col] = df[col].map(qual_map)

    df = pd.get_dummies(df)

    return df
