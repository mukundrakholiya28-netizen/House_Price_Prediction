import numpy as np

def split(df, validate_size=0.2, random_state=42):
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    split_idx = int((1 - validate_size) * len(df))

    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    y_train = np.log1p(train_df['SalePrice'])
    y_val = np.log1p(val_df['SalePrice'])

    X_train = train_df.drop(['SalePrice', 'Id'], axis=1)
    X_val = val_df.drop(['SalePrice', 'Id'], axis=1)

    return X_train, y_train, X_val, y_val
