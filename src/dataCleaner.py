import pandas as pd

def clean(df):

    # # Calculate the percentage of missing values for each column
    # missing_data = df.isnull().sum().sort_values(ascending=False)
    # missing_percentage = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)

    # # Combine into a table to see the top 20
    # missing_info = pd.concat([missing_data, missing_percentage], axis=1, keys=['Total', 'Percent'])
    # print(missing_info.head(20))

    cols_to_fill_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
    for col in cols_to_fill_none:
        df[col] = df[col].fillna('None')

    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'] \
                           .transform(lambda x: x.fillna(x.median()))

    cat_cols = [
        'GarageType','GarageFinish','GarageQual','GarageCond',
        'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'
    ]
    for col in cat_cols:
        df[col] = df[col].fillna('None')

    num_cols = [
        'GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1',
        'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath'
    ]
    for col in num_cols:
        df[col] = df[col].fillna(0)

    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    df['MSZoning'] = df.groupby('Neighborhood')['MSZoning'] \
                        .transform(lambda x: x.fillna(x.mode()[0]))

    df['MasVnrType'] = df['MasVnrType'].fillna('None')
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("None")
        else:
            df[col] = df[col].fillna(0)

    return df
