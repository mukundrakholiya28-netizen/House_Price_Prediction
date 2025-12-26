def featureEngineering(df):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    df['TotalBath'] = (
        df['FullBath'] +
        0.5 * df['HalfBath'] +
        df['BsmtFullBath'] +
        0.5 * df['BsmtHalfBath']
    )

    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['YearsSinceRemodel'] = df['YrSold'] - df['YearRemodAdd']

    return df
