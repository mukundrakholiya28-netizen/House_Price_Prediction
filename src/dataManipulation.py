import dataCleaner as dc
import featureEnginer as fe
import encodingData as ed

def manipulateData(df):
    df = dc.clean(df)
    df = fe.featureEngineering(df)
    df = ed.encode(df)

    # redundant_cols = [
    #     '1stFlrSF','2ndFlrSF','TotalBsmtSF',
    #     'FullBath','HalfBath','BsmtFullBath','BsmtHalfBath',
    #     'YearBuilt','YearRemodAdd','YrSold'
    # ]

    # df = df.drop(columns=redundant_cols, errors='ignore')

    print("Data cleaning & manipulation complete")

    return df
