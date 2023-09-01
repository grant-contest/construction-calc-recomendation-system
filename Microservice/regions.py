import pandas as pd

def get_regions():
    data_path = "Data/"
    regions = pd.read_csv(data_path + "region.csv")
    regions = regions[regions.columns[2]]
    b = pd.Series(["Донецкая Народная Республика (ДНР)", "Луганская Народная Республика (ЛНР)", "Запорожская область", "Херсонская область"], index=[86, 87, 88, 89])
    regions = regions._append(b)
    regions = regions.drop(85)
    return regions