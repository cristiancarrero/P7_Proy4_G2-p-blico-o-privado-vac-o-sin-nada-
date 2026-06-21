import pandas as pd

df = pd.read_csv("data/raw/idealista18_madrid_sale (1).csv.gz")
print("Shape:", df.shape)
print("\nTipos:")
print(df.dtypes.to_string())
print("\nPRICE describe:")
print(df["PRICE"].describe())
print("\nPERIOD unique:", df["PERIOD"].nunique(), list(df["PERIOD"].unique()[:5]))
print("\nCorrelaciones con PRICE (top 10):")
corr = df.select_dtypes(include="number").corr()["PRICE"].abs().sort_values(ascending=False)
print(corr.head(11).to_string())
