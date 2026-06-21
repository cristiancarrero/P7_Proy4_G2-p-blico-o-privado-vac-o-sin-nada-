import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_parquet("data/processed/idealista18_madrid.parquet")
print("Filas:", len(df))
print(df.isnull().sum())

os.makedirs("data/figs", exist_ok=True)

sns.histplot(df["PRICE"], bins=50)
plt.title("Distribucion de precio de venta")
plt.xlabel("Precio (EUR)")
plt.savefig("data/figs/01_precio_dist.png")
plt.close()

sns.scatterplot(data=df.sample(5000, random_state=42), x="CONSTRUCTEDAREA", y="PRICE", alpha=0.3)
plt.title("Precio vs Superficie")
plt.xlabel("Superficie (m2)")
plt.ylabel("Precio (EUR)")
plt.savefig("data/figs/02_precio_area.png")
plt.close()

sns.boxplot(data=df, x="HASLIFT", y="PRICE")
plt.title("Precio segun ascensor")
plt.savefig("data/figs/03_precio_ascensor.png")
plt.close()

print("EDA completo. Graficos guardados en data/figs/")
