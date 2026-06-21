import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_parquet("data/processed/habitaciones_madrid.parquet")
print("Filas:", len(df))
print(df.isnull().sum())

os.makedirs("data/figs", exist_ok=True)

sns.histplot(df["precio"], bins=50)
plt.title("Distribucion de precio de alquiler de habitacion")
plt.savefig("data/figs/01_precio_dist.png")
plt.close()

precio_zona = df.groupby("zona_agrupada")["precio"].median().sort_values(ascending=False).head(15)
precio_zona.plot(kind="barh")
plt.title("Precio mediano por zona (top 15)")
plt.tight_layout()
plt.savefig("data/figs/02_precio_zona.png")
plt.close()

sns.boxplot(data=df, x="gastos_incluidos", y="precio")
plt.title("Precio segun gastos incluidos")
plt.savefig("data/figs/03_precio_gastos.png")
plt.close()

print("EDA completo. Graficos guardados en data/figs/")
