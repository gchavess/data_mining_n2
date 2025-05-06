import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar o dataset
df = pd.read_csv("database.csv", sep=';', on_bad_lines='skip')

# 2. Selecionar as variáveis de interesse
colunas = [
    "Q20Age",      # Idade
    "Q21Gender",   # Gênero
    "Q22Income",   # Renda
    "Q23FLY",      # Frequência de voos
    "Q6LONGUSE"    # Tempo de uso do aeroporto
]
df_selecionado = df[colunas].copy()

# 3. Pré-processamento
# Remover linhas com valores ausentes
df_selecionado = df_selecionado.dropna()

# Transformar variável categórica (Gênero) em número
df_selecionado["Q21Gender"] = df_selecionado["Q21Gender"].astype("category").cat.codes

# Normalizar os dados
scaler = StandardScaler()
dados_normalizados = scaler.fit_transform(df_selecionado)

# 4. Aplicar o KMeans com 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(dados_normalizados)

# Adicionar a coluna de cluster ao DataFrame
df_selecionado["cluster"] = clusters

# 5. Analisar o tamanho de cada cluster
tamanho_percentual = df_selecionado["cluster"].value_counts(normalize=True) * 100
print("Tamanho dos clusters (%):\n", tamanho_percentual)

# 6. Ver o perfil médio de cada cluster
perfil = df_selecionado.groupby("cluster").mean()
print("\nPerfil médio por cluster:\n", perfil)

# 7. (Opcional) Visualizar os clusters
sns.pairplot(df_selecionado, hue="cluster")
plt.suptitle("Clusters de Passageiros - SFO", y=1.02)
plt.show()
