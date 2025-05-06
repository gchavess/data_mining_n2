from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
df_selecionado = df_selecionado.dropna()

# Transformar variável categórica (Gênero) em número
df_selecionado["Q21Gender"] = df_selecionado["Q21Gender"].astype("category").cat.codes

# Criar variável binária para a frequência de voos
df_selecionado["frequente"] = (df_selecionado["Q23FLY"] > df_selecionado["Q23FLY"].median()).astype(int)

# Selecionar variáveis independentes e dependente
X = df_selecionado[["Q20Age", "Q21Gender", "Q22Income", "Q6LONGUSE"]]
y = df_selecionado["frequente"]

# Normalizar os dados
scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)

# 4. Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_normalizado, y, test_size=0.2, random_state=42)

# 5. Treinar o modelo de Regressão Logística
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# 6. Fazer previsões
y_pred = modelo.predict(X_test)

# 7. Avaliar o modelo
print("Acurácia do modelo:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
