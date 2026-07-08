import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ------------------------------
# 1. Gerando dataset automático
# ------------------------------

np.random.seed(42)

n = 200  # quantidade de dados

distancia = np.random.randint(200, 2000, n)
atraso_anterior = np.random.randint(0, 2, n)
clima_ruim = np.random.randint(0, 2, n)
trafego_alto = np.random.randint(0, 2, n)

# Regra para definir atraso (simulando lógica real)
atrasado = (
    (atraso_anterior == 1) |
    (clima_ruim == 1) |
    (trafego_alto == 1)
).astype(int)

df = pd.DataFrame({
    'distancia': distancia,
    'atraso_anterior': atraso_anterior,
    'clima_ruim': clima_ruim,
    'trafego_alto': trafego_alto,
    'atrasado': atrasado
})

# ------------------------------
# 2. Separando dados
# ------------------------------

X = df.drop('atrasado', axis=1)
y = df['atrasado']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# 3. Normalização
# ------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------
# 4. Modelo de rede neural
# ------------------------------

model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=500)

model.fit(X_train, y_train)

# ------------------------------
# 5. Avaliação
# ------------------------------

accuracy = model.score(X_test, y_test)
print(f"Acurácia: {accuracy:.2f}")

# ------------------------------
# 6. Previsões
# ------------------------------

y_pred = model.predict(X_test)

# ------------------------------
# 7. Matriz de confusão (gráfico)
# ------------------------------

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Matriz de Confusão - Classificação de Voos")
plt.show()