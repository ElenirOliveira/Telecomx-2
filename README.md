# Projeto Churn de Clientes - Telecom X

Este projeto faz parte da iniciativa da empresa Telecom X para compreender e reduzir a evasão de clientes (churn). A análise é conduzida em três camadas: Bronze (Extração), Silver (Transformação) e Gold (Análise e Visualização).

## Objetivo

Identificar padrões e fatores associados à evasão de clientes, gerando insights valiosos para a equipe de Data Science, que poderá utilizá-los na construção de modelos preditivos.

## Tecnologias Utilizadas

* Python
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

## Etapas do Projeto

### ✨ Camada Bronze - Extração

* Importação dos dados via API ou fonte JSON.
* Conversão dos dados para um DataFrame Pandas.
* Expansão de colunas aninhadas (ex: `customer`, `phone`, `internet`, `account`).

### 🌟 Camada Silver - Transformação

* Verificação e tratamento de dados ausentes, duplicados e inconsistências.
* Criação da coluna `Contas_Diarias` (valor do faturamento mensal / 30).
* Conversão de variáveis categóricas (ex: "Sim" / "Não") para formato binário (1 / 0).
* Renomeação e padronização de colunas para clareza.

### 🔍 Camada Gold - Análise e Visualização

* Análise descritiva (média, mediana, desvio padrão, etc.).
* Visualização da proporção de clientes que cancelaram e os que permaneceram.
* Estudo de churn por variáveis categóricas: gênero, contrato, pagamento.
* Análise por variáveis numéricas: total gasto, tempo de contrato.
* (Opcional) Análise de correlação entre variáveis e churn.

---

# Parte 2 - Prevendo Churn com Machine Learning

## Descrição Geral

A segunda parte do projeto tem como objetivo prever o cancelamento de clientes utilizando algoritmos de machine learning, regressão e estatística aplicada.

## 📈 Aplicação Prática

Você irá:

* Aplicar conceitos de estatística e regressão linear para modelar os dados.
* Separar os dados em treino e teste de forma equilibrada.
* Realizar análise de correlação para entender variáveis relevantes.
* Construir um pipeline de machine learning para prever churn.

## 🎯 Missão

Desenvolver modelos preditivos robustos que identifiquem clientes com alto risco de evasão, permitindo ações preventivas.

## 🧰 O que você vai praticar

* ✅ Pré-processamento de dados (tratamento, encoding, normalização)
* ✅ Seleção de variáveis com base em correlação
* ✅ Treinamento de modelos de classificação
* ✅ Avaliação de desempenho (acurácia, precisão, recall, F1-score)
* ✅ Interpretação dos resultados com foco em negócios

## 👨‍💻 Pipeline Sugerido

```python
# 📦 Importação de bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 🧼 Pré-processamento
df_ml = df_expandido.copy()

# Encoding de variáveis categóricas
label_cols = df_ml.select_dtypes(include='object').columns
for col in label_cols:
    df_ml[col] = LabelEncoder().fit_transform(df_ml[col].astype(str))

# Separação entre variáveis e alvo
X = df_ml.drop('churn', axis=1)
y = df_ml['churn']

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🤖 Modelos
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# 📊 Avaliação
print("Regressão Logística:\n", classification_report(y_test, y_pred_lr))
print("Random Forest:\n", classification_report(y_test, y_pred_rf))

# Matriz de Confusão
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues')
plt.title('Matriz - Regressão Logística')

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens')
plt.title('Matriz - Random Forest')
plt.show()

# 🔍 Importância das variáveis (Random Forest)
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False).plot(kind='bar', figsize=(12,6), title='Importância das Variáveis - RF')
plt.tight_layout()
plt.show()
```
![Matriz de correlaçao](https://github.com/ElenirOliveira/Telecomx-2/blob/main/Import%C3%A2ncia%20das%20Vari%C3%A1veis%20-%20RF.png)
![Matriz de correlaçao](https://github.com/ElenirOliveira/Telecomx-2/blob/main/Matriz%20-%20Random%20Forest.png)
![Matriz de correlaçao](https://github.com/ElenirOliveira/Telecomx-2/blob/main/matriz%20de%20confusao.png)
## 📊 Entregáveis

* Pipeline completo de machine learning
* Relatório com principais métricas e importância das variáveis
* Conclusão estratégica com insights de negócio

---

Este projeto foi desenvolvido com foco em boas práticas de ETL, Análise de Dados e Machine Learning.


