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

## 👨‍💻 Modelos recomendados

* Regressão Logística
* Árvore de Decisão / Random Forest
* Gradient Boosting (opcional)

## 📊 Entregáveis

* Pipeline completo de machine learning
* Relatório com principais métricas e importância das variáveis
* Conclusão estratégica com insights de negócio

---

Este projeto foi desenvolvido com foco em boas práticas de ETL, Análise de Dados e Machine Learning.

