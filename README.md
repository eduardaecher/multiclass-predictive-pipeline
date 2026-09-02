# MultiClass Predictive Pipeline

Pipeline preditivo multiclasse de ponta a ponta para classificação de dígitos manuscritos, desenvolvido como Mini-Projeto Avaliativo do Módulo 2 do curso de Desenvolvimento de IA para Análise Preditiva.

## 📌 Sobre o projeto

Este projeto implementa, treina, valida, compara e estressa três modelos de Machine Learning distintos — **KNN**, **Random Forest** e uma **Rede Neural (MLP)** — para classificar dígitos manuscritos de 0 a 9 a partir do dataset **MNIST** (70.000 imagens em escala de cinza, 28×28 pixels).

Além do benchmark comparativo entre os modelos, o projeto avalia a robustez e a capacidade de generalização extrema dos classificadores diante de cenários fora do padrão ideal de laboratório:

- **Class Masking**: treinamento de um modelo sem nunca ter visto duas classes específicas (dígitos 4 e 9).
- **Teste de Generalização Extrema (OOD)**: submissão desse modelo a imagens exclusivamente dessas classes ocultas, analisando o comportamento de "falsa certeza" (overconfidence).
- **Inferência com imagens próprias**: teste do melhor modelo com dígitos manuscritos criados pela autora, com um pipeline próprio de pré-processamento (escala de cinza, inversão de cores, centralização e normalização).

## 🎯 Problema que resolve

Classificar corretamente dígitos manuscritos (0–9) a partir da sua representação em pixels, comparando criticamente diferentes paradigmas de aprendizado de máquina — aprendizado por distância/similaridade (KNN), aprendizado por regras e árvores (Random Forest) e aprendizado por redes neurais (MLP) — e discutir suas limitações diante de dados fora de distribuição e de fontes reais do mundo, como caligrafia produzida pelo próprio usuário.

## 🧠 Técnicas e tecnologias utilizadas

**Linguagem e ambiente:**
- Python 3.9
- Jupyter Notebook (desenvolvido no VS Code)

**Bibliotecas principais:**
- [scikit-learn](https://scikit-learn.org/) — KNN, Random Forest, métricas de avaliação, divisão estratificada dos dados e normalização (`MinMaxScaler`)
- [TensorFlow / Keras](https://www.tensorflow.org/) — construção e treinamento da rede neural MLP
- [pandas](https://pandas.pydata.org/) e [NumPy](https://numpy.org/) — manipulação e análise dos dados
- [matplotlib](https://matplotlib.org/) e [seaborn](https://seaborn.pydata.org/) — visualização de dados, gráficos de distribuição e mapas de calor das matrizes de confusão
- [Pillow (PIL)](https://python-pillow.org/) — pipeline de pré-processamento das imagens manuscritas próprias

**Metodologia:**
- Análise Exploratória de Dados (EDA) com verificação de balanceamento de classes
- Divisão estratificada em treino / validação / teste
- Normalização via `MinMaxScaler`
- Ajuste de hiperparâmetros com comparação justificada em conjunto de validação
- Avaliação multiclasse via matriz de confusão, acurácia, precisão, recall e F1-score ponderados
- Testes de robustez com Class Masking e inferência Out-of-Distribution (OOD)
- Pipeline próprio de pré-processamento de imagens para inferência com dados reais

## 📁 Estrutura do repositório

```
multiclass-predictive-pipeline/
├── data/
│   ├── mnist_X.npy                 # cache local dos dados do MNIST (gerado na primeira execução)
│   ├── mnist_y.npy                 # cache local dos rótulos do MNIST (gerado na primeira execução)
│   └── imagens_proprias/           # dígitos manuscritos (0-9) desenhados pela autora
├── multiclass-predictive-pipeline.ipynb   # notebook principal com todo o pipeline
├── requirements.txt                 # dependências do projeto
└── README.md
```

> Observação: os arquivos `.npy` do MNIST não são versionados no repositório (dataset gerado automaticamente na primeira execução). As imagens manuscritas próprias, usadas na Fase 5.3, estão versionadas em `data/imagens_proprias/`.

## ⚙️ Como executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/<seu-usuario>/multiclass-predictive-pipeline.git
   cd multiclass-predictive-pipeline
   ```

2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Abra o notebook no VS Code ou Jupyter:
   ```bash
   jupyter notebook multiclass-predictive-pipeline.ipynb
   ```

5. Execute as células em ordem (ou use "Run All"). Na primeira execução, o dataset MNIST será baixado via `fetch_openml` e armazenado localmente em `data/` para acelerar execuções futuras.

## 📊 Resultados obtidos

| Modelo         | Acurácia | Precision (ponderada) | Recall (ponderado) | F1-Score (ponderado) | Tempo de treino |
|----------------|----------|------------------------|---------------------|------------------------|------------------|
| KNN            | 0.9722   | 0.9724                 | 0.9722              | 0.9722                 | 0.04s            |
| Random Forest  | 0.9648   | 0.9648                 | 0.9648              | 0.9648                 | 5.41s            |
| MLP            | 0.9736   | 0.9739                 | 0.9736              | 0.9737                 | 33.49s           |

O **MLP** obteve a melhor acurácia entre os três, seguido de perto pelo **KNN** — que, apesar do tempo de treino praticamente nulo, concentra seu custo computacional na etapa de predição. O par de dígitos mais confundido pelos modelos clássicos (KNN e Random Forest) foi **4 e 9**, o que motivou os testes de robustez da Fase 5.

No teste de Class Masking + OOD (dígitos 4 e 9 ocultados do treino), o modelo demonstrou o fenômeno de **"falsa certeza" (overconfidence)**: mesmo diante de dígitos nunca vistos, ele atribuiu confiança considerável a classes conhecidas, principalmente ao dígito 7.

No teste com imagens manuscritas próprias, o modelo acertou **9 de 10 dígitos** desenhados no Paint, errando apenas o dígito 9 (confundido com o 3).

## 🚀 Possíveis melhorias futuras

- Incluir um quarto modelo baseado em redes convolucionais (CNN), que preservam a estrutura espacial 2D das imagens, ao invés de vetorizá-las.
- Implementar busca de hiperparâmetros mais ampla via `GridSearchCV` ou `RandomizedSearchCV`.
- Medir e comparar também o tempo de predição de cada modelo (não só o tempo de treino), para uma análise de custo computacional mais completa.
- Aplicar técnicas de detecção de dados Out-of-Distribution (OOD detection) para mitigar o problema de falsa certeza identificado na Fase 5.
- Ampliar o conjunto de imagens manuscritas próprias, incluindo caligrafia real (papel e caneta) além do traço digital via Paint.

## 🎥 Vídeo de apresentação

O vídeo de apresentação do projeto está disponível em: `<inserir link do Google Drive aqui>`

## 👩‍💻 Autoria

Projeto desenvolvido por Eduarda Echer como parte do Mini-Projeto Avaliativo do Módulo 2 (Semana 05) do curso de Desenvolvimento de IA para Análise Preditiva.
