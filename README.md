# ANÁLISE DE DADOS CARDÍACOS

Este projeto foi desenvolvido como parte da obtenção de nota do Curso de Data Science Experience da Universidade Presbiteriana Mackenzie. O objetivo principal é aplicar técnicas de ciência de dados e aprendizado de máquina para prever o risco de insuficiência cardíaca com base em dados clínicos.

## 📁 Conteúdo do Repositório

- [`Insuficiência Cardiaca.pbix`](./Insuficiência%20Cardiaca.pbix): Dashboard do Power BI com visualizações interativas.
- [`README.md`](./README.md): Descrição geral do projeto e instruções de uso.
- [`analise_cardiacos_final.py`](./analise_cardiacos_final.py): Script Python com o pipeline completo de análise e modelagem.
- [`heart.csv`](./heart.csv): Base de dados utilizada para análise e modelagem.
- [`idade_por_target.png`](./idade_por_target.png): Gráfico ilustrando a distribuição de idade por status cardíaco.
- [`matriz_correlacao.png`](./matriz_correlacao.png): Matriz de correlação gerada com Seaborn.

## 💻 Informação Importate
* Certifique-se que a branch esteja na `Master`

## 🎯 Objetivos do Projeto

* **Previsão de Risco:** Desenvolver modelos preditivos capazes de estimar o risco de problemas cardíacos em pacientes.
* **Análise de Padrões:** Identificar e analisar padrões relevantes nos dados clínicos que podem influenciar o diagnóstico de doenças cardíacas.
* **Suporte à Decisão Clínica:** Fornecer uma ferramenta de apoio para triagem e prevenção de casos graves em ambientes clínicos.
* **Aplicação de Técnicas de ML:** Demonstrar a aplicação prática de um pipeline de Machine Learning, desde o carregamento dos dados até a modelagem.

## 🚀 Funcionalidades

O projeto implementa um pipeline de Machine Learning que inclui as seguintes etapas:

1.  **Carregamento e Tratamento de Dados:**
    * Carregamento robusto do dataset `heart.csv`, com detecção automática de delimitador (vírgula ou tabulação).
    * Verificação e conversão de tipos de dados para numéricos.
2.  **Pré-processamento e Engenharia de Atributos Simplificada:**
    * Remoção de colunas não utilizadas (`oldpeak`, `slp`, `caa`, `thall`).
    * Renomeação de colunas para nomes mais intuitivos (ex: `age` para `idade`, `chol` para `colesterol`).
    * **Binarização de variáveis chave:** `colesterol`, `angina` (tipo de dor no peito), `pressao_sanguinea` e `max_ecg` (frequência cardíaca máxima) são transformadas em variáveis binárias (0 ou 1) com base em thresholds predefinidos no código.
3.  **Análise Exploratória dos Dados (EDA):**
    * Cálculo de médias de idade por status cardíaco.
    * Determinação de porcentagens de pacientes com/sem diabetes, colesterol alto/ideal, presença/ausência de angina e sexo em relação ao status cardíaco.
    * Análise de batimentos cardíacos máximos esperados para homens e mulheres.
    * Geração de **Matriz de Correlação** (`matriz_correlacao.png`) para visualizar as relações entre as variáveis.
    * **Boxplot de Distribuição de Idade por Status Cardíaco** (analisado no relatório técnico, não gerado pelo script).
4.  **Modelagem de Machine Learning:**
    * Divisão dos dados em conjuntos de treino e teste.
    * Implementação de dois modelos preditivos:
        * **Regressão Linear:** Utilizada para identificar a relação linear entre as variáveis e o `target`.
        * **Regressão Logística:** Aplicada como modelo de classificação para prever a probabilidade de um problema cardíaco (variável `target`).
5.  **Avaliação dos Modelos:**
    * **Para Regressão Linear:** Métricas de regressão (MAE, MSE, RMSE) são calculadas.
    * **Para Regressão Logística:** Relatório de Classificação (Precision, Recall, F1-Score, Acurácia) e Matriz de Confusão detalhada são gerados para avaliar o desempenho do modelo na classificação.

## 📊 Resultados Principais

O modelo de **Regressão Logística** foi destacado no relatório técnico por sua performance no problema de classificação.

* **Acurácia:** Em torno de 85% (conforme relatório técnico).
* **Relatório de Classificação:** Detalhes de Precision, Recall e F1-Score são fornecidos na saída do código.
* **Matriz de Confusão:** Apresentada para análise de falsos positivos e falsos negativos, fundamental para problemas de saúde.

## 🔍 Insights do Negócio

As análises realizadas e o modelo desenvolvido fornecem insights para o departamento médico:

* **Padrões de Risco por Idade:** Análises como o boxplot de idade por status cardíaco indicam que, embora a mediana de pacientes com problema cardíaco possa ser ligeiramente menor, a faixa etária com risco se concentra entre 43 e 60 anos.
* **Fatores Binarizados:** Variáveis como colesterol e pressão sanguínea, após a binarização, podem indicar de forma simplificada a contribuição de níveis altos para o risco cardíaco.
* **Prevenção Focada:** A identificação de padrões e a capacidade preditiva do modelo podem direcionar ações preventivas para grupos de risco.
* **Suporte ao Diagnóstico:** O modelo pode apoiar a triagem de pacientes, indicando aqueles com maior probabilidade de desenvolver problemas cardíacos.
* **Otimização de Recursos:** A antecipação de diagnósticos pode levar à redução de custos hospitalares e à melhoria do atendimento preventivo.

## 📦 Estrutura do Projeto


├── analise_cardiacos_final.py  # Script principal com o pipeline de ML

├── heart.csv                  # Dataset utilizado no projeto

├── matriz_correlacao.png      # Gráfico de matriz de correlação gerado pelo script

├── idade_por_target.png       # Exemplo de boxplot de idade

└── README.md                  # Este arquivo


## 🛠️ Tecnologias Utilizadas

* **Python 3.x**
* **Bibliotecas Principais:**
    * `pandas`: Manipulação e análise de dados.
    * `numpy`: Operações numéricas.
    * `scikit-learn`: Modelos de Machine Learning (`LogisticRegression`, `LinearRegression`, `train_test_split`, `metrics`).
    * `matplotlib`: Geração de gráficos estáticos.
    * `seaborn`: Geração de gráficos estatísticos (Matriz de Correlação).
    * `os`: Interação com o sistema de arquivos.
    * `warnings`: Controle de alertas.

### Requisitos do Sistema

* Python 3.x instalado.
* Pelo menos 4GB de RAM (recomendado para datasets maiores).

## 🔧 Instalação

Para configurar o ambiente e executar o projeto:

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/pygaudiello/Data_Science_Experience.git](https://github.com/pygaudiello/Data_Science_Experience.git)
    cd Data_Science_Experience
    ```
2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    # No Linux/macOS:
    source venv/bin/activate
    # No Windows:
    .\venv\Scripts\activate
    ```
3.  **Instale as dependências:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

## 💻 Como Usar

Para executar o pipeline completo de análise e modelagem:

1.  Certifique-se de que o arquivo `heart.csv` esteja no mesmo diretório que o script `analise_cardiacos_final.py`.
2.  Execute o script Python a partir do terminal:
    ```bash
    python analise_cardiacos_final.py
    ```
    O script irá carregar os dados, pré-processá-los, realizar a análise exploratória (imprimindo resultados no console e gerando `matriz_correlacao.png`) e, finalmente, treinar e avaliar os modelos de Regressão Linear e Regressão Logística, exibindo os resultados no console.

---

⭐ Projeto desenvolvido para conclusão de matéria na Universidade Presbiteriana Mackenzie.
