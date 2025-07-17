"""
ANÁLISE DE DADOS CARDÍACOS
Esse projeto foi desenvolido como parte da obtenção de nota do Curso de Data Science Experience
da Universidade Presbiteriana Mackenzie
"""


# Importação das bibliotecas utilizadas com tratamento de erro
try:
    import pandas as pd # Manipulação de dados
    import numpy as np # Cálculos numéricos
    import seaborn as sns # Visualização gráfica
    import matplotlib.pyplot as plt # Gráficos
    from sklearn.model_selection import train_test_split  # Separação dos dados em treino e teste
    from sklearn.linear_model import LogisticRegression, LinearRegression # Modelos de ML
    from sklearn import metrics  # Avaliação dos modelos
    import os # Acesso ao sistema de arquivos
    import warnings # Controle de alertas
    warnings.filterwarnings('ignore') # Ignora alertas que não impactam no código
except ImportError as e:
    print("\nERRO: Bibliotecas nao instaladas. Execute no terminal:")
    print("pip install pandas numpy scikit-learn matplotlib seaborn")
    exit()

# Configurações visuais e funcionais para melhorar a experiência durante a análise
sns.set(style="whitegrid")  # Define o estilo dos gráficos
plt.rcParams['figure.figsize'] = (12, 8) # Tamanho padrão das figuras
pd.set_option('display.max_columns', None) # Mostra todas as colunas nos prints
pd.set_option('display.encoding', 'utf-8') # Define encoding UTF-8 para textos

def carregar_dados():
    """Função que carrega o dataset cardíaco e trata problemas comuns de leitura de arquivos."""
    try:
        # Localização do arquivo
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, 'heart.csv')
        
        # Verifica se o arquivo existe
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Arquivo 'heart.csv' nao encontrado em: {current_dir}")
        
        # Verifica se o separador é vírgula ou tabulação
        with open(csv_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            delimiter = '\t' if '\t' in first_line else ','
        
        # Carrega os dados
        df = pd.read_csv(csv_path, delimiter=delimiter, encoding='utf-8')
        
        # Se ainda tiver apenas uma coluna, tenta separar por tabulação
        if len(df.columns) == 1:
            print("\nAVISO: Apenas uma coluna detectada - separando por tabulação")
            df = df.iloc[:, 0].str.split('\t', expand=True)
            # Define os nomes das colunas baseado na primeira linha
            df.columns = df.iloc[0]
            df = df[1:]
        
        print("\nDados carregados com sucesso!")
        print(f"Total de registros: {len(df)}")
        print("\nPrimeiras linhas do dataset:")
        print(df.head())
        
        return df
    
    except Exception as e:
        print(f"\nERRO ao carregar dados: {str(e)}")
        exit()

def preprocessar_dados(df):
    """Função responsável por tratar e normalizar os dados antes da modelagem."""
    try:
        # Verifica e converte tipos de dados
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
        
        # Remove colunas que não existem sem erro
        cols_to_drop = ['oldpeak', 'slp', 'caa', 'thall']
        cols_existentes = [col for col in cols_to_drop if col in df.columns]
        
        if cols_existentes:
            df.drop(cols_existentes, axis=1, inplace=True)
            print(f"\nColunas removidas: {cols_existentes}")
        
        # Renomeia colunas (apenas as que existem)
        rename_map = {
            'age': 'idade',
            'sex': 'sexo',
            'cp': 'angina',
            'chol': 'colesterol',
            'thalachh': 'max_ecg',
            'restecg': 'min_ecg',
            'fbs': 'diabetes',
            'trtbps': 'pressao_sanguinea',
            'exng': 'exerc_induz_angina',
            'output': 'target'
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
        
        # Normalização dos dados
        print("\nAplicando transformacoes nos dados:")
        
        # Binariza colunas com base em regras definidas na análise
        if 'colesterol' in df.columns:
            df['colesterol'] = np.where(df['colesterol'] <= 130, 0, 1)
            print("- Colesterol binarizado (<=130 = 0, >130 = 1)")
        
        if 'angina' in df.columns:
            df['angina'] = np.where(df['angina'] > 0, 1, 0)
            print("- Angina binarizada (0 = nao, 1 = sim)")
        
        if 'pressao_sanguinea' in df.columns:
            df['pressao_sanguinea'] = np.where(df['pressao_sanguinea'] <= 120, 0, 1)
            print("- Pressao sanguinea binarizada (<=120 = 0, >120 = 1)")
        
        if 'max_ecg' in df.columns:
            df['max_ecg'] = np.where(df['max_ecg'] <= 164, 0, 1)
            print("- Frequencia cardiaca maxima binarizada (<=164 = 0, >164 = 1)")
        
        print("\nDataset apos pre-processamento:")
        print(df.head())
        
        return df
    
    except Exception as e:
        print(f"\nERRO no pre-processamento: {str(e)}")
        exit()

def analise_exploratoria(df):
    """RRealiza análise estatística e visual dos dados para entender os padrões."""
    try:
        print("\n=== ANALISE EXPLORATORIA ===")
        
        # Média de idade por classe de risco
        print("\nMédias de idade:")
        idade_media_target1 = df.query('target == 1')['idade'].mean()
        idade_media_target0 = df.query('target == 0')['idade'].mean()
        print(f"Média de idade com problema cardiaco: {idade_media_target1:.2f}")
        print(f"Média de idade sem problema cardiaco: {idade_media_target0:.2f}")
        
        # Porcentagens relacionadas à diabetes
        total = len(df)
        perc_diabetes0 = len(df.query('diabetes == 0 and target == 1')) / total * 100
        perc_diabetes1 = len(df.query('diabetes == 1 and target == 1')) / total * 100
        print("\nPorcentagem de pessoas com/sem diabetes e problemas cardiacos:")
        print(f"Sem diabetes e com problemas: {perc_diabetes0:.2f}%")
        print(f"Com diabetes e com problemas: {perc_diabetes1:.2f}%")
        
        # Porcentagens de colesterol
        perc_colesterol1 = len(df.query('colesterol == 1 and target == 1')) / total * 100
        perc_colesterol0 = len(df.query('colesterol == 0 and target == 1')) / total * 100
        print("\nPorcentagem de pessoas com colesterol alto/ideal e problemas cardiacos:")
        print(f"Colesterol alto e com problemas: {perc_colesterol1:.2f}%")
        print(f"Colesterol ideal e com problemas: {perc_colesterol0:.2f}%")
        
        # Angina
        perc_angina1 = len(df.query('angina == 1 and target == 1')) / total * 100
        perc_angina0 = len(df.query('angina == 0 and target == 1')) / total * 100
        print("\nPorcentagem de pessoas com/sem angina e problemas cardiacos:")
        print(f"Com angina e com problemas: {perc_angina1:.2f}%")
        print(f"Sem angina e com problemas: {perc_angina0:.2f}%")
        
        # Sexo
        perc_sexo1 = len(df.query('sexo == 1 and target == 1')) / total * 100
        perc_sexo0 = len(df.query('sexo == 0 and target == 1')) / total * 100
        print("\nPorcentagem de homens/mulheres com problemas cardiacos:")
        print(f"Homens com problemas: {perc_sexo1:.2f}%")
        print(f"Mulheres com problemas: {perc_sexo0:.2f}%")
        
        # Frequência cardíaca máxima para:
            # Mulheres
        perc_mulher_abaixo = len(df.query('max_ecg <= (226 - idade) and target == 1 and sexo == 0')) / total * 100
        perc_mulher_acima = len(df.query('max_ecg >= (226 - idade) and target == 1 and sexo == 0')) / total * 100
        print("\nMulheres com batimentos abaixo/acima do maximo esperado:")
        print(f"Abaixo do maximo: {perc_mulher_abaixo:.2f}%")
        print(f"Acima do maximo: {perc_mulher_acima:.2f}%")
        
            # Homens
        perc_homem_abaixo = len(df.query('max_ecg <= (220 - idade) and target == 1 and sexo == 1')) / total * 100
        perc_homem_acima = len(df.query('max_ecg >= (220 - idade) and target == 1 and sexo == 1')) / total * 100
        print("\nHomens com batimentos abaixo/acima do maximo esperado:")
        print(f"Abaixo do maximo: {perc_homem_abaixo:.2f}%")
        print(f"Acima do maximo: {perc_homem_acima:.2f}%")
        
        # Matriz de correlação
        plt.figure(figsize=(12, 8))
        heat = df.corr()
        sns.heatmap(heat, cmap=['#464AF0','#487BFA','#4DA0E3','#48D9FA','white'], annot=True)
        plt.title("Matriz de Correlacao", size=16, pad=20)
        plt.tight_layout()
        plt.savefig('matriz_correlacao.png')
        plt.close()
        print("\nMatriz de correlacao salva como 'matriz_correlacao.png'")
        
    except Exception as e:
        print(f"\nERRO na analise exploratoria: {str(e)}")

def modelagem_dados(df):
    """Realiza a modelagem dos dados"""
    try:
        print("\n=== MODELAGEM DE DADOS ===")
        
        # Verifica se a coluna target existe
        if 'target' not in df.columns:
            raise ValueError("Coluna 'target' nao encontrada para modelagem")
        
        # Preparação dos dados
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Regressão Linear
        print("\n>>> REGRESSAO LINEAR <<<")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=50)
        
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        
        # Coeficientes
        coeff = pd.DataFrame(modelo.coef_, X.columns, columns=['Coeficiente'])
        print("\nCoeficientes da Regressao Linear:")
        print(coeff)
        
        # Métricas
        predictions = modelo.predict(X_test)
        print('\nMetricas de Regressao:')
        print('MAE:', metrics.mean_absolute_error(y_test, predictions))
        print('MSE:', metrics.mean_squared_error(y_test, predictions))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
        
        # Regressão Logística
        print("\n>>> REGRESSAO LOGISTICA <<<")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=101)
        
        logmodel = LogisticRegression(solver='lbfgs', max_iter=1000)
        logmodel.fit(X_train, y_train)
        
        predictions = logmodel.predict(X_test)
        
        print("\nRelatorio de Classificacao:")
        print(metrics.classification_report(y_test, predictions))
        
        print("\nMatriz de Confusao:")
        print(metrics.confusion_matrix(y_test, predictions))
        
        print("\nMatriz de confusao detalhada:")
        print(pd.crosstab(y_test, predictions, rownames=['Real'], 
                         colnames=['Predito'], margins=True, margins_name='Todos'))
        
    except Exception as e:
        print(f"\nERRO na modelagem: {str(e)}")

def main():
    """Função principal que chama todas as etapas do pipeline de Machine Learning."""
    print("\n=== ANALISE DE DADOS CARDIACOS ===\n")
    
    # Carregar dados
    df = carregar_dados()
    
    # Pré-processamento
    df = preprocessar_dados(df)
    
    # Análise exploratória
    analise_exploratoria(df)
    
    # Modelagem
    modelagem_dados(df)
    
    print("\nProcesso concluido com sucesso!")

if __name__ == "__main__":
    main()
    
