# Importações com tratamento de erro
try:
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn import metrics
    import os
except ImportError as e:
    print("\nERRO: Bibliotecas não instaladas. Execute no terminal:")
    print("pip install pandas numpy scikit-learn matplotlib seaborn")
    exit()

# Configurações iniciais (substituído plt.style.use por estilo mais recente)
plt.style.use('seaborn-v0_8')
sns.set_palette('coolwarm')
pd.set_option('display.max_columns', None)

def carregar_dados():
    """Carrega e prepara os dados com tratamento robusto de erros"""
    try:
        # Localização do arquivo (na mesma pasta do script)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, 'heart.csv')
        
        # Verifica se o arquivo existe
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Arquivo 'heart.csv' não encontrado em: {current_dir}")
        
        # Tenta ler com diferentes combinações de parâmetros
        for encoding in ['utf-8', 'latin1']:
            try:
                # Primeiro tenta com delimitador de tabulação
                df = pd.read_csv(csv_path, delimiter='\t', encoding=encoding)
                print(f"\nArquivo lido com encoding {encoding} (delimitador: tabulação)")
                break
            except:
                try:
                    # Se falhar, tenta com delimitador padrão (vírgula)
                    df = pd.read_csv(csv_path, delimiter=',', encoding=encoding)
                    print(f"\nArquivo lido com encoding {encoding} (delimitador: vírgula)")
                    break
                except:
                    continue
        else:
            raise ValueError("Não foi possível ler o arquivo com os encodings testados")
        
        print("\nDados carregados com sucesso!")
        print(f"Total de registros: {len(df)}")
        print("\nPrimeiras linhas do dataset:")
        print(df.head())
        
        return df
    
    except Exception as e:
        print(f"\nERRO ao carregar dados: {str(e)}")
        exit()

def preprocessar_dados(df):
    """Preprocessa os dados com verificações de segurança"""
    try:
        print("\nColunas originais disponíveis:")
        print(df.columns.tolist())
        
        # Verifica se as colunas estão separadas corretamente
        if len(df.columns) == 1:
            print("\nAVISO: Apenas uma coluna detectada - problema com delimitador")
            # Tenta dividir a coluna única
            df = df.iloc[:, 0].str.split('\t', expand=True)
            # Define os nomes das colunas baseado na primeira linha
            df.columns = df.iloc[0]
            df = df[1:]
            print("\nColunas após correção:")
            print(df.columns.tolist())
        
        # Lista de colunas para remover (se existirem)
        cols_to_drop = ['oldpeak', 'slp', 'caa', 'thall']
        cols_existentes = [col for col in cols_to_drop if col in df.columns]
        
        if cols_existentes:
            df.drop(cols_existentes, axis=1, inplace=True)
            print(f"\nColunas removidas: {cols_existentes}")
        else:
            print("\nNenhuma das colunas a remover foi encontrada")
        
        # Dicionário de renomeação
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
        
        # Aplica renomeação apenas para colunas existentes
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
        
        # Verificação de colunas essenciais
        colunas_essenciais = ['idade', 'sexo', 'angina', 'colesterol', 'target']
        for col in colunas_essenciais:
            if col not in df.columns:
                raise ValueError(f"Coluna essencial '{col}' não encontrada após processamento")
        
        # Normalização dos dados
        print("\nAplicando transformações nos dados:")
        
        if 'colesterol' in df.columns:
            df['colesterol'] = np.where(df['colesterol'].astype(float) <= 130, 0, 1)
            print("- Colesterol binarizado (<=130 = 0, >130 = 1)")
        
        if 'angina' in df.columns:
            df['angina'] = np.where(df['angina'].astype(float) > 0, 1, 0)
            print("- Angina binarizada (0 = não, 1 = sim)")
        
        if 'pressao_sanguinea' in df.columns:
            df['pressao_sanguinea'] = np.where(df['pressao_sanguinea'].astype(float) <= 120, 0, 1)
            print("- Pressão sanguínea binarizada (<=120 = 0, >120 = 1)")
        
        # Converte todas as colunas para numérico (exceto se for string)
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    pass
        
        print("\nDataset após pré-processamento:")
        print(df.head())
        
        return df
    
    except Exception as e:
        print(f"\nERRO no pré-processamento: {str(e)}")
        exit()

def analise_exploratoria(df):
    """Realiza análise exploratória dos dados"""
    try:
        print("\n=== ANÁLISE EXPLORATÓRIA ===")
        
        # Estatísticas básicas
        print("\nEstatísticas descritivas:")
        print(df.describe())
        
        # Distribuição da variável target
        print("\nDistribuição da variável target:")
        print(df['target'].value_counts(normalize=True))
        
        # Matriz de correlação
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlação')
        plt.tight_layout()
        plt.savefig('matriz_correlacao.png')
        print("\nMatriz de correlação salva como 'matriz_correlacao.png'")
        plt.close()
        
        # Distribuição de idade por target
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='target', y='idade', data=df)
        plt.title('Distribuição de Idade por Status Cardíaco')
        plt.xlabel('Problema Cardíaco (0=Não, 1=Sim)')
        plt.ylabel('Idade')
        plt.tight_layout()
        plt.savefig('idade_por_target.png')
        print("Gráfico de distribuição de idade salvo como 'idade_por_target.png'")
        plt.close()
        
    except Exception as e:
        print(f"\nERRO na análise exploratória: {str(e)}")

def modelagem_dados(df):
    """Realiza a modelagem dos dados"""
    try:
        print("\n=== MODELAGEM DE DADOS ===")
        
        # Verifica se a coluna target existe
        if 'target' not in df.columns:
            raise ValueError("Coluna 'target' não encontrada para modelagem")
        
        # Preparação dos dados
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Divisão treino-teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=50, stratify=y)
        
        print(f"\nDimensões dos conjuntos:")
        print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
        
        # Regressão Logística
        print("\n>>> REGRESSÃO LOGÍSTICA <<<")
        log_model = LogisticRegression(max_iter=1000)
        log_model.fit(X_train, y_train)
        y_pred_log = log_model.predict(X_test)
        
        print("\nRelatório de Classificação:")
        print(metrics.classification_report(y_test, y_pred_log))
        
        print("\nMatriz de Confusão:")
        print(metrics.confusion_matrix(y_test, y_pred_log))
        
        # Regressão Linear
        print("\n>>> REGRESSÃO LINEAR <<<")
        lin_model = LinearRegression()
        lin_model.fit(X_train, y_train)
        y_pred_lin = lin_model.predict(X_test)
        
        print("\nMétricas de Regressão:")
        print(f"MAE: {metrics.mean_absolute_error(y_test, y_pred_lin):.4f}")
        print(f"MSE: {metrics.mean_squared_error(y_test, y_pred_lin):.4f}")
        print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_test, y_pred_lin)):.4f}")
        
        # Coeficientes
        coeff_df = pd.DataFrame({
            'Feature': X.columns,
            'Coeficiente': lin_model.coef_
        }).sort_values('Coeficiente', ascending=False)
        
        print("\nCoeficientes da Regressão Linear:")
        print(coeff_df)
        
    except Exception as e:
        print(f"\nERRO na modelagem: {str(e)}")

def analise_populacional(df):
    """Realiza análises populacionais adicionais"""
    try:
        print("\n=== ANÁLISES POPULACIONAIS ===")
        
        # Idade média
        idade_media = df.groupby('target')['idade'].mean()
        print(f"\nIdade média com problema cardíaco: {idade_media[1]:.2f}")
        print(f"Idade média sem problema cardíaco: {idade_media[0]:.2f}")
        
        # Proporções por sexo
        if 'sexo' in df.columns:
            prop_sexo = df.groupby(['sexo', 'target']).size().unstack()
            prop_sexo = prop_sexo.div(prop_sexo.sum(axis=1), axis=0) * 100
            print("\nProporção de problemas cardíacos por sexo (%):")
            print(prop_sexo)
        
        # Proporções por colesterol
        if 'colesterol' in df.columns:
            prop_col = df.groupby(['colesterol', 'target']).size().unstack()
            prop_col = prop_col.div(prop_col.sum(axis=1), axis=0) * 100
            print("\nProporção de problemas cardíacos por nível de colesterol (%):")
            print(prop_col)
        
    except Exception as e:
        print(f"\nERRO na análise populacional: {str(e)}")

def main():
    """Função principal que orquestra todo o fluxo"""
    print("\n=== ANÁLISE DE DADOS CARDÍACOS ===\n")
    
    # 1. Carregar dados
    df = carregar_dados()
    
    # 2. Pré-processamento
    df = preprocessar_dados(df)
    
    # 3. Análise exploratória
    analise_exploratoria(df)
    
    # 4. Modelagem
    modelagem_dados(df)
    
    # 5. Análises adicionais
    analise_populacional(df)
    
    print("\nProcesso concluído com sucesso!")

if __name__ == "__main__":
    main()