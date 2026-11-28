# Modelo de aprendizado de máquina explicável para predição de burnout em profissionais da saúde

## 1. Sobre o projeto

Este Trabalho de Conclusão de Curso (TCC) de Ciência da Computação da Universidade de Santa Cruz do Sul (UNISC) investiga o uso de características demográficas, profissionais e organizacionais para identificar profissionais da saúde que atendem ao critério de alto risco de burnout. O repositório reúne o notebook dos experimentos e os scripts para gerar previsões com explicações individuais a partir de um CSV.

## 2. Objetivos

- Estimar a probabilidade de um profissional da saúde atender ao critério de alto risco de burnout a partir de características demográficas, profissionais e respostas ao Safety Attitudes Questionnaire (SAQ).
- Avaliar o desempenho dos modelos de classificação e selecionar a configuração com maior ROC-AUC média na validação cruzada.
- Interpretar as previsões com SHAP (*SHapley Additive exPlanations*), identificando os atributos mais relevantes para o modelo e suas contribuições individuais.

## 3. Conjunto de dados

O conjunto de dados, disponibilizado por Denning et al. (2021), contém **3.537 registros** de um estudo transversal realizado com profissionais da saúde durante a pandemia de COVID-19, principalmente no Reino Unido, na Polônia e em Singapura. A coluna `Country` também contém 37 registros classificados na categoria `Other`.

A variável-alvo foi definida a partir da coluna `Burnout?`, derivada das dimensões de exaustão e distanciamento do *Oldenburg Burnout Inventory* (OLBI). O trabalho preserva a classificação original disponibilizada pelos autores.

| Classe | Códigos originais | Definição | Registros |
| --- | --- | --- | ---: |
| 0 | `0`, `1` ou `2` | Não atende ao critério combinado de alto risco (nenhuma dimensão elevada ou apenas uma elevada) | 1.173 (33,16%) |
| 1 | `3` | Atende ao critério combinado de alto risco (exaustão e distanciamento elevados) | 2.364 (66,84%) |

As informações sobre a fonte e a licença do conjunto de dados estão disponíveis na seção 14.

## 4. Metodologia

1. **Preparação.** Os dados foram submetidos à limpeza, normalização e agrupamento de categorias. A pipeline aplicou imputação de valores ausentes, padronização dos atributos quantitativos e One-Hot Encoding dos categóricos.

2. **Divisão e validação.** A base foi dividida de forma estratificada em **2.829 registros de treinamento e 708 de teste**, com semente 42. Os hiperparâmetros foram ajustados por busca em grade com validação cruzada estratificada de cinco partições, evitando vazamento de dados no pré-processamento.

3. **Comparação e seleção.** Foram avaliados Regressão Logística, Random Forest, XGBoost e Multilayer Perceptron (MLP), considerando duas representações do SAQ, com e sem ponderação das classes. O **XGBoost com os itens do SAQ, sem ponderação**, apresentou a maior ROC-AUC média (0,80137), sendo selecionado com 46 preditores. Os atributos do OLBI foram excluídos para evitar vazamento de dados.

4. **Explicabilidade.** O Tree SHAP foi utilizado para interpretar as previsões do modelo selecionado, utilizando gráficos de barras e *beeswarm* para explicações globais e *waterfall* para explicações individuais.

## 5. Resultados experimentais

As métricas abaixo foram calculadas nos **708 registros do conjunto de teste do notebook**. O *baseline* prevê sempre a classe mais frequente no treino, a classe 1, e atribui a todos os registros uma pontuação constante igual à proporção dessa classe no treinamento.

| Métrica | XGBoost | *Baseline* |
| --- | ---: | ---: |
| ROC-AUC | 0,792 | 0,500 |
| Acurácia | 72,9% | 66,8% |
| Acurácia balanceada | 66,0% | 50,0% |
| Recall da classe 1 | 86,5% | 100,0% |
| Recall da classe 0 | 45,5% | 0,0% |
| F1 macro | 0,668 | 0,401 |

O XGBoost acertou **516 registros**, com **128 falsos positivos** e **64 falsos negativos**. O *baseline* identificou todos os registros da classe 1 ao prever essa classe para todo o teste e errou os 235 registros da classe 0. O modelo apresentou maior capacidade de identificar a classe 1 do que a classe 0. Os resultados completos estão no [notebook dos experimentos](notebooks/burnout_prediction.ipynb).

## 6. Estrutura do projeto

```text
explainable-burnout-prediction/
├── data/
│   ├── training/
│   │   └── denning_dataset.csv
│   └── inputs/
│       └── synthetic_inputs.csv
├── models/                      # gerada localmente no treino
│   └── burnout_model.pkl
├── notebooks/
│   └── burnout_prediction.ipynb
├── outputs/                     # gerada localmente na previsão
├── references/
│   ├── denning_article.pdf
│   └── denning_questionnaires.docx
├── preprocess.py
├── train.py
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

- `preprocess.py` valida e prepara as entradas e define as transformações da pipeline.
- `train.py` treina a pipeline selecionada com a base completa e a salva em `models/burnout_model.pkl`.
- `predict.py` recebe um CSV e gera as previsões e as explicações individuais.
- `notebooks/` reúne a análise dos dados, a comparação dos modelos e a avaliação experimental.
- `references/` reúne o artigo de origem e os materiais dos questionários.
- `requirements.txt` fixa as versões das dependências.

## 7. Tecnologias utilizadas

| Tecnologia | Uso |
| --- | --- |
| Python | Scripts e notebook |
| pandas e NumPy | Leitura, tratamento e cálculo com dados tabulares |
| scikit-learn | Pré-processamento, treinamento, validação cruzada e métricas |
| XGBoost | Classificador final |
| SHAP e Matplotlib | Explicações e visualizações |
| IPython e ipykernel | Execução do notebook |

## 8. Instalação e configuração

O projeto foi executado com **Python 3.12.1**. A partir da raiz do repositório, use o PowerShell para criar o ambiente e instalar as dependências.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Para executar o notebook, selecione o kernel Python da `.venv` e use a pasta `notebooks/` como diretório de trabalho.

## 9. Como utilizar

Execute os comandos na raiz do projeto, com o ambiente virtual ativo.

**Treinar e salvar o modelo**

```powershell
python train.py
```

O script reajusta o pré-processamento e treina o XGBoost com os hiperparâmetros selecionados no notebook, utilizando os 3.537 registros da base, sem uma nova busca de hiperparâmetros. A pipeline completa é salva em `models/burnout_model.pkl`. 

**Gerar previsões e explicações**

```powershell
python predict.py data/inputs/synthetic_inputs.csv
```

Substitua o caminho do exemplo pelo CSV que você deseja processar. Se o modelo salvo estiver ausente, o script valida a entrada, treina o modelo e segue para a previsão. O terminal informa o andamento e o caminho da pasta de resultados.

## 10. Formato dos dados de entrada

Use um CSV em **UTF-8**, separado por vírgulas, com cabeçalho e pelo menos um registro. Os **46 cabeçalhos** devem ter os mesmos nomes do arquivo de exemplo. A ordem das colunas é livre, e o script usa somente as colunas previstas no esquema.

| Campos obrigatórios | Valores esperados |
| --- | --- |
| `gender`, `role`, `Country` e `base` | Categorias textuais da base original, como `Female`, `Nurse`, `UK` e `medicine` |
| `Have you been redeployed to a different area, hospital, or specialty?` | `Yes` ou `No` |
| `If so, to which specialty or area have you been redeployed?` | Destino da realocação, como `Intensive Care` |
| `Have you treated a COVID-19 positive patient in the past week?` | Categorias textuais da base, como `Yes`, `No`, `Don't know` ou `Not applicable (e.g. do not treat patients)` |
| `How many days did you spend at work in the last 7 days?` | Número inteiro de `0` a `7` |
| `What was the average duration of your shift?` | Horas positivas e finitas, como `7.5`, ou uma faixa textual da base, como `8 to less than 12 hours` |
| `If redeployed: how would you describe the training you received in preparation for redeployment?` | `Very poor`, `Poor`, `Neutral`, `Good`, `Very good` ou `No training received` |
| `If redeployed: how would you describe the support you received during redeployment?` | `Very poor`, `Poor`, `Neutral`, `Good`, `Very good` ou `No support received` |
| `SAQ.1` a `SAQ.36`, exceto `SAQ.8` | `0` ou `1`, conforme a codificação da base, incluindo os itens com pontuação invertida |

Com realocação `No`, deixe destino, treinamento e suporte vazios ou use `NAO_SE_APLICA`. O sistema preenche esses três campos automaticamente. Com realocação `Yes`, informe respostas aplicáveis aos três campos. Após a preparação, todos os campos devem estar preenchidos. Se restar um nulo, o script interrompe o processamento e informa o primeiro registro incompleto e suas colunas ausentes.

Envie os atributos no formato da base original. A pipeline aplica as transformações ajustadas no treinamento. A coluna-alvo `Burnout?` e os escores do OLBI não são necessários para realizar previsões. Categorias desconhecidas que passam pela preparação geram um aviso. Na codificação One-Hot, todas as colunas correspondentes àquele atributo recebem zero, o que pode limitar a confiabilidade da previsão.

## 11. Dados sintéticos

O arquivo `data/inputs/synthetic_inputs.csv` contém dez registros fictícios, gerados artificialmente com o objetivo exclusivo de demonstrar o funcionamento do sistema de previsão e explicabilidade. Esses registros não são utilizados para avaliar o desempenho do modelo ou extrair conclusões científicas.

## 12. Arquivos de saída

Cada execução cria uma pasta com o nome do CSV de entrada e a data e hora da execução. Caso exista uma pasta com o mesmo nome, um sufixo numérico é acrescentado para preservar as saídas anteriores.

```text
outputs/
└── synthetic_inputs_AAAA-MM-DD_HHMMSS/
    ├── predictions.csv
    └── explanations/
        ├── registro_001.png
        ├── registro_002.png
        └── ...
```

O `predictions.csv` contém o nome do arquivo de entrada (`arquivo_entrada`), o identificador sequencial do registro (`registro`), a classe prevista (`classe_prevista`), a estimativa para a classe 1 (`probabilidade_classe_1`, de 0 a 1) e o caminho relativo do gráfico (`grafico`). Os identificadores seguem a ordem dos registros no CSV de entrada.

Cada gráfico *waterfall* mostra como os atributos daquele registro aumentaram ou reduziram a saída do modelo. As contribuições SHAP estão em **log-odds**, a escala interna do classificador. A probabilidade consta separadamente no CSV.

## 13. Limitações

- A base de dados é transversal, ou seja, representa as condições dos profissionais no momento da coleta. O modelo não permite prever o surgimento de burnout no futuro, o que exigiria dados longitudinais.
- Os dados foram coletados durante a pandemia, em outros países, por amostragem de conveniência. Isso limita a representatividade da amostra e a generalização dos resultados para o Brasil e outros contextos. Além disso, o modelo não foi validado em uma população externa.
- O uso de questionários respondidos pelos próprios participantes, a presença de valores ausentes, a binarização dos itens do SAQ e o agrupamento de categorias podem limitar a qualidade e o nível de detalhamento das informações.
- A presença de registros duplicados pode influenciar os resultados da avaliação, pois registros idênticos podem estar presentes tanto no conjunto de treinamento quanto no de teste.
- O SHAP permite identificar a contribuição dos atributos para as previsões, mas não estabelecer relações de causa e efeito. As explicações geradas também não foram avaliadas por profissionais da saúde.
- O sistema possui finalidade acadêmica e demonstrativa, não devendo ser utilizado como ferramenta de diagnóstico ou substituir a avaliação de profissionais qualificados.

## 14. Fonte e licença dos dados

O conjunto de dados utilizado neste projeto foi disponibilizado por Denning et al. (2021) como material suplementar S2 do artigo publicado na *PLOS ONE*. A publicação está licenciada sob a Creative Commons Attribution 4.0 International (CC BY 4.0).

Os dados foram submetidos a procedimentos de limpeza, transformação e pré-processamento para o desenvolvimento dos modelos de aprendizado de máquina.

**Referência:**

DENNING, Max et al. Determinants of burnout and other aspects of psychological well-being in healthcare workers during the Covid-19 pandemic: a multinational cross-sectional study. *PLOS ONE*, v. 16, n. 4, e0238666, 2021.

- Artigo original: https://doi.org/10.1371/journal.pone.0238666
- Conjunto de dados (S2): https://doi.org/10.1371/journal.pone.0238666.s002
- Licença: https://creativecommons.org/licenses/by/4.0/

## 15. Autoria

- **Autor:** Brendon Kappel.
- **Orientadora:** Profa. Dra. Rejane Frozza.
- **Colaboradora:** Profa. Dra. Janine Koepp.
- **Curso:** Ciência da Computação.
- **Instituição:** Universidade de Santa Cruz do Sul (UNISC).