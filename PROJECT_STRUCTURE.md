# Estrutura do Projeto ETL Finance

Este documento descreve a organização das pastas e arquivos do projeto `ETL Finance`.

## Visão Geral

O projeto é uma aplicação de Dashboard Financeiro que coleta dados da B3, armazena em banco de dados SQLite, treina modelos de IA (LSTM/MLP) para previsão de preços e gera otimização de carteira (Markowitz).

## Estrutura de Diretórios

```
ETL Finance/
├── data/                   # Armazenamento de dados locais
│   ├── finance.db          # Banco de dados SQLite com cotações, previsões e portfólios
│   ├── transactions.json   # Registro de transações da carteira pessoal
│   └── sectors.json        # Cache de setores dos ativos (evita chamadas repetidas API)
│
├── models/                 # Modelos de Machine Learning treinados (.pkl)
│
├── src/                    # Código fonte da aplicação
│   ├── app.py              # Aplicação principal (Streamlit Frontend)
│   ├── db.py               # Gerenciamento de banco de dados (SQLite)
│   ├── etl.py              # Pipeline de Extração, Transformação e Carga (yfinance -> DB)
│   ├── models.py           # Definição dos modelos de IA (LSTM, MLP)
│   ├── portfolio.py        # Lógica de Otimização de Carteira (Markowitz + Monte Carlo)
│   ├── portfolio_manager.py# Lógica da Carteira Pessoal (Backend)
│   ├── train.py            # Script para treinamento dos modelos de IA
│   └── utils.py            # Funções utilitárias (ex: busca de setores)
│
├── .gitignore              # Arquivos ignorados pelo Git
├── README.md               # Documentação principal
├── requirements.txt        # Dependências do projeto
└── setup.py                # Configuração de instalação do pacote
```

## Descrição dos Módulos Principais

### `src/app.py`
Interface gráfica construída com Streamlit. Possui 5 abas:
1.  **Gráfico de Preços**: Visualização histórica (Candlestick).
2.  **Previsões IA**: Projeção de preços futuros com modelos treinados.
3.  **Carteira Sugerida**: Visualização da Fronteira Eficiente e alocação por setores.
4.  **Dados Brutos**: Tabela com os dados históricos.
5.  **Minha Carteira**: Gerenciamento de compras/vendas e sugestões de rebalanceamento.

### `src/portfolio_manager.py`
Gerencia a carteira do usuário.
-   Salva/Carrega transações em `transactions.json`.
-   Calcula saldo atual, preço médio e PnL (Lucro/Prejuízo).
-   Gera sugestões de compra/venda comparando a carteira atual com a ótima (Markowitz).

### `src/utils.py`
Contém função `get_stock_sector(ticker)` que consulta a API do Yahoo Finance para descobrir o setor de uma empresa e o armazena em cache.

### `src/train.py`
Script responsável por:
1.  Carregar dados do DB.
2.  Treinar modelos LSTM e MLP para cada ativo.
3.  Salvar previsões futuras no DB.
4.  Executar a otimização de portfólio (Markowitz) com base nos retornos previstos.

## Como Executar

1.  **Instalar dependências**:
    ```bash
    pip install .
    ```

2.  **Executar ETL (Coleta de Dados)**:
    ```bash
    etl-finance
    # ou
    python src/etl.py
    ```

3.  **Treinar Modelos e Otimizar**:
    ```bash
    python src/train.py
    ```

4.  **Iniciar o Dashboard**:
    ```bash
    streamlit run src/app.py
    ```
