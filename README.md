# Geracao de Superficie de Casco via Splines Cubicas

## Descricao
Este projeto reconstrui a geometria de um casco naval a partir de uma tabela de offsets.
A proposta e converter os dados da planilha para CSV e preparar a base para interpolacao com splines cubicas naturais.

## Estrutura Atual
```text
Hidrostatica/
|
|-- .venv/
|-- Pre_Processamento/
|   |-- offsettable.xlsx
|   `-- offsettable.csv
|
|-- utils/
|   |-- __init__.py
|   `-- preprocess.py
|
|-- tests/
|   |-- __init__.py
|   `-- test_preprocess.py
|
|-- main.py
|-- requirements.txt
|-- README.md
`-- .gitignore
```

## Dependencias
As dependencias estao em `requirements.txt`:
- numpy
- pandas
- matplotlib
- openpyxl

Instalacao:
```bash
pip install -r requirements.txt
```

## Fluxo de Uso
1. Garanta que o arquivo de entrada esteja em `Pre_Processamento/offsettable.xlsx`.
2. Execute o pre-processamento:
```bash
python utils/preprocess.py
```
3. O arquivo convertido sera salvo em `Pre_Processamento/offsettable.csv`.
4. Execute o programa principal:
```bash
python main.py
```

## Testes
Para rodar os testes:
```bash
pytest tests/
```

## Observacoes
- A estrutura foi reorganizada para separar dados (`Pre_Processamento`), utilitarios (`utils`) e testes (`tests`).
- O script `utils/preprocess.py` ja esta configurado para os novos caminhos.
