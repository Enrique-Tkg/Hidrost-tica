# Geracao de Superficie de Casco via Splines Cubicas

## Descricao do Projeto
Este projeto tem como objetivo realizar a representacao geometrica de um casco naval a partir de uma tabela de offsets (offset table), utilizando splines cubicas naturais implementadas manualmente em Python (sem uso de bibliotecas prontas como SciPy).

A metodologia segue os fundamentos apresentados na disciplina de Hidrostatica e Estabilidade, onde a geometria do casco e reconstruida a partir de dados discretos.

## Objetivos
- Converter uma tabela de offsets (Excel) para um formato estruturado (CSV)
- Implementar splines cubicas naturais do zero
- Interpolar curvas:
  - Em cada estacao (z -> y)
  - Ao longo do comprimento (x -> y)
- Preparar os dados para geracao de superficie do casco
- Validar os resultados com testes

## Estrutura do Projeto
```text
Hidrostatica/
|
|-- .venv/
|-- Pre_Processamento/
|   |-- offsettable.xlsx      # Tabela original de offsets
|   `-- offsettable.csv       # Dados convertidos (x, z, y)
|
|-- utils/
|   |-- __init__.py
|   `-- preprocess.py         # Conversao Excel -> CSV
|
|-- tests/
|   |-- __init__.py
|   `-- test_preprocess.py
|
|-- main.py                   # Execucao principal
|-- requirements.txt
|-- README.md
`-- .gitignore
```

## Formato dos Dados
### Entrada (Excel)
A planilha original segue o padrao de engenharia naval:
- Linhas -> estacoes (x)
- Colunas -> waterlines (z)
- Valores -> meia boca (y)

### Saida (CSV)
Apos o pre-processamento, os dados sao convertidos para:

```csv
x,z,y
0,0,0
0,1,1.2
0,2,2.3
5,0,0
5,1,1.5
```

Onde:
- `x` -> posicao longitudinal (estacao)
- `z` -> altura (waterline)
- `y` -> meia boca

## Metodologia
### 1. Pre-processamento
Conversao da tabela Excel para formato estruturado:
- Leitura com `pandas`
- Transformacao da matriz em lista de pontos
- Exportacao para CSV

### 2. Splines Cubicas Naturais
As splines sao construidas conforme:
- Continuidade de funcao
- Continuidade de derivada primeira
- Continuidade de derivada segunda
- Condicoes de contorno:
  - S''(x0) = 0
  - S''(xn) = 0

### 3. Sistema Linear
Para determinar os coeficientes, resolve-se um sistema linear para `g_k = S''(x_k)`, usando `numpy.linalg.solve`.

### 4. Construcao das Curvas
Para cada intervalo:

`S_k(x) = a_k(x - x_k)^3 + b_k(x - x_k)^2 + c_k(x - x_k) + d_k`

### 5. Reconstrucao do Casco
O casco e obtido atraves de:
1. Splines em cada estacao (z -> y)
2. Splines longitudinais (x -> y)
3. Geracao de malha (grid)

## Tecnologias Utilizadas
- Python 3.x
- NumPy (operacoes matriciais)
- Pandas (manipulacao de dados)
- Matplotlib (visualizacao)
- OpenPyXL (leitura de Excel)

## Como Executar
### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Executar pre-processamento
```bash
python utils/preprocess.py
```

O arquivo de entrada deve estar em `Pre_Processamento/offsettable.xlsx` e o arquivo convertido sera salvo em `Pre_Processamento/offsettable.csv`.

### 3. Rodar o programa principal
```bash
python main.py
```

## Testes
Os testes garantem:
- Continuidade da spline
- Interpolacao correta
- Estabilidade numerica

Executar:
```bash
pytest tests/
```

## Possiveis Extensoes
- Geracao de superficie 3D completa
- Exportacao para CAD (IGES/STEP)
- Uso de NURBS
- Interface grafica
- Integracao com CFD

## Observacoes
- O projeto nao utiliza SciPy, visando entendimento completo do metodo numerico.
- A precisao depende da qualidade da tabela de offsets.
- Valores ausentes devem ser tratados no pre-processamento.

## Referencias
- Notas de aula de Hidrostatica e Estabilidade
- Metodos Numericos para Engenharia
- Teoria de Splines

## Autor
Projeto desenvolvido para fins academicos em Engenharia.
