from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_XLSX = BASE_DIR / "Pre_Processamento" / "offsettable.xlsx"
OUTPUT_CSV = BASE_DIR / "Pre_Processamento" / "offsettable.csv"


def convert_offset_table() -> None:
    """Converte a planilha de offsets para CSV no formato x,z,y."""
    df = pd.read_excel(INPUT_XLSX, header=None)
    
    # Extrair valores de Z da linha 2 (headers com waterlines)
    z_values = df.iloc[2, 1:].values
    
    # Mapear nomes de waterlines para valores numéricos
    z_numeric = []
    for z in z_values:
        if z == 'Base Line':
            z_numeric.append(0)
        elif isinstance(z, str) and 'W.L' in str(z):
            # Extrair número de "1.0 W.L", "2.0 W.L", etc.
            try:
                z_val = float(str(z).split()[0])
                z_numeric.append(z_val)
            except:
                z_numeric.append(None)
        else:
            z_numeric.append(z)
    
    dados = []
    # Começar a partir da linha 3 (onde estão os dados reais)
    for i in range(3, df.shape[0]):
        x = df.iloc[i, 0]
        
        # Ignorar linhas com valores inválidos de X
        if pd.isna(x) or x == '-':
            continue
        
        for j in range(1, df.shape[1]):
            y = df.iloc[i, j]
            z = z_numeric[j - 1]
            
            # Converter '-' para NaN e ignorar valores nulos
            if isinstance(y, str) and y == '-':
                continue
            
            # Validar que Z é numérico
            if pd.isna(z) or isinstance(z, str):
                continue
            
            if pd.notna(y) and pd.notna(z):
                try:
                    y_val = float(y)
                    dados.append([float(x), z, y_val])
                except:
                    continue

    novo_df = pd.DataFrame(dados, columns=["x", "z", "y"])
    novo_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ CSV gerado em: {OUTPUT_CSV}")
    print(f"✓ Total de pontos válidos: {len(novo_df)}")
    print(f"\nPrimeiras 10 linhas:")
    print(novo_df.head(10).to_string(index=False))
    print(f"\nÚltimas 10 linhas:")
    print(novo_df.tail(10).to_string(index=False))


if __name__ == "__main__":
    convert_offset_table()
