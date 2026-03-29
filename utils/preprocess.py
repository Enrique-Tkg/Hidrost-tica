from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_XLSX = BASE_DIR / "Pre_Processamento" / "offsettable.xlsx"
OUTPUT_CSV = BASE_DIR / "Pre_Processamento" / "offsettable.csv"


def convert_offset_table() -> None:
    """Converte a planilha de offsets para CSV no formato x,z,y."""
    df = pd.read_excel(INPUT_XLSX, header=None)
    z_values = df.iloc[0, 1:].values

    dados = []
    for i in range(1, df.shape[0]):
        x = df.iloc[i, 0]
        for j in range(1, df.shape[1]):
            y = df.iloc[i, j]
            z = z_values[j - 1]
            if pd.notna(y):
                dados.append([x, z, y])

    novo_df = pd.DataFrame(dados, columns=["x", "z", "y"])
    novo_df.to_csv(OUTPUT_CSV, index=False)
    print(f"CSV gerado em: {OUTPUT_CSV}")


if __name__ == "__main__":
    convert_offset_table()
