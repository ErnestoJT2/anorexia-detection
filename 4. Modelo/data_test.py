import pandas as pd
import shutil
import os

def cargar_data_final(ruta_csv: str) -> pd.DataFrame:
    """
    Carga el CSV completo de data_final.csv y devuelve el DataFrame.
    """
    return pd.read_csv(ruta_csv)

def guardar_data_completa(df: pd.DataFrame, ruta_salida: str):
    """
    Guarda el DataFrame completo en la ruta indicada.
    """
    df.to_csv(ruta_salida, index=False)

def copiar_data_sin_procesar(ruta_origen: str, ruta_destino: str):
    """
    Copia el archivo CSV sin modificarlo de ruta_origen a ruta_destino.
    Útil si no quieres cargarlo en memoria, solo copiado directo.
    """
    shutil.copyfile(ruta_origen, ruta_destino)

def main():
    # 1) Ruta del data_final.csv original
    ruta_data_final_original = r"C:\Users\ernes\OneDrive\Escritorio\Reto Final\1. Preprocesamiento de Texto\data_final.csv"
    
    # 2) Ruta donde guardaremos la copia completa en 4. Modelo
    ruta_data_destino = r"C:\Users\ernes\OneDrive\Escritorio\Reto Final\4. Modelo\data_all.csv"
    
    # Opción A: copiar directamente el CSV sin leer (mantiene formatos idénticos)
    print("Copiando data_final.csv completo a 4. Modelo/data_all.csv ...")
    copiar_data_sin_procesar(ruta_data_final_original, ruta_data_destino)
    
    # Si prefieres cargar y volver a guardar (por si quisieras validarlo o reindexar), podrías usar:
    # df = cargar_data_final(ruta_data_final_original)
    # print("Guardando nuevamente a data_all.csv (recreado desde DataFrame)...")
    # guardar_data_completa(df, ruta_data_destino)
    
    # 3) Informar por consola
    df_check = pd.read_csv(ruta_data_destino)
    print(f"data_all.csv generado en: {ruta_data_destino}")
    print(f"  → Total de ejemplos copiados: {len(df_check)}")
    print("Distribución de clases en data_all.csv:")
    print(df_check["class"].value_counts())

if __name__ == "__main__":
    main()
