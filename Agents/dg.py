import os
import sys
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from datetime import datetime

# =========================
# CONFIGURACIÓN
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("⚠️ Debes definir GROQ_API_KEY en .env")

# =========================
# LECTURA DE DATOS
# =========================
def cargar_matriz(fuente, sheet_name=0):
    """Carga CSV, Excel, JSON o DataFrame a un DataFrame de pandas."""
    if isinstance(fuente, pd.DataFrame):
        return fuente

    if isinstance(fuente, (bytes, bytearray)):
        try:
            return pd.read_excel(fuente, sheet_name=sheet_name)
        except Exception:
            return pd.read_csv(fuente)

    if isinstance(fuente, str) and os.path.isfile(fuente):
        ext = os.path.splitext(fuente)[1].lower()
        if ext == ".csv":
            return pd.read_csv(fuente)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(fuente, sheet_name=sheet_name)
        elif ext == ".json":
            return pd.DataFrame(json.load(open(fuente, encoding="utf-8")))
        else:
            raise ValueError(f"Formato no soportado: {ext}")

    if isinstance(fuente, (dict, list)):
        return pd.DataFrame(fuente)

    raise ValueError("No se pudo interpretar la fuente de datos.")

# =========================
# PERFILADO AVANZADO
# =========================
def perfilar_matriz(df):
    """Genera un perfil detallado de la matriz de datos."""
    perfil = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_count": df.isnull().sum().to_dict(),
        "missing_percent": (df.isnull().mean()*100).round(2).to_dict(),
        "duplicated_rows": int(df.duplicated().sum()),
        "unique_counts": df.nunique().to_dict(),
        "sample": df.head(5).to_dict(orient="records")
    }

    # Columnas numéricas
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        perfil["numeric_summary"] = df[num_cols].describe(include="all").T.to_dict()
        # Outliers con IQR
        outliers = {}
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers[col] = int(((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum())
        perfil["numeric_outliers"] = outliers

    # Columnas categóricas
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    if len(cat_cols) > 0:
        cat_summary = {}
        for col in cat_cols:
            cat_summary[col] = {
                "top_values": df[col].value_counts().head(5).to_dict(),
                "unique": int(df[col].nunique()),
                "mode": df[col].mode().iloc[0] if not df[col].mode().empty else None
            }
        perfil["categorical_summary"] = cat_summary

    return perfil

# =========================
# ANÁLISIS DE DATOS
# =========================
def analisis_contenido(df):
    contenido = {}

    # Detectar tipos avanzados
    contenido["tipos"] = df.dtypes.astype(str).to_dict()
    contenido["missing_summary"] = {
        "count": df.isnull().sum().to_dict(),
        "percent": (df.isnull().mean()*100).round(2).to_dict()
    }

    # Estadísticas numéricas
    num_cols = df.select_dtypes(include=[np.number]).columns
    contenido["numerical_analysis"] = {}
    for col in num_cols:
        serie = df[col].dropna()
        contenido["numerical_analysis"][col] = {
            "min": serie.min(),
            "max": serie.max(),
            "mean": serie.mean(),
            "median": serie.median(),
            "std": serie.std(),
            "q25": serie.quantile(0.25),
            "q75": serie.quantile(0.75),
            "missing": int(df[col].isnull().sum()),
        }

    # Análisis categórico
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    contenido["categorical_analysis"] = {}
    for col in cat_cols:
        conteo = df[col].value_counts()
        contenido["categorical_analysis"][col] = {
            "top5": dict(conteo.head(5)),
            "unique": int(df[col].nunique()),
            "missing": int(df[col].isnull().sum())
        }

    # Correlaciones numéricas
    if len(num_cols) > 1:
        contenido["correlations"] = df[num_cols].corr().round(3).to_dict()

    # Relaciones categóricas-numéricas (media por categoría)
    relaciones = {}
    for cat in cat_cols:
        for num in num_cols:
            relaciones[f"{cat}__vs__{num}"] = df.groupby(cat)[num].mean().to_dict()
    contenido["cat_num_relations"] = relaciones

    # Temporalidad
    fechas = df.select_dtypes(include=["datetime", "datetimetz"]).columns
    temporalidad = {}
    for col in fechas:
        df_temp = df.copy()
        df_temp["year"] = df[col].dt.year
        df_temp["month"] = df[col].dt.month
        temporalidad[col] = {
            "count_by_year": df_temp["year"].value_counts().sort_index().to_dict(),
            "count_by_month": df_temp["month"].value_counts().sort_index().to_dict()
        }
    if temporalidad:
        contenido["temporal_analysis"] = temporalidad

    return contenido

# =========================
# GENERAR INFORME CON IA
# =========================
def analizar_matriz(fuente, descripcion_db="", temperature=0.3):
    df = cargar_matriz(fuente)
    perfil = perfilar_matriz(df)
    contenido = analisis_contenido(df)

    resumen_perfil = json.dumps(perfil, indent=2, ensure_ascii=False)
    resumen_contenido = json.dumps(contenido, indent=2, ensure_ascii=False)

    prompt_template = PromptTemplate(
        input_variables=["descripcion_db", "perfil", "contenido"],
        template="""
Eres un experto en análisis de datos estratégicos. Analiza el siguiente dataset:

Descripción: {descripcion_db}
Perfil técnico: {perfil}
Contenido y patrones: {contenido}

Genera un informe detallado con hallazgos clave, calidad de datos, tendencias, anomalías, riesgos y recomendaciones.
"""
    )

    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=temperature, api_key=GROQ_API_KEY)
    prompt = prompt_template.format(
        descripcion_db=descripcion_db,
        perfil=resumen_perfil,
        contenido=resumen_contenido
    )
    respuesta = llm.invoke(prompt)

    return {
        "informe": respuesta.content,
        "perfil": perfil,
        "contenido": contenido,
        "fecha": datetime.now().isoformat()
    }

# =========================
# EJECUCIÓN DIRECTA
# =========================
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python dg.py <archivo> <descripcion_db>")
        sys.exit(1)

    archivo = sys.argv[1]
    descripcion_db = sys.argv[2]

    resultado = analizar_matriz(archivo, descripcion_db=descripcion_db)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
