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
# LECTURA DE MATRIZ
# =========================
def cargar_matriz(fuente, sheet_name=0):
    """Carga CSV, Excel o JSON a DataFrame."""
    if isinstance(fuente, pd.DataFrame):
        return fuente

    if isinstance(fuente, (bytes, bytearray)):
        try:
            return pd.read_excel(fuente, sheet_name=sheet_name)
        except Exception:
            return pd.read_csv(fuente)

    if isinstance(fuente, str) and os.path.isfile(fuente):
        ext = os.path.splitext(fuente)[1].lower()
        if ext in [".csv"]:
            return pd.read_csv(fuente)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(fuente, sheet_name=sheet_name)
        elif ext in [".json"]:
            return pd.DataFrame(json.load(open(fuente)))
        else:
            raise ValueError(f"Formato no soportado: {ext}")

    if isinstance(fuente, dict) or isinstance(fuente, list):
        return pd.DataFrame(fuente)

    raise ValueError("No se pudo interpretar la fuente de datos.")

# =========================
# PERFILADO DE MATRIZ
# =========================
def perfilar_matriz(df, sample_rows=5, sample_cols=10):
    perfil = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
    }

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        perfil["numeric_summary"] = df[num_cols].describe().T.to_dict()

    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    perfil["categorical_tops"] = {
        col: df[col].value_counts().head(5).to_dict() for col in cat_cols
    }

    perfil["preview"] = df.head(sample_rows).iloc[:, :sample_cols].to_dict(orient="list")
    return perfil

# =========================
# ANÁLISIS DE CONTENIDO
# =========================
def analisis_contenido(df, sample_size=20):
    """Genera un resumen enriquecido del contenido para análisis estratégico."""
    contenido = {}

    # Muestra aleatoria
    contenido["muestra"] = df.sample(min(sample_size, len(df)), random_state=42).to_dict(orient="records")

    # Distribuciones numéricas
    num_cols = df.select_dtypes(include=[np.number]).columns
    distribuciones = {}
    for col in num_cols:
        distribuciones[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "media": float(df[col].mean()),
            "mediana": float(df[col].median()),
            "desviacion": float(df[col].std()),
            "q25": float(df[col].quantile(0.25)),
            "q75": float(df[col].quantile(0.75)),
            "outliers_estimados": int(
                ((df[col] < (df[col].quantile(0.25) - 1.5*df[col].std())) | 
                 (df[col] > (df[col].quantile(0.75) + 1.5*df[col].std()))).sum()
            )
        }
    contenido["distribuciones_numericas"] = distribuciones

    # Distribuciones categóricas
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    distrib_cat = {}
    for col in cat_cols:
        distrib_cat[col] = {
            "top5": dict(df[col].value_counts().head(5)),
            "unicos": int(df[col].nunique()),
            "moda": str(df[col].mode().iloc[0]) if not df[col].mode().empty else None
        }
    contenido["distribuciones_categoricas"] = distrib_cat

    # Correlaciones entre numéricas
    if len(num_cols) > 1:
        corr_matrix = df[num_cols].corr().round(3).to_dict()
        contenido["correlaciones_numericas"] = corr_matrix

    # Relaciones entre categóricas y numéricas (media por categoría)
    relaciones = {}
    for cat in cat_cols:
        for num in num_cols:
            relaciones[f"{cat}__vs__{num}"] = df.groupby(cat)[num].mean().to_dict()
    contenido["relaciones_cat_num"] = relaciones

    # Si hay columna tipo fecha → analizar temporalidad
    fechas = df.select_dtypes(include=["datetime", "datetimetz"]).columns
    if len(fechas) > 0:
        temporalidad = {}
        for col in fechas:
            df_temp = df.copy()
            df_temp["año"] = df[col].dt.year
            df_temp["mes"] = df[col].dt.month
            temporalidad[col] = {
                "conteo_por_año": df_temp["año"].value_counts().sort_index().to_dict(),
                "conteo_por_mes": df_temp["mes"].value_counts().sort_index().to_dict(),
            }
        contenido["temporalidad"] = temporalidad

    return contenido

# =========================
# AGENTE DE ANÁLISIS
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
Eres un experto senior en análisis de datos estratégicos, minería de patrones complejos e inteligencia de negocio.
Tu misión no es solo describir los datos, sino extraer inferencias útiles, hipótesis contrafactuales, narrativas con ejemplos concretos y propuestas de valor aplicables.

Dispones de:
- Descripción de la base de datos: {descripcion_db}
- Perfil técnico de la matriz: {perfil}
- Resumen de contenido y patrones básicos: {contenido}

Genera un informe extremadamente detallado, con un tono consultivo y narrativo siguiendo la estructura:

1 Hallazgos clave
   Resumen ejecutivo con los descubrimientos más importantes
   Incluye tanto hallazgos técnicos como estratégicos
   Señala también qué no está en los datos pero podría ser relevante
   Aporta ejemplos ilustrativos de casos representativos

2 Calidad de datos y problemas detectados
   Análisis profundo de valores nulos, duplicados, inconsistencias y errores posibles
   Impacto de estos problemas en el análisis
   Explica qué nuevos datos sería útil recolectar

3 Patrones, tendencias y correlaciones
   Busca tendencias temporales como picos, caídas, estacionalidad o ciclos
   Identifica correlaciones relevantes entre variables numéricas y categóricas
   Detecta interacciones entre variables que podrían no ser evidentes
   Explica posibles causas detrás de los patrones observados
   Si hay fechas, analiza estacionalidad y eventos atípicos

4 Anomalías y outliers relevantes
   Ejemplos específicos con valores concretos
   Posibles explicaciones o hipótesis

5 Riesgos y oportunidades de negocio
   Interpreta los hallazgos con un enfoque práctico
   Incluye escenarios prospectivos

6 Recomendaciones estratégicas y operativas
   Sugerencias accionables basadas en los datos
   Propón acciones experimentales
   Agrega recomendaciones para capturar valor más allá de lo explícito en los datos
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
    print(resultado["informe"])
