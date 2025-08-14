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
    """Genera un resumen más compacto del contenido para análisis estratégico."""
    contenido = {}

    # Muestra aleatoria (limitada para no saturar tokens)
    contenido["muestra"] = df.sample(min(sample_size, len(df)), random_state=42).to_dict(orient="records")

    # Distribuciones por columnas numéricas (resumidas)
    num_cols = df.select_dtypes(include=[np.number]).columns
    distribuciones = {}
    for col in num_cols:
        distribuciones[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "media": float(df[col].mean()),
            "mediana": float(df[col].median()),
            "desviacion": float(df[col].std())
        }
    contenido["distribuciones_numericas"] = distribuciones

    # Distribuciones por columnas categóricas (solo top 5 valores)
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    distrib_cat = {}
    for col in cat_cols:
        distrib_cat[col] = dict(list(df[col].value_counts().head(5).items()))
    contenido["distribuciones_categoricas"] = distrib_cat

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
Eres un experto senior en análisis de datos estratégicos y minería de patrones complejos.
Dispones de:
- Descripción de la base de datos: {descripcion_db}
- Perfil técnico de la matriz: {perfil}
- Resumen de contenido y patrones básicos: {contenido}

Genera un informe **extremadamente detallado** con la siguiente estructura:

1. **Hallazgos clave**  
   - Resumen ejecutivo con los descubrimientos más importantes.  
   - Incluye tanto hallazgos técnicos como estratégicos.

2. **Calidad de datos y problemas detectados**  
   - Análisis profundo de valores nulos, duplicados, inconsistencias y errores posibles.  
   - Impacto de estos problemas en el análisis.

3. **Patrones, tendencias y correlaciones (SECCIÓN EXTENDIDA)**  
   - Busca tendencias temporales (picos, caídas, estacionalidad, ciclos).  
   - Identifica correlaciones relevantes entre variables numéricas y categóricas.  
   - Detecta interacciones entre variables que podrían no ser evidentes.  
   - Explica posibles causas detrás de los patrones observados.  
   - Si hay fechas, analiza estacionalidad y eventos atípicos.  
   - Si hay datos regionales, compara entre regiones y busca relaciones con otros atributos.  
   - Formula hipótesis fundamentadas basadas en los datos.
   - Incluye ejemplos concretos de registros que representen estos patrones.

4. **Anomalías y outliers relevantes**  
   - Ejemplos específicos con valores concretos.  
   - Posibles explicaciones o hipótesis.

5. **Riesgos y oportunidades de negocio**  
   - Interpreta los hallazgos con un enfoque práctico.

6. **Recomendaciones estratégicas y operativas**  
   - Sugerencias accionables basadas en los datos.

Responde en formato Markdown con subtítulos claros y explicaciones narrativas ricas en detalles.
"""
)

    llm = ChatGroq(model_name="llama3-70b-8192", temperature=temperature, api_key=GROQ_API_KEY)
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
