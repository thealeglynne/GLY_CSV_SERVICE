from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os
import traceback
import logging
from datetime import datetime
import uvicorn
import json
import numpy as np

# Custom JSON encoder to handle NaN values
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if np.isnan(obj):
            return None
        return super().default(obj)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from Agents.dg import analizar_matriz
except ImportError:
    from .Agents.dg import analizar_matriz

app = FastAPI(
    title="Servicio de Análisis de Datos con LLM",
    description="API para recibir un archivo CSV/Excel/JSON y una descripción, procesarlo y devolver un informe detallado",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analizar")
async def analizar_endpoint(
    descripcion: str = Form(...),
    file: UploadFile = File(...)
):
    start_time = datetime.now()
    logger.info(f"Starting analysis request at {start_time}")
    
    try:
        # Validaciones previas
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        if not descripcion or len(descripcion.strip()) < 10:
            raise HTTPException(status_code=400, detail="Description must be at least 10 characters")

        allowed_extensions = ['.csv', '.xlsx', '.xls', '.json']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
            )

        # Procesamiento del archivo
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            if not contents:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            
            logger.info(f"Creating temp file with {len(contents)} bytes")
            tmp.write(contents)
            tmp_path = tmp.name

        logger.info(f"Temp file created at: {tmp_path}")
        
        # Ejecutar análisis
        try:
            resultado = analizar_matriz(tmp_path, descripcion_db=descripcion)
        except Exception as analysis_error:
            logger.error(f"Analysis failed: {str(analysis_error)}")
            raise HTTPException(
                status_code=422,
                detail=f"Analysis failed: {str(analysis_error)}"
            )
        
        # Limpieza
        try:
            os.remove(tmp_path)
            logger.info("Temp file removed successfully")
        except Exception as cleanup_error:
            logger.warning(f"Could not remove temp file: {str(cleanup_error)}")

        # Procesar resultado para manejar NaN
        def clean_nans(obj):
            if isinstance(obj, dict):
                return {k: clean_nans(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nans(item) for item in obj]
            elif isinstance(obj, float) and np.isnan(obj):
                return None
            return obj

        resultado_clean = clean_nans(resultado)
        
        # Agregar metadatos
        resultado_clean["metadata"] = {
            "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
            "file_size_bytes": len(contents),
            "file_type": file_extension
        }

        return JSONResponse(
            content=resultado_clean,
            media_type="application/json"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": str(e),
                "traceback": traceback.format_exc() if os.getenv("DEBUG") else None
            }
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        timeout_keep_alive=120
    )