from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import uvicorn

from ml_model import predict_image
from agent import get_agent_research

app = FastAPI(title="Sea Animal Classifier API")

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AgentRequest(BaseModel):
    label: str

@app.get("/")
async def root():
    return {"status": "alive", "message": "Sea Animal Classifier API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        
        # Run prediction
        results = predict_image(image)
        
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/research")
async def research(req: AgentRequest):
    try:
        research_data = get_agent_research(req.label)
        return research_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
