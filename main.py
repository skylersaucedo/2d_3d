from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
from google.cloud import storage
from claude_mcp_builder import ClaudeMCPBuilder

load_dotenv()

app = FastAPI(title="3D Model Generation API")

class ImagePaths(BaseModel):
    side1_path: str
    side2_path: str
    side3_path: str

class ModelResponse(BaseModel):
    stl_path: str
    brep_path: str

@app.post("/generate-3d-model", response_model=ModelResponse)
async def generate_3d_model(image_paths: ImagePaths):
    try:
        # Initialize GCP storage client
        storage_client = storage.Client()
        bucket_name = os.getenv("GCP_BUCKET_NAME")
        bucket = storage_client.bucket(bucket_name)

        # Initialize ClaudeMCPBuilder
        builder = ClaudeMCPBuilder()
        
        # Generate 3D model
        stl_path, brep_path = await builder.generate_model(
            image_paths.side1_path,
            image_paths.side2_path,
            image_paths.side3_path
        )

        return ModelResponse(
            stl_path=stl_path,
            brep_path=brep_path
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 