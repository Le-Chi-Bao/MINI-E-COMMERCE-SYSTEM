from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from dotenv import load_dotenv

from app.database import create_tables, init_sample_data
from app.api.endpoints import router as api_router

load_dotenv()

create_tables()
init_sample_data()

app = FastAPI(
    title="Phone Price Predictor API",
    description="AI Phone Price Prediction System",
    version="1.0.0",
    docs_url="/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/")
async def root():
    return {
        "message": "Phone Price Predictor API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)