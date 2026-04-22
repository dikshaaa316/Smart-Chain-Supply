"""
Placeholder main.py
Person C will implement the FastAPI app and routes here.
"""
from fastapi import FastAPI
from app.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

@app.get("/")
async def root():
    return {"message": "Welcome to LogiSense API"}
