from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor


# 加载模型和处理器
model = CLIPModel.from_pretrained("/home/data/workgroup/lengmou/Models/openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("/home/data/workgroup/lengmou/Models/openai/clip-vit-large-patch14")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 函数：生成文本嵌入
def get_text_embedding(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embedding = model.get_text_features(**inputs)
    return embedding.cpu().numpy()

def get_image_embedding(image_url):
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = model.get_image_features(**inputs)
        return embedding.cpu().numpy()
    except Exception as e:
        return None


class EmbeddingService:
    def __init__(self, max_concurrency=5):
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def get_embedding(self, index, param, result, candidate_type):
        async with self.semaphore:
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as pool:
                if candidate_type == "text":
                    result[index] = await loop.run_in_executor(pool, get_text_embedding, param)
                elif candidate_type == "image":
                    result[index] = await loop.run_in_executor(pool, get_image_embedding, param)


app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    candidates: list[str]
    query_type: str = "text"  # 默认为文本
    candidate_type: str = "text"  # 默认为文本
    
    
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2.T) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))



@app.post("/similarity")
async def similarity(request: QueryRequest):
    # 解析请求数据
    query = request.query
    candidates = request.candidates
    query_type = request.query_type
    candidate_type = request.candidate_type

    # 生成查询嵌入
    if query_type == "text":
        query_embedding = get_text_embedding(query).tolist()  # 转换为可序列化格式
    elif query_type == "image":
        query_embedding = get_image_embedding(query)
        if query_embedding is None:
            raise HTTPException(status_code=400, detail="Failed to load query image from URL")
        query_embedding = query_embedding.tolist()  # 转换为可序列化格式
    else:
        raise HTTPException(status_code=400, detail="Invalid query_type")

    # 使用并发生成候选嵌入
    result = [None] * len(candidates)
    embedding_service = EmbeddingService(max_concurrency=5)

    # 并发执行任务，限制同时运行的任务数
    await asyncio.gather(*[
        embedding_service.get_embedding(i, candidate, result, candidate_type)
        for i, candidate in enumerate(candidates)
    ])

    # 计算相似度
    similarities = []
    for candidate, candidate_embedding in zip(candidates, result):
        if candidate_embedding is None:
            raise HTTPException(status_code=400, detail=f"Failed to load candidate image from URL: {candidate}")
        similarity_score = cosine_similarity(query_embedding, candidate_embedding)
        similarities.append((candidate, float(similarity_score)))  # 确保 similarity_score 是 float 类型

    # 按相似度排序并返回最相似的候选结果
    similarities.sort(key=lambda x: x[1], reverse=True)

    return {"similarities": similarities}