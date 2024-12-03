from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from lavis.models import load_model_and_preprocess

import os
os.environ["HF_HUB_URL"] = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)


def get_text_embedding_blip2(text):
    """ Extract text embeddings using BLIP-2 """
    text_input = txt_processors["eval"](text)
    sample = {"text_input": [text_input]}
    features = model.extract_features(sample, mode="text")
    return features.text_embeds_proj


def get_image_embedding_blip2(image_url):
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image_tensor = vis_processors["eval"](image).unsqueeze(0).to(device)
        sample = {"image": image_tensor}
        features = model.extract_features(sample, mode="image")
        return features.image_embeds_proj
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
                    result[index] = await loop.run_in_executor(pool, get_text_embedding_blip2, param)
                elif candidate_type == "image":
                    result[index] = await loop.run_in_executor(pool, get_image_embedding_blip2, param)

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    candidates: list[str]
    query_type: str = "text"  # 默认为文本
    candidate_type: str = "text"  # 默认为文本

def cosine_similarity(vec1, vec2):
    vec1_first_row = vec1.mean(dim=1)[0, :].cpu().numpy()
    vec2_first_row = vec2.mean(dim=1)[0, :].cpu().numpy()
    
    # 计算余弦相似度
    dot_product = np.dot(vec1_first_row, vec2_first_row)
    norm_vec1 = np.linalg.norm(vec1_first_row)
    norm_vec2 = np.linalg.norm(vec2_first_row)
    
    cos_sim = dot_product / (norm_vec1 * norm_vec2)
    print(cos_sim)
    return cos_sim

@app.post("/similarity")
async def similarity(request: QueryRequest):
    # 解析请求数据
    query = request.query
    candidates = request.candidates
    query_type = request.query_type
    candidate_type = request.candidate_type

    # 生成查询嵌入
    if query_type == "text":
        query_embedding = get_text_embedding_blip2(query)  # 转换为numpy数组
    elif query_type == "image":
        query_embedding = get_image_embedding_blip2(query)
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
        similarity_score = cosine_similarity(query_embedding, candidate_embedding)
        similarities.append((candidate, float(similarity_score)))  # 确保 similarity_score 是 float 类型

    # 按相似度排序并返回最相似的候选结果
    similarities.sort(key=lambda x: x[1], reverse=True)

    return {"similarities": similarities}
