# 基于CLIP/BLIP的Embedding服务构建多模态RAG向量检索模块

在信息爆炸的时代，如何快速从海量数据中找到最相关的信息成为了一个重要的研究课题。RAG（Retrieval-Augmented Generation，检索增强生成）技术作为一种高效的信息检索与生成结合的方法，在自然语言处理领域展现了强大的应用潜力。其核心在于将预训练语言模型与检索模块结合，通过嵌入向量的高效匹配实现信息的精准获取。

在RAG技术中，嵌入向量的生成和匹配是关键环节。本文介绍了一种基于CLIP/BLIP模型的嵌入服务，该服务支持文本和图像的嵌入生成与相似度计算，为多模态信息检索提供了基础能力。

本项目的嵌入服务基于 FastAPI 框架，围绕以下功能展开：
1. **文本嵌入生成**：输入任意文本，生成对应的嵌入向量。
2. **图像嵌入生成**：支持通过 URL 获取图像并生成其嵌入向量。
3. **相似度计算**：通过余弦相似度计算查询数据与候选数据的相关性。
4. **多任务并发处理**：利用异步编程和线程池加速大规模候选集的嵌入生成过程。
5. **高效排序**：根据相似度对候选数据进行排序，返回最相关的结果。

## **技术亮点**

本嵌入服务在设计和实现上具有以下技术优势：

- **多模态支持**：无缝处理文本和图像数据，适配多场景需求，为跨模态信息检索提供统一的解决方案。
- **高性能优化**：利用 GPU 加速嵌入生成，并通过异步任务调度与线程池并发执行，提升服务吞吐量。
- **易扩展性**：采用模块化设计，便于根据业务需求扩展其他功能，如多语言支持或其他嵌入模型集成。
- **可靠性**：通过任务并发数控制（Semaphore）与异常处理，确保在高并发环境下的服务稳定性。

## **适用场景**

- **跨模态检索**  
  结合文本和图像的嵌入特性，支持从图像集匹配文本描述，或从文本库中匹配图像描述的能力。例如，通过输入“海边日落”，从图像集中检索出符合描述的照片。

- **内容推荐**  
  利用嵌入相似性，为用户个性化推荐内容，例如基于用户搜索关键词推荐相关图片或文章。

- **RAG增强生成**  
  嵌入服务可以作为 RAG 系统的检索模块，为生成式模型提供上下文支持，生成与用户问题高度相关的回答或内容。

- **智能问答与搜索**  
  应用于多模态问答系统，通过匹配用户问题与多模态知识库内容，实现更精准的检索与回答。


## **CLIP**

**CLIP（Contrastive Language–Image Pre-training）** 是由 OpenAI 提出的一种多模态模型，能够同时处理文本和图像数据。通过对齐这两种模态的嵌入空间，CLIP 为跨模态检索奠定了坚实的基础。在构建 RAG 系统时，利用 CLIP 生成嵌入可以显著提升多模态信息检索的准确性。

**下载模型**
HuggingFace：openai/clip-vit-large-patch14

### 完整代码

```python
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
```

**启动方式**
```shell
uvicorn embedding:app --host 0.0.0.0 --port 9502
```

**调用方式**

query和candidate均支持text和image，所以支持text-image, text-text, image-image, image-text四种检索模式，以下以其中两种为例

```shell
curl -X POST "http://0.0.0.0:9502/similarity" \
-H "Content-Type: application/json" \
-d '{
  "query": "What is the cycle life of this 3.2V 280ah Lifepo4 battery?",
  "candidates": [
    "https://sc04.alicdn.com/kf/H3510328463d740b2afbcf401c8c108f2J/240062176/H3510328463d740b2afbcf401c8c108f2J.jpg",
    "https://sc04.alicdn.com/kf/H75608c12162a47a4ad41fd331c212e29X/240062176/H75608c12162a47a4ad41fd331c212e29X.jpg",
    "https://sc04.alicdn.com/kf/H1c593aa026e64725a43e1a538be6951ay/240062176/H1c593aa026e64725a43e1a538be6951ay.jpg",
    "https://sc04.alicdn.com/kf/Hb7cda33e8bdc476091ff2962cb4f0ae3x/240062176/Hb7cda33e8bdc476091ff2962cb4f0ae3x.jpg",
    "https://sc04.alicdn.com/kf/Hc00c90da8dcb43b8aeee7eb11b12291b1/240062176/Hc00c90da8dcb43b8aeee7eb11b12291b1.jpg",
    "https://sc04.alicdn.com/kf/H9b13be1329344c3a96f295f144932582u/240062176/H9b13be1329344c3a96f295f144932582u.jpg",
    "https://sc04.alicdn.com/kf/H471ca5edf21a4caea852192af7fbefe7T/240062176/H471ca5edf21a4caea852192af7fbefe7T.jpg",
    "https://sc04.alicdn.com/kf/H38de8263ae5847cb9e6662cdee53743cA/240062176/H38de8263ae5847cb9e6662cdee53743cA.jpg",
    "https://sc04.alicdn.com/kf/H1ea2aa793f5c4d009923d18a473ac219k/240062176/H1ea2aa793f5c4d009923d18a473ac219k.png",
    "https://sc04.alicdn.com/kf/H7fec7cd6293c48168fdd1d41c48ab9e0O/240062176/H7fec7cd6293c48168fdd1d41c48ab9e0O.jpg",
    "https://sc04.alicdn.com/kf/He8d4b88d4323492689455acfa3e44564g/240062176/He8d4b88d4323492689455acfa3e44564g.jpg",
    "https://sc04.alicdn.com/kf/Hff4f46cf682d4deea2094bb71ecc446fu/240062176/Hff4f46cf682d4deea2094bb71ecc446fu.jpg",
    "https://sc04.alicdn.com/kf/Hc5b49b124f1c491aa2fb3078a921929db/240062176/Hc5b49b124f1c491aa2fb3078a921929db.png"
  ],
  "query_type": "text",
  "candidate_type": "image"
}'



curl -X POST "http://0.0.0.0:9502/similarity" \
-H "Content-Type: application/json" \
-d '{
  "query": "How old are you?",
  "candidates": [
    "what is your age?",
    "How are you?",
    "Hello, how tall are you?"
  ],
  "query_type": "text",
  "candidate_type": "text"
}'
```

```text
{
    "similarities": [
        [
            "https://sc04.alicdn.com/kf/H75608c12162a47a4ad41fd331c212e29X/240062176/H75608c12162a47a4ad41fd331c212e29X.jpg",
            0.2983326340781355
        ],
        [
            "https://sc04.alicdn.com/kf/H3510328463d740b2afbcf401c8c108f2J/240062176/H3510328463d740b2afbcf401c8c108f2J.jpg",
            0.2810638577319867
        ],
        [
            "https://sc04.alicdn.com/kf/Hb7cda33e8bdc476091ff2962cb4f0ae3x/240062176/Hb7cda33e8bdc476091ff2962cb4f0ae3x.jpg",
            0.2579443287539946
        ],
        [
            "https://sc04.alicdn.com/kf/H38de8263ae5847cb9e6662cdee53743cA/240062176/H38de8263ae5847cb9e6662cdee53743cA.jpg",
            0.23193220626383945
        ],
        [
            "https://sc04.alicdn.com/kf/Hff4f46cf682d4deea2094bb71ecc446fu/240062176/Hff4f46cf682d4deea2094bb71ecc446fu.jpg",
            0.23164048103513624
        ],
        [
            "https://sc04.alicdn.com/kf/H9b13be1329344c3a96f295f144932582u/240062176/H9b13be1329344c3a96f295f144932582u.jpg",
            0.1929952811358636
        ],
        [
            "https://sc04.alicdn.com/kf/H7fec7cd6293c48168fdd1d41c48ab9e0O/240062176/H7fec7cd6293c48168fdd1d41c48ab9e0O.jpg",
            0.1907971915217675
        ],
        [
            "https://sc04.alicdn.com/kf/H1c593aa026e64725a43e1a538be6951ay/240062176/H1c593aa026e64725a43e1a538be6951ay.jpg",
            0.1895544170769464
        ],
        [
            "https://sc04.alicdn.com/kf/H1ea2aa793f5c4d009923d18a473ac219k/240062176/H1ea2aa793f5c4d009923d18a473ac219k.png",
            0.1886165346225119
        ],
        [
            "https://sc04.alicdn.com/kf/Hc5b49b124f1c491aa2fb3078a921929db/240062176/Hc5b49b124f1c491aa2fb3078a921929db.png",
            0.17306495409231493
        ],
        [
            "https://sc04.alicdn.com/kf/H471ca5edf21a4caea852192af7fbefe7T/240062176/H471ca5edf21a4caea852192af7fbefe7T.jpg",
            0.14305392620079355
        ],
        [
            "https://sc04.alicdn.com/kf/Hc00c90da8dcb43b8aeee7eb11b12291b1/240062176/Hc00c90da8dcb43b8aeee7eb11b12291b1.jpg",
            0.14155516563239476
        ],
        [
            "https://sc04.alicdn.com/kf/He8d4b88d4323492689455acfa3e44564g/240062176/He8d4b88d4323492689455acfa3e44564g.jpg",
            0.10170703460927805
        ]
    ]
}
```

```text
{
    "similarities": [
        [
            "what is your age?",
            0.9363009189669045
        ],
        [
            "How are you?",
            0.8448722668528517
        ],
        [
            "Hello, how tall are you?",
            0.7906618164331359
        ]
    ]
}
```


## **BLIP**

**BLIP（Bootstrapped Language–Image Pre-training）** 是一种由 Salesforce 提出的新型多模态模型，专注于提升图文理解与生成能力。BLIP 通过结合多阶段预训练方法，在对齐语言和视觉嵌入空间的同时，显著提高了跨模态任务的性能表现。在构建 RAG 系统时，利用 BLIP 生成嵌入不仅可以实现高效的多模态信息检索，还能支持更丰富的图文生成与交互应用。

**Lavis**
使用时需要先安装Lavis，从github上clone源码，然后python setup.py install。

**HuggingFace配置**
有时从hf官网下载模型会比较慢或者下不下来，可以通过配置镜像来下载，如果科学上网环境良好：
```shell
export HF_HUB_URL=https://hf-mirror.com
export HF_ENDPOINT=https://hf-mirror.com
```

**完整代码**

```python
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
```

**启动方式**

```shell
uvicorn embedding:app --host 0.0.0.0 --port 9502
```

**调用方式**
```shell
curl -X POST "http://0.0.0.0:9502/similarity" \
-H "Content-Type: application/json" \
-d '{
  "query": "What is the cycle life of this 3.2V 280ah Lifepo4 battery?",
  "candidates": [
    "https://sc04.alicdn.com/kf/H3510328463d740b2afbcf401c8c108f2J/240062176/H3510328463d740b2afbcf401c8c108f2J.jpg",
    "https://sc04.alicdn.com/kf/H75608c12162a47a4ad41fd331c212e29X/240062176/H75608c12162a47a4ad41fd331c212e29X.jpg",
    "https://sc04.alicdn.com/kf/H1c593aa026e64725a43e1a538be6951ay/240062176/H1c593aa026e64725a43e1a538be6951ay.jpg",
    "https://sc04.alicdn.com/kf/Hb7cda33e8bdc476091ff2962cb4f0ae3x/240062176/Hb7cda33e8bdc476091ff2962cb4f0ae3x.jpg",
    "https://sc04.alicdn.com/kf/Hc00c90da8dcb43b8aeee7eb11b12291b1/240062176/Hc00c90da8dcb43b8aeee7eb11b12291b1.jpg",
    "https://sc04.alicdn.com/kf/H9b13be1329344c3a96f295f144932582u/240062176/H9b13be1329344c3a96f295f144932582u.jpg",
    "https://sc04.alicdn.com/kf/H471ca5edf21a4caea852192af7fbefe7T/240062176/H471ca5edf21a4caea852192af7fbefe7T.jpg",
    "https://sc04.alicdn.com/kf/H38de8263ae5847cb9e6662cdee53743cA/240062176/H38de8263ae5847cb9e6662cdee53743cA.jpg",
    "https://sc04.alicdn.com/kf/H1ea2aa793f5c4d009923d18a473ac219k/240062176/H1ea2aa793f5c4d009923d18a473ac219k.png",
    "https://sc04.alicdn.com/kf/H7fec7cd6293c48168fdd1d41c48ab9e0O/240062176/H7fec7cd6293c48168fdd1d41c48ab9e0O.jpg",
    "https://sc04.alicdn.com/kf/He8d4b88d4323492689455acfa3e44564g/240062176/He8d4b88d4323492689455acfa3e44564g.jpg",
    "https://sc04.alicdn.com/kf/Hff4f46cf682d4deea2094bb71ecc446fu/240062176/Hff4f46cf682d4deea2094bb71ecc446fu.jpg",
    "https://sc04.alicdn.com/kf/Hc5b49b124f1c491aa2fb3078a921929db/240062176/Hc5b49b124f1c491aa2fb3078a921929db.png"
  ],
  "query_type": "text",
  "candidate_type": "image"
}'


curl -X POST "http://0.0.0.0:9502/similarity" \
-H "Content-Type: application/json" \
-d '{
  "query": "How old are you?",
  "candidates": [
    "what is your age?",
    "How are you?",
    "Hello, how tall are you?"
  ],
  "query_type": "text",
  "candidate_type": "text"
}'
```

```text
{
    "similarities": [
        [
            "https://sc04.alicdn.com/kf/H3510328463d740b2afbcf401c8c108f2J/240062176/H3510328463d740b2afbcf401c8c108f2J.jpg",
            0.4438434839248657
        ],
        [
            "https://sc04.alicdn.com/kf/H38de8263ae5847cb9e6662cdee53743cA/240062176/H38de8263ae5847cb9e6662cdee53743cA.jpg",
            0.3966401517391205
        ],
        [
            "https://sc04.alicdn.com/kf/H75608c12162a47a4ad41fd331c212e29X/240062176/H75608c12162a47a4ad41fd331c212e29X.jpg",
            0.35076430439949036
        ],
        [
            "https://sc04.alicdn.com/kf/Hb7cda33e8bdc476091ff2962cb4f0ae3x/240062176/Hb7cda33e8bdc476091ff2962cb4f0ae3x.jpg",
            0.3383423089981079
        ],
        [
            "https://sc04.alicdn.com/kf/Hff4f46cf682d4deea2094bb71ecc446fu/240062176/Hff4f46cf682d4deea2094bb71ecc446fu.jpg",
            0.31132861971855164
        ],
        [
            "https://sc04.alicdn.com/kf/He8d4b88d4323492689455acfa3e44564g/240062176/He8d4b88d4323492689455acfa3e44564g.jpg",
            0.23726868629455566
        ],
        [
            "https://sc04.alicdn.com/kf/Hc00c90da8dcb43b8aeee7eb11b12291b1/240062176/Hc00c90da8dcb43b8aeee7eb11b12291b1.jpg",
            0.23664134740829468
        ],
        [
            "https://sc04.alicdn.com/kf/H1c593aa026e64725a43e1a538be6951ay/240062176/H1c593aa026e64725a43e1a538be6951ay.jpg",
            0.19802381098270416
        ],
        [
            "https://sc04.alicdn.com/kf/Hc5b49b124f1c491aa2fb3078a921929db/240062176/Hc5b49b124f1c491aa2fb3078a921929db.png",
            0.18722020089626312
        ],
        [
            "https://sc04.alicdn.com/kf/H471ca5edf21a4caea852192af7fbefe7T/240062176/H471ca5edf21a4caea852192af7fbefe7T.jpg",
            0.1803196370601654
        ],
        [
            "https://sc04.alicdn.com/kf/H1ea2aa793f5c4d009923d18a473ac219k/240062176/H1ea2aa793f5c4d009923d18a473ac219k.png",
            0.16748569905757904
        ],
        [
            "https://sc04.alicdn.com/kf/H7fec7cd6293c48168fdd1d41c48ab9e0O/240062176/H7fec7cd6293c48168fdd1d41c48ab9e0O.jpg",
            0.14610795676708221
        ],
        [
            "https://sc04.alicdn.com/kf/H9b13be1329344c3a96f295f144932582u/240062176/H9b13be1329344c3a96f295f144932582u.jpg",
            0.11708579212427139
        ]
    ]
}

{
    "similarities": [
        [
            "How are you?",
            0.8776232600212097
        ],
        [
            "what is your age?",
            0.8666055202484131
        ],
        [
            "Hello, how tall are you?",
            0.8007299304008484
        ]
    ]
}
```