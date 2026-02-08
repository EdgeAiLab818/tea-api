import os
import httpx
import uvicorn
import json
import re
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# --- 設定 ---
OLLAMA_URL = "http://ollama:11434"
MODEL_NAME = "gemma2:9b"
EMBED_MODEL = "gemma2:9b"
DB_DIR = "./db"

app = FastAPI()

class AskRequest(BaseModel):
    question: str

# 起動時のDB接続
embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model=EMBED_MODEL)
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings) if os.path.exists(DB_DIR) else None

# --- 補助関数 ---

async def generate_search_queries(question: str):
    """質問から検索用のクエリを最適化"""
    return f"{question} お茶 商品 一覧"

async def route_question_with_weights(question: str):
    """質問内容に基づいて、商品データ(short)と詳細情報(medium)の重みを決定"""
    if any(x in question for x in ["どんな", "種類", "一覧", "価格", "いくら", "販売"]):
        return {"short": 1.0, "medium": 0.4}
    return {"short": 0.5, "medium": 0.5}

# --- メインAPIロジック ---

@app.post("/ask")
async def ask(req: AskRequest):
    global vector_db
    if vector_db is None: 
        return {"error": "DB not found. Please check if ./db exists."}

    optimized_query = await generate_search_queries(req.question)
    weights = await route_question_with_weights(req.question)
    all_results = []

    is_list_query = any(x in req.question for x in ["どんな", "種類", "一覧", "販売", "価格", "いくら"])

    for category, weight in weights.items():
        actual_weight = weight
        if is_list_query and category == "short":
            actual_weight = max(weight, 0.8)

        if actual_weight < 0.1: continue

        query_addon = " 商品名 価格 税込 全種類 一覧"
        combined_query = f"{optimized_query} {query_addon}"

        res = vector_db.max_marginal_relevance_search(
            combined_query, k=20, fetch_k=60, filter={"doc_type": category}
        )
        all_results.extend(res)

    context_parts = []
    for doc in all_results:
        clean_content = re.sub(r'\[cite.*?\]', '', doc.page_content)
        source = doc.metadata.get('source', 'unknown')
        context_parts.append(f"出典:{source}\n内容:{clean_content}")
    context = "\n\n".join(context_parts)

    # ステップ3: ロジック解決（厳格なデータ参照ルール）
    # 出典ファイル名を指定することでハルシネーションを強力に抑制します
    logic_prompt = f"""あなたは藤八茶寮の店主として、正確な「計算」と「提案」を行います。

【データ（検索結果）】:
{context}

【厳守ルール】:
1. 回答に必要な「商品名」「価格」は、出典が knowledge/short/master_products.md であるデータのみを根拠にしてください。
2. 上記データに記載のない商品は、絶対に提案に含めないでください。
3. 商品A + 商品B = 合計金額 を、データ上の価格で正確に計算してください。
4. 予算（5000円以内など）に合う組み合わせがない場合は「ご提案できる組み合わせがございません」と回答してください。

質問: {req.question}
"""

    async with httpx.AsyncClient(timeout=None) as client:
        # ロジックドラフト生成
        logic_resp = await client.post(
            f"{OLLAMA_URL}/api/generate", 
            json={"model": MODEL_NAME, "prompt": logic_prompt, "stream": False, "options": {"temperature": 0.0}}
        )
        raw_answer = logic_resp.json().get("response")

        # ステップ4: 接客用清書（茶々丸モード）
        # 文中の全角記号を最小限にし、プログラム的に安全な構造にしました
        clean_up_prompt = f"""あなたは藤八茶寮の看板スタッフ、茶々丸です。
ドラフトの情報を、接客らしく丁寧に整えてください。

ドラフト内容:
{raw_answer}

清書ルール:
1. 挨拶は省き、すぐ本題に入ってください。
2. 丁寧な語尾（〜です、〜ございます、〜はいかがでしょうか）を使ってください。
3. ドラフトにある商品名と価格は絶対に削らないでください。
4. 箇条書きを活用して見やすくしてください。

回答:"""

        final_resp = await client.post(
            f"{OLLAMA_URL}/api/generate", 
            json={"model": MODEL_NAME, "prompt": clean_up_prompt, "stream": False, "options": {"temperature": 0.0}}
        )
        
        return {
            "answer": final_resp.json().get("response"),
            "weights": weights,
            "debug_raw_logic": raw_answer
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
