import os
import httpx
import uvicorn
import json
import re
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# --- 設定 ---
OLLAMA_URL = "http://ollama:11434"
MODEL_NAME = "gemma2:9b"
EMBED_MODEL = "gemma2:9b"
DB_DIR = "./db"

app = FastAPI()

class AskRequest(BaseModel):
    question: str

embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model=EMBED_MODEL)
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings) if os.path.exists(DB_DIR) else None

# --- 重複排除と検索強化を盛り込んだ ask 関数 ---
@app.post("/ask")
async def ask(req: AskRequest):
    global vector_db
    if vector_db is None: return {"error": "DB not found."}

    optimized_query = await generate_search_queries(req.question)
    weights = await route_question_with_weights(req.question)
    all_results = []

    # 商品リスト・価格に関わる質問か判定
    is_list_query = any(x in req.question for x in ["どんな", "種類", "一覧", "販売", "価格", "いくら"])

    for category, weight in weights.items():
        actual_weight = weight
        if is_list_query and category == "short":
            actual_weight = max(weight, 0.8) # 商品リストの時は short を優先

        if actual_weight < 0.1: continue

        query_addon = " 商品名 価格 税込 全種類 一覧 商品マスター"
        combined_query = f"{optimized_query} {query_addon}"

        # 検索件数を20に増やして網羅性を確保
        res = vector_db.max_marginal_relevance_search(
            combined_query, k=20, fetch_k=60, filter={"doc_type": category}
        )
        all_results.extend(res)

    # 検索結果をコンテキスト化
    context_parts = []
    for doc in all_results:
        clean_content = re.sub(r'\[cite.*?\]', '', doc.page_content)
        source = doc.metadata.get('source', 'unknown')
        context_parts.append(f"出典:{source}\n内容:{clean_content}")
    context = "\n\n".join(context_parts)

    # ステップ3: ロジック解決（抽出と判定の分離・重複排除）
    logic_prompt = f"""あなたは藤八茶寮の正確なデータ管理担当です。
提供された【データ】を客観的に分析し、回答案を作成してください。

【データ】:
{context}
質問: {req.question}

【思考・回答プロセス】:
1. **網羅性の確保**: 【データ】内の商品を全て見つけ出してください（ほうじ茶パウダー、お得用等も含む）。
2. **情報の必須化**: 「税込価格」を必ずセットで抽出してください。
3. **重複の厳禁**: 同じ商品を二度リストアップしないでください。一意のリストを作成してください。
4. **捏造禁止**: 【データ】にない商品名や価格は絶対に書かないでください。
"""
    
    async with httpx.AsyncClient(timeout=None) as client:
        logic_resp = await client.post(
            f"{OLLAMA_URL}/api/generate", 
            json={"model": MODEL_NAME, "prompt": logic_prompt, "stream": False, "options": {"temperature": 0.0}}
        )
        raw_answer = logic_resp.json().get("response")

        # ステップ4: 接客用清書（挨拶禁止・情報の完全維持）
        clean_up_prompt = f"""あなたは藤八茶寮の看板スタッフ「茶々丸」です。
【計算ドラフト】の情報を、接客らしく丁寧に整えてください。

【計算ドラフト】:
{raw_answer}

【清書ルール】:
1. **冒頭の挨拶は不要**: 「ようこそ」などの挨拶は省き、すぐ本題に入ってください。
2. **丁寧な語尾**: リストの前後や説明には「〜です」「〜ございます」を使い、接客らしい言葉遣いにしてください。
3. **情報の完全維持**: ドラフトにある「全ての商品名」と「価格」は絶対に削らないでください。
4. **見やすさ**: 箇条書きを活用し、一目で価格がわかるようにしてください。

回答（接客用清書）:"""

        final_resp = await client.post(
            f"{OLLAMA_URL}/api/generate", 
            json={"model": MODEL_NAME, "prompt": clean_up_prompt, "stream": False, "options": {"temperature": 0.0}}
        )
        
        return {
            "answer": final_resp.json().get("response"),
            "weights": weights,
            "debug_raw_logic": raw_answer
        }
