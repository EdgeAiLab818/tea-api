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
    """質問から余計な装飾を削り、商品検索に特化したキーワードを抽出する"""
    return question

async def route_question_with_weights(question: str):
    """質問内容に基づいて、商品データ(short)と詳細情報(medium)の重みを決定"""
    # 商品一覧や価格を尋ねるキーワードが含まれる場合
    if any(x in question for x in ["どんな", "種類", "一覧", "価格", "いくら", "販売"]):
        return {"short": 1.0, "medium": 0.4}
    # それ以外はバランスよく検索
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

    # 商品情報を優先すべきか判定
    is_list_query = any(x in req.question for x in ["どんな", "種類", "一覧", "販売", "価格", "いくら"])

    for category, weight in weights.items():
        actual_weight = weight
        if is_list_query and category == "short":
            actual_weight = max(weight, 0.8)

        if actual_weight < 0.1: continue

        # 検索キーワードに「商品マスター」的ニュアンスを付与
        query_addon = " 商品名 価格 税込 全種類 一覧"
        combined_query = f"{optimized_query} {query_addon}"

        # 網羅性を高めるため k=20 で検索
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

    # ステップ3: ロジック解決（提案・計算モード）
    logic_prompt = f"""あなたは藤八茶寮の店主です。
【データ】の内容「のみ」を使用して、お客様に提案してください。

【データ】:
{context}
質問: {req.question}

【厳守事項】:
1. **データ外の提案禁止**: 【データ】に記載されていない商品（抹茶碗、お菓子、玉露など）は、一般常識であっても絶対に提案しないでください。
2. **在庫にあるものだけ**: 今、あなたの手元（データ）にある商品だけで、5000円以内の組み合わせを作ってください。
3. **事実のみ回答**: もしデータ内に適切な商品が一つもない場合は、正直に「現在ご案内できる商品がございません」と回答してください。
4. **計算の明示**: 商品A + 商品B = 合計金額 を、データ上の価格で正確に計算してください。
"""

    async with httpx.AsyncClient(timeout=None) as client:
        # ロジックドラフト生成
        logic_resp = await client.post(
            f"{OLLAMA_URL}/api/generate", 
            json={"model": MODEL_NAME, "prompt": logic_prompt, "stream": False, "options": {"temperature": 0.0}}
        )
        raw_answer = logic_resp.json().get("response")

        # ステップ4: 接客用清書（茶々丸モード）
        clean_up_prompt = f"""あなたは藤八茶寮の看板スタッフ「茶々丸」です。
【計算ドラフト】の情報を、接客らしく丁寧に整えてください。

【清書ルール】:
1. **冒頭の挨拶は不要**: すぐに本題に入ってください。
2. **丁寧な語尾**: 「〜です」「〜ございます」「〜はいかがでしょうか」を使ってください。
3. **情報の維持**: ドラフトにある「全ての商品名」「組み合わせ」「価格」は削らないでください。
4. **見やすさ**: 箇条書きを活用してください。

【計算ドラフト】:
{raw_answer}

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
