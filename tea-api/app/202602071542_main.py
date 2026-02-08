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

def rebuild_vector_db():
    base_dir = "./knowledge"
    persist_directory = DB_DIR
    configs = {
        "short": {"chunk_size": 2000, "chunk_overlap": 300}, 
        "medium": {"chunk_size": 1000, "chunk_overlap": 200},
        "long": {"chunk_size": 1500, "chunk_overlap": 200}
    }
    all_docs = []
    for folder, config in configs.items():
        target_dir = os.path.join(base_dir, folder)
        if not os.path.exists(target_dir): continue
        print(f"📦 フォルダ読み込み中: {folder}...")
        for filename in os.listdir(target_dir):
            if filename.endswith((".md", ".txt")):
                loader = TextLoader(os.path.join(target_dir, filename), encoding='utf-8')
                raw_docs = loader.load()
                for d in raw_docs:
                    d.metadata["doc_type"] = folder
                    d.metadata["source"] = filename
                split_docs = RecursiveCharacterTextSplitter(
                    chunk_size=config["chunk_size"], chunk_overlap=config["chunk_overlap"]
                ).split_documents(raw_docs)
                all_docs.extend(split_docs)
    if all_docs:
        if os.path.exists(persist_directory):
            import shutil
            shutil.rmtree(persist_directory)
        vdb = Chroma.from_documents(documents=all_docs, embedding=embeddings, persist_directory=persist_directory)
        print(f"✅ DB登録完了: {len(all_docs)} チャンク")
        return vdb
    return None

async def route_question_with_weights(question: str):
    prompt = f"""
以下のユーザーの質問に対し、回答に必要なカテゴリの重要度(0.0-1.0)を判定しJSON形式で回答してください。
- short: 正確な価格、送料、数値計算
- medium: 味、淹れ方、レシピ、ギフト提案
質問: {question}
回答（JSONのみ）:"""
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json={"model": MODEL_NAME, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}})
        try:
            text = resp.json().get("response", "").strip()
            return json.loads(text[text.find('{'):text.rfind('}')+1])
        except: return {"short": 0.5, "medium": 0.5, "long": 0.0}

async def generate_search_queries(question: str):
    shop_info = "藤八茶寮。伊勢の深蒸し茶、ほうじ茶、和紅茶、伊勢茶パウダー。データ:価格(short),送料(short),レシピ(medium)"
    prompt = f"""藤八茶寮のデータ抽出AIとして、質問をDB検索用キーワードに変換してください。
【ショップ情報】:{shop_info}
回答ルール: キーワードのみ出力。
質問: {question}
検索クエリ:"""
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json={"model": MODEL_NAME, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}})
        return resp.json().get("response", "").strip()

@app.post("/ask")
async def ask(req: AskRequest):
    global vector_db
    if vector_db is None: return {"error": "DB not found."}

    optimized_query = await generate_search_queries(req.question)
    weights = await route_question_with_weights(req.question)

    all_results = []
    for category, weight in weights.items():
        if weight < 0.1: continue
        
        query_addon = " 商品名 価格 税込"
        if any(x in req.question for x in ["送料", "送る", "運賃", "届ける"]):
            # 検索時に質問文そのものを混ぜることで、地域名（沖縄など）を確実に拾わせる
            query_addon = f" 送料ポイント 全国一律280円 送料計算ルール 地域別送料表 {req.question}"
            
        combined_query = f"{optimized_query} {query_addon}"
        res = vector_db.max_marginal_relevance_search(
            combined_query, k=12, fetch_k=50, filter={"doc_type": category}
        )
        all_results.extend(res)

    all_results.sort(key=lambda x: 0 if "products" in x.metadata.get("source", "") else 1)

    context_parts = []
    for i, doc in enumerate(all_results):
        clean_content = re.sub(r'\[cite.*?\]', '', doc.page_content)
        source = doc.metadata.get('source', 'unknown')
        ctype = doc.metadata.get('doc_type', 'unknown')
        context_parts.append(f"出典:{source} (カテゴリ:{ctype})\n内容:{clean_content}")

    context = "\n\n".join(context_parts)

    # 5. ステップ3: ロジック解決（内部計算ドラフト）
    logic_prompt = f"""あなたは藤八茶寮の正確な接客担当です。
提供された【データ】を元に、以下の手順を「一歩ずつ書き出して」送料を計算してください。

【データ】:
{context}

質問: {req.question}

【送料計算の鉄則】:
1. **商品の抽出**: 質問にある全ての商品をリストアップし、それぞれの「送料ポイント」を【データ】から特定してください。
2. **ポイント合算**: (商品Aのポイント × 個数) + (商品Bのポイント × 個数) ... = 合計ポイント を算数として計算してください。
3. **条件分岐**:
   - 合計が **6.0ポイント以下** なら、全国一律 **280円**。
   - 合計が **6.0ポイントを超える** 場合のみ、配送地域（例：沖縄県）の送料を適用。
4. **送料無料**: 合計20,000円以上なら0円。

回答（計算プロセスを全て書くこと）:"""
    
    async with httpx.AsyncClient(timeout=None) as client:
        logic_resp = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL_NAME, "prompt": logic_prompt, "stream": False, "options": {"temperature": 0.0}}
        )
        raw_answer = logic_resp.json().get("response")

        # 6. ステップ4: 整形・検証（接客用清書）
        clean_up_prompt = f"""あなたは藤八茶寮の看板スタッフ「茶々丸」です。
【計算ドラフト】の数値を【データ】と照らし合わせ、お客様の質問にだけ丁寧に答えてください。

【データ】:
{context}
【元の質問】:
{req.question}
【計算ドラフト】:
{raw_answer}

清書ルール:
1. **計算プロセス（Step X、ポイントの合算式など）は回答には絶対に含めない。**
2. 質問が送料についてなら、合計ポイントが6.0を超えているかを【計算ドラフト】から読み取り、正確な金額（280円か地域別か）を案内する。
3. 6.0pt超で地域別送料になる場合は「規定のサイズを超えるため」と理由を添える。
4. 価格だけを聞かれた場合は、送料の話はしない。
5. 親身で温かいトーンで回答を完結させる。

回答（接客用清書）:"""

        final_resp = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL_NAME, "prompt": clean_up_prompt, "stream": False, "options": {"temperature": 0.0}}
        )
        
        return {
            "answer": final_resp.json().get("response"),
            "weights": weights,
            "debug_raw_logic": raw_answer # トラブルシューティング用に一旦戻しています
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
