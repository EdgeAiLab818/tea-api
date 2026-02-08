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
- medium: 味、淹れ方、レシピ、ギフト提案、歴史
質問: {question}
回答（JSONのみ）:"""
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json={"model": MODEL_NAME, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}})
        try:
            text = resp.json().get("response", "").strip()
            return json.loads(text[text.find('{'):text.rfind('}')+1])
        except: return {"short": 0.5, "medium": 0.5, "long": 0.0}

async def generate_search_queries(question: str):
    shop_info = "藤八茶寮。伊勢茶専門店。深蒸し茶、ほうじ茶、和紅茶、パウダー、ティーバッグ。"
    prompt = f"""藤八茶寮のデータ抽出AIとして、質問をDB検索用キーワードに変換してください。
【ショップ情報】:{shop_info}
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
        query_addon = " 商品名 価格 税込 全種類 一覧"
        if any(x in req.question for x in ["送料", "送る", "運賃", "届ける"]):
            query_addon = f" 送料ポイント 6.0pt判定 地域別送料表 {req.question}"
        combined_query = f"{optimized_query} {query_addon}"
        res = vector_db.max_marginal_relevance_search(
            combined_query, k=15, fetch_k=50, filter={"doc_type": category}
        )
        all_results.extend(res)

    all_results.sort(key=lambda x: 0 if "products" in x.metadata.get("source", "") else 1)

    context_parts = []
    for doc in all_results:
        clean_content = re.sub(r'\[cite.*?\]', '', doc.page_content)
        source = doc.metadata.get('source', 'unknown')
        context_parts.append(f"出典:{source}\n内容:{clean_content}")

    context = "\n\n".join(context_parts)

    # ステップ3: ロジック解決（抽出と判定の分離）
    logic_prompt = f"""あなたは藤八茶寮のデータ管理・推論担当です。
提供された【データ】を客観的に分析し、以下の【思考プロセス】に従って回答案を作成してください。

【データ】:
{context}
質問: {req.question}

【思考プロセス】:
1. **意図判定**: 質問は「事実確認（商品一覧、価格、特徴など）」か「送料計算」か？
2. **Extractor（事実確認）**: 
   - 質問に関連する商品名を【データ】から正確に「抽出」してください。
   - 【禁止】: データにない商品名（焙煎茶など）や架空の価格を捏造することは「重大な規約違反」です。
   - 【必須】: ティーバッグやパウダーなど、形状（風袋）が異なるものも全て個別にリストアップしてください。
3. **Reasoner（送料計算）**:
   - 送料計算が必要な場合のみ、商品名・数量・配送先を確認し、【送料計算の鉄則】を適用してください。

【送料計算の鉄則】:
1. 商品ポイント特定。 2. (ポイント × 注文数) の合計算出。 3. 6.0pt判定（280円 or 地域別）。 4. 20,000円以上無料。

回答（抽出された情報を正確に記述）:"""
    
    async with httpx.AsyncClient(timeout=None) as client:
        logic_resp = await client.post(f"{OLLAMA_URL}/api/generate", json={"model": MODEL_NAME, "prompt": logic_prompt, "stream": False, "options": {"temperature": 0.0}})
        raw_answer = logic_resp.json().get("response")

        # ステップ4: 整形・検証（情報維持の徹底）
        clean_up_prompt = f"""あなたは藤八茶寮の看板スタッフ「茶々丸」です。
【計算ドラフト】の情報を、親しみやすい接客文に整えてください。

【計算ドラフト】:
{raw_answer}

【清書ルール】:
1. **情報の完全維持（最重要）**: ドラフトにある「具体的な商品名」や「価格」は、要約したり削ったりせず、必ず全て回答に含めてください。「豊富ですね」と一言でまとめるのは「禁止」です。
2. **データの裏付け**: ドラフトの内容が【データ】と矛盾していないか最終確認し、矛盾があれば【データ】の数値を優先してください。
3. 計算式やポイント数などの内部プロセスは、接客に不要なため削除してください。
4. 質問に送料が含まれない場合は、送料の話題は出さないでください。

回答（接客用清書）:"""

        final_resp = await client.post(f"{OLLAMA_URL}/api/generate", json={"model": MODEL_NAME, "prompt": clean_up_prompt, "stream": False, "options": {"temperature": 0.0}})
        
        return {
            "answer": final_resp.json().get("response"),
            "weights": weights,
            "debug_raw_logic": raw_answer
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
