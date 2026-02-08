import os
import httpx
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

# 1. Embeddingsの初期化（共通）
embeddings = OllamaEmbeddings(
    base_url=OLLAMA_URL,
    model=EMBED_MODEL
)

# 2. vector_db の初期化
# 起動時にDBが存在すれば読み込み、なければ None にしておく
if os.path.exists(DB_DIR):
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
else:
    vector_db = None

def rebuild_vector_db():
    base_dir = "./knowledge"
    persist_directory = DB_DIR
    
    configs = {
        "short": {"chunk_size": 600, "chunk_overlap": 0},
        "medium": {"chunk_size": 800, "chunk_overlap": 100},
        "long": {"chunk_size": 1500, "chunk_overlap": 200}
    }

    all_docs = []
    for folder, config in configs.items():
        target_dir = os.path.join(base_dir, folder)
        if not os.path.exists(target_dir):
            continue

        print(f"Processing folder: {folder}...")
        for filename in os.listdir(target_dir):
            if filename.endswith((".md", ".txt")):
                file_path = os.path.join(target_dir, filename)
                loader = TextLoader(file_path, encoding='utf-8')
                raw_docs = loader.load()
                for d in raw_docs:
                    d.metadata["doc_type"] = folder
                    d.metadata["source"] = filename

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config["chunk_size"],
                    chunk_overlap=config["chunk_overlap"]
                )
                split_docs = text_splitter.split_documents(raw_docs)
                all_docs.extend(split_docs)

    if all_docs:
        vdb = Chroma.from_documents(
            documents=all_docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print(f"Successfully registered {len(all_docs)} chunks to DB.")
        return vdb
    return None

async def route_question(question: str):
    prompt = f"""以下のユーザーの質問を [short, medium, long] のいずれか1つに分類してください。
回答は必ず単語1つ（short または medium または long）だけで返してください。

- short: 価格、送料、合計金額、個数、在庫、購入手続きに関する具体的な質問
- medium: 味の特徴、淹れ方、レシピ、保存方法、ギフトの相談、おすすめの提案
- long: 伊勢茶の歴史、地域、伝来、人物（高瀬藤八）、産地の背景知識

質問: {question}
分類:"""

    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}}
        )
        category = resp.json().get("response", "").strip().lower()
        if category not in ["short", "medium", "long"]:
            return "medium"
        return category

@app.post("/ask")
async def ask(req: AskRequest):
    global vector_db
    if vector_db is None:
        return {"error": "Vector DB is not initialized. Please run rebuild first."}

    doc_type = await route_question(req.question)
    print(f"DEBUG: ルーター判定結果: {doc_type}")

    results = vector_db.similarity_search(
        req.question,
        k=5,
        filter={"doc_type": doc_type}
    )

    context = "\n".join([doc.page_content for doc in results])
    
    # AIへの最終プロンプト
    # AIへの最終プロンプトを強化
    final_prompt = f"""あなたは「藤八茶寮」の接客担当、茶々丸です。
以下の【参考情報】のみを使用して、ユーザーの質問に親身に回答してください。
【参考情報】にない商品は絶対に提案しないでください。

【参考情報】:
{context}

質問: {req.question}

回答のルール:
1. 【参考情報】に記載されている商品名と価格を正確に引用してください。
2. 5000円以内という予算がある場合は、その範囲内の組み合わせを提案してください。
3. もし情報が足りない場合は、適当なことを言わず「現在のデータからはお答えできません」と伝えてください。
4. 歴史やこだわり（medium/longの情報）があれば、それを添えて魅力を伝えてください。

回答（日本語で出力）:"""

    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL_NAME, "prompt": final_prompt, "stream": False}
        )
        return {"answer": resp.json().get("response"), "category": doc_type}

if __name__ == "__main__":
    # スクリプトとして実行された場合はDB再構築を行う
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
