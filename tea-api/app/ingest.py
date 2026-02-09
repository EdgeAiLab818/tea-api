import os
import json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- 設定 ---
OLLAMA_URL = "http://ollama:11434"
EMBED_MODEL = "gemma2:9b"
DB_DIR = "./db"
KNOWLEDGE_DIR = "./knowledge"

def main():
    documents = []
    
    # 1. ファイルの探索と読み込み
    for root, dirs, files in os.walk(KNOWLEDGE_DIR):
        for file in files:
            path = os.path.join(root, file)
            # フォルダ名からカテゴリ判定
            doc_type = "short" if "short" in root else "medium"

            # --- JSONファイルの処理 ---
            if file.endswith(".json"):
                print(f"📄 Processing JSON: {path}")
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # リスト形式（[{商品1}, {商品2}]）を想定
                        items = data if isinstance(data, list) else [data]
                        
                        for item in items:
                            # AIが検索しやすいようにテキスト化（キー：値 の形式）
                            content = "\n".join(f"{k}: {v}" for k, v in item.items())
                            
                            # メタデータにも情報を保持（将来のフィルタリング用）
                            doc = Document(
                                page_content=content,
                                metadata={
                                    "source": path, 
                                    "doc_type": doc_type,
                                    **{k: str(v) for k, v in item.items()} # すべて文字列で保持
                                }
                            )
                            documents.append(doc)
                except Exception as e:
                    print(f"❌ Error loading JSON {path}: {e}")

            # --- Markdownファイルの処理 ---
            elif file.endswith(".md"):
                print(f"📄 Processing Markdown: {path}")
                try:
                    loader = TextLoader(path)
                    docs = loader.load()
                    for d in docs:
                        d.metadata["doc_type"] = doc_type
                        d.metadata["source"] = path
                    documents.extend(docs)
                except Exception as e:
                    print(f"❌ Error loading Markdown {path}: {e}")

    if not documents:
        print("⚠️ No documents found to ingest.")
        return

    # 2. テキスト分割
    # JSONデータはすでに分割されているため、Markdownのみを考慮した大きめの設定
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n\n", "\n"]
    )
    split_docs = splitter.split_documents(documents)

    # 3. ベクトルDB作成
    print(f"🚀 Creating Vector DB with {len(split_docs)} chunks...")
    embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model=EMBED_MODEL)
    
    # 既存のDBがあれば上書き、なければ新規作成
    vector_db = Chroma.from_documents(
        documents=split_docs, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    
    print(f"🎉 Success: Vector DB created at {DB_DIR}")

if __name__ == "__main__":
    main()