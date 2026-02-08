# ingest_all.py
import requests

API_URL = "https://tea.edgeailab.jp/ingest/url"
API_KEY = "chachamaru_secret_key_2026"

urls = [
    "https://108teaworks.com/",
    "https://www.maff.go.jp/j/keikaku/syokubunka/",
    "https://www.maff.go.jp/j/shokusan/think-food-agri/tea.html",
    "https://www.maff.go.jp/j/pr/aff/1704/spe1_01.html",
    "https://www.maff.go.jp/j/pr/aff/1704/spe1_02.html",
    "https://www.maff.go.jp/j/pr/aff/1704/spe1_03.html",
    "https://www.maff.go.jp/j/pr/aff/1704/spe1_04.html",
    "https://www.maff.go.jp/j/keikaku/syokubunka/traditional-foods/menu/ise_tya.html",
    "https://www.nihon-cha.or.jp/",
    "https://gjtea.org/jp/japanese-tea-innovators-intro/",
    "https://www.pref.mie.lg.jp/NOUSAN/HP/77019045892.htm",
    "https://www.isecha.net/",
    "https://www.zennoh.or.jp/me/product/tea/type/",
    "http://www.mie-isecha.org/",
    "https://www.isecha.net/index.html",
    "https://www.pref.mie.lg.jp/common/05/ci600004388.htm"
]

for url in urls:
    headers = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}
    payload = {"url": url}
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        print(f"Sent: {url} -> Status: {response.status_code}")
    except Exception as e:
        print(f"Failed: {url} -> {e}")

print("\n--- 全URLの送信が完了しました！バックグラウンド処理を見守りましょう ---")
