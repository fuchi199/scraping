a = [1,2,3,4]
b = [{"name":"太郎","id":2},{"name":"一郎","id":5},{"name":"小太郎","id":4}]


a_set = set(a)  # O(1) での探索が可能に
output = [item for item in b if item["id"] in a_set]
print(output)


a = [{"age": "21", "id": 2}, {"age": "24", "id": 2}, {"age": "25", "id": 5}, {"age": "27", "id": 5}]
b = [{"name": "太郎", "id": 2}, {"name": "一郎", "id": 5}, {"name": "小太郎", "id": 4}]

# b を id をキーとした辞書に変換
b_dict = {item["id"]: item for item in b}

# マッチングして出力
output = [
    {**b_dict[item["id"]], **item}
    for item in a
    if item["id"] in b_dict
]

print(output)

from collections import defaultdict

output = [
    {"name": "太郎", "id": 2, "age": "21"},
    {"name": "太郎", "id": 2, "age": "24"},
    {"name": "一郎", "id": 5, "age": "25"},
    {"name": "一郎", "id": 5, "age": "27"}
]

# 件数カウント
counts = defaultdict(int)
for item in output:
    counts[item["name"]] += 1

# 配列に格納
result = [{"name": name, "count": count} for name, count in counts.items()]


import pandas as pd

# 例のデータフレーム
df1 = pd.DataFrame({"name": ["太郎", "一郎"], "count": [2, 2]})
df2 = pd.DataFrame({"age": [21, 24, 25, 27]})

# Excelファイルに書き出し（複数シート）
with pd.ExcelWriter("a.xlsx", engine="openpyxl") as writer:
    df1.to_excel(writer, sheet_name="Summary", index=False)
    df2.to_excel(writer, sheet_name="AgeData", index=False)


data = [{"id": i, "name": f"name{i}"} for i in range(501)]

chunk_size = 100

for i in range(0, len(data), chunk_size):
    chunk = data[i:i + chunk_size]
    # ここにchunkを処理する関数を呼び出すなど
    print(f"{i//chunk_size + 1} バッチ目: {len(chunk)} 件")



import aiohttp
import asyncio

# ダミーAPIのURL（あなたのAPIエンドポイントに差し替えてください）
API_URL = "https://example.com/api/send"

# 送信関数（非同期）
async def send_chunk(session, chunk):
    try:
        async with session.post(API_URL, json=chunk) as resp:
            resp.raise_for_status()
            result = await resp.json()
            print(f"送信完了: {len(chunk)}件, レスポンス: {result}")
    except Exception as e:
        print(f"送信エラー: {e}")

# メイン処理
async def main():
    # サンプルデータ（500件）
    data = [{"id": i, "name": f"name{i}"} for i in range(1, 501)]
    chunk_size = 100

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            tasks.append(send_chunk(session, chunk))
        await asyncio.gather(*tasks)

# 実行
if __name__ == "__main__":
    asyncio.run(main())
