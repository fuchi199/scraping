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

