from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ====== 文章をリストに入れる ======
documents = [
    "CSV登録エラー 再実行してください",
    "CSV登録失敗 CSV修正後に再実行",
    "ネットワークエラー 接続を確認",
    "サーバー応答なし サーバー再起動",
    "CSV読み込みできませんでした",
    "通信タイムアウト 接続エラー",
]

# ====== TF-IDF で文章を数値化 ======
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# ====== KMeans でクラスタリング ======
k = 2  # ←グループ数（必要に応じて増やせる）
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(X)

# ====== 結果表示 ======
for text, label in zip(documents, kmeans.labels_):
    print(f"[Cluster {label}] {text}")