import pandas as pd
import matplotlib.pyplot as plt

# CSV読み込み
df = pd.read_csv("error_dataset.csv", encoding="utf-8-sig")

# 日付をdatetime型に変換
df["日付"] = pd.to_datetime(df["日付"])

# 年月（例：2025-01）を抽出
df["年月"] = df["日付"].dt.to_period("M")

# ==== 月ごとのエラー件数をカウント ====
monthly_errors = df.groupby("年月")["エラー内容"].count().reset_index()
monthly_errors.columns = ["年月", "件数"]

# ==== 表で確認 ====
print(monthly_errors)

# ==== グラフ化 ====
plt.figure(figsize=(8, 5))
plt.plot(monthly_errors["年月"].astype(str), monthly_errors["件数"], marker="o")
plt.title("月ごとのエラー発生件数")
plt.xlabel("年月")
plt.ylabel("件数")
plt.grid(True)
plt.show()


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


# =============================================================
# 1. データ読み込み
# =============================================================
df = pd.read_csv("error_logs.csv")  # ← CSVの列に "error_log" と "category_name" が必要

error_texts = df["error_log"].tolist()
category_texts = df["category_name"].tolist()


# =============================================================
# 2. TF-IDF → NumPy
# =============================================================
log_vectorizer = TfidfVectorizer(max_features=5000)
X = log_vectorizer.fit_transform(error_texts).toarray()

encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(np.array(category_texts).reshape(-1, 1))

n_input = X.shape[1]
n_output = Y.shape[1]


# =============================================================
# 3. 自作 NN（中間層 + Dropout + Softmax）
# =============================================================

class MiddleLayer:
    def __init__(self, n_input, n_output):
        self.w = np.random.randn(n_input, n_output) * 0.01
        self.b = np.zeros(n_output)

    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = 1 / (1 + np.exp(-u))  # sigmoid

    def backward(self, grad_y):
        delta = grad_y * (1 - self.y) * self.y
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)

    def update(self, lr):
        self.w -= lr * self.grad_w
        self.b -= lr * self.grad_b


class DropoutMiddleLayer(MiddleLayer):
    def __init__(self, n_input, n_output, dropout_ratio):
        super().__init__(n_input, n_output)
        self.dropout_ratio = dropout_ratio

    def forward(self, x, train_flg=True):
        self.x = x
        u = np.dot(x, self.w) + self.b
        y = 1 / (1 + np.exp(-u))  # sigmoid

        if train_flg:
            self.mask = np.random.rand(*y.shape) > self.dropout_ratio
            self.y = y * self.mask
        else:
            self.y = y * (1 - self.dropout_ratio)


class OutputLayer:
    def __init__(self, n_input, n_output):
        self.w = np.random.randn(n_input, n_output) * 0.01
        self.b = np.zeros(n_output)

    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        exp_u = np.exp(u - np.max(u, axis=1, keepdims=True))
        self.y = exp_u / np.sum(exp_u, axis=1, keepdims=True)

    def backward(self, target):
        delta = (self.y - target) / len(target)
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)

    def update(self, lr):
        self.w -= lr * self.grad_w
        self.b -= lr * self.grad_b


# =============================================================
# 4. ネットワーク構築（中間層を増やす）
# =============================================================

# ネットワーク構造（中間層2つ、Dropoutあり）
n_middle1 = 256
n_middle2 = 128
dropout_ratio = 0.3

layer1 = DropoutMiddleLayer(n_input, n_middle1, dropout_ratio)
layer2 = DropoutMiddleLayer(n_middle1, n_middle2, dropout_ratio)
output = OutputLayer(n_middle2, n_output)

lr = 0.1
epochs = 1000


# =============================================================
# 5. 学習
# =============================================================

loss_list = []

for epoch in range(epochs):
    # ---- forward ----
    layer1.forward(X)
    layer2.forward(layer1.y)
    output.forward(layer2.y)

    # ---- backward ----
    output.backward(Y)
    layer2.backward(output.grad_x)
    layer1.backward(layer2.grad_x)

    # ---- update ----
    layer1.update(lr)
    layer2.update(lr)
    output.update(lr)

    # ---- loss ----
    loss = -np.sum(Y * np.log(output.y + 1e-7)) / len(Y)
    loss_list.append(loss)

    if epoch % 100 == 0:
        print(f"epoch {epoch} loss = {loss}")


# =============================================================
# 6. グラフ表示
# =============================================================

plt.plot(loss_list)
plt.title("Training Loss (Softmax + CrossEntropy)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# =============================================================
# 7. 推論関数
# =============================================================
def predict(text):
    vec = log_vectorizer.transform([text]).toarray()
    layer1.forward(vec, train_flg=False)
    layer2.forward(layer1.y, train_flg=False)
    output.forward(layer2.y)

    idx = np.argmax(output.y)
    print("\n=== 推測結果 ===")
    print("推測カテゴリー:", encoder.categories_[0][idx])
