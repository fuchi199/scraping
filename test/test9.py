import numpy as np
import matplotlib.pyplot as plt
import pickle

# ===============================
# ハイパーパラメータ
# ===============================
n_in = 800       # TF-IDF 次元
n_mid = 1500     # 中間層ニューロン数
n_out = 20       # カテゴリ数

dropout_ratio = 0.3
eta = 0.003        # 学習率
epoch = 500
batch_size = 32

patience = 30      # EarlyStopping
min_delta = 1e-5   # 改善しないと判断するしきい値

# ===============================
# 自作レイヤー
# ===============================
class BaseLayer:
    def __init__(self, n_upper, n):
        self.w = 0.1 * np.random.randn(n_upper, n)
        self.b = 0.1 * np.random.randn(n)

        self.h_w = np.zeros((n_upper, n)) + 1e-8
        self.h_b = np.zeros(n) + 1e-8

    def update(self, eta):
        self.h_w += self.grad_w * self.grad_w
        self.w -= eta / np.sqrt(self.h_w) * self.grad_w

        self.h_b += self.grad_b * self.grad_b
        self.b -= eta / np.sqrt(self.h_b) * self.grad_b


class MiddleLayer(BaseLayer):
    def forward(self, x):
        self.x = x
        self.u = np.dot(x, self.w) + self.b
        self.y = np.maximum(self.u, 0)  # ReLU

    def backward(self, grad_y):
        delta = grad_y * (self.u > 0)
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)


class OutputLayer(BaseLayer):
    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = np.exp(u) / np.sum(np.exp(u), axis=1, keepdims=True)

    def backward(self, t):
        delta = self.y - t
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)


class Dropout:
    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio

    def forward(self, x, is_train):
        if is_train:
            self.mask = (np.random.rand(*x.shape) > self.dropout_ratio).astype(np.float32)
            self.y = x * self.mask
        else:
            self.y = x * (1.0 - self.dropout_ratio)

    def backward(self, grad_y):
        self.grad_x = grad_y * self.mask


# ===============================
# ネットワーク構築
# ===============================
middle1 = MiddleLayer(n_in, n_mid)
drop1 = Dropout(dropout_ratio)
middle2 = MiddleLayer(n_mid, n_mid)
drop2 = Dropout(dropout_ratio)
output = OutputLayer(n_mid, n_out)


# ===============================
# Forward / Backprop / Loss
# ===============================
def forward(x, is_train=True):
    middle1.forward(x)
    drop1.forward(middle1.y, is_train)
    middle2.forward(drop1.y)
    drop2.forward(middle2.y, is_train)
    output.forward(drop2.y)

def backward(t):
    output.backward(t)
    drop2.backward(output.grad_x)
    middle2.backward(drop2.grad_x)
    drop1.backward(middle2.grad_x)
    middle1.backward(drop1.grad_x)

def update():
    middle1.update(eta)
    middle2.update(eta)
    output.update(eta)

def get_loss(t, batch_size):
    return -np.sum(t * np.log(output.y + 1e-7)) / batch_size

def accuracy(y_pred, y_true):
    pred = np.argmax(y_pred, axis=1)
    true = np.argmax(y_true, axis=1)
    return np.mean(pred == true)


# ===============================
# 学習ループ
# ===============================
train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

best_loss = float("inf")
wait = 0

n_train = len(input_train)
n_batch = n_train // batch_size

for ep in range(epoch):

    # --------------- Shuffle minibatch ---------------
    index = np.random.permutation(n_train)

    for i in range(0, n_train, batch_size):
        mb_index = index[i : i + batch_size]
        x = input_train[mb_index]
        t = correct_train[mb_index]

        forward(x, True)
        backward(t)
        update()

    # --------------- 評価（Train / Test）------------
    forward(input_train, False)
    train_loss = get_loss(correct_train, n_train)
    train_acc = accuracy(output.y, correct_train)

    forward(input_test, False)
    test_loss = get_loss(correct_test, len(correct_test))
    test_acc = accuracy(output.y, correct_test)

    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    print(f"Epoch {ep}/{epoch}  loss: {train_loss:.4f}/{test_loss:.4f}  acc: {train_acc:.4f}/{test_acc:.4f}")

    # ---------- EarlyStopping ----------
    if test_loss < best_loss - min_delta:
        best_loss = test_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"EarlyStopping 発動！(ep={ep})")
            break


# ===============================
# モデル保存
# ===============================
with open("model_weights.pkl", "wb") as f:
    pickle.dump([
        middle1.w, middle1.b,
        middle2.w, middle2.b,
        output.w, output.b,
    ], f)

print("✅ モデル保存完了： model_weights.pkl")


# ===============================
# グラフ描画
# ===============================
plt.figure(figsize=(12,5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label="Train Loss")
plt.plot(test_loss_list, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label="Train Accuracy")
plt.plot(test_acc_list, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")

plt.show()





from sentence_transformers import SentenceTransformer

# モデル読み込み（多言語）
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

# テキスト → ベクトル化
embeddings = model.encode([
    "Network timeout error",
    "接続がタイムアウトしました。",
])

print(embeddings.shape)  # (2, 512)

tokenizer = model.tokenizer

text = "DxSuite API"
tokens = tokenizer.tokenize(text)

print(tokens)




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# -------------------------------
# 1. TF-IDF + SVD（次元削減）
# -------------------------------

texts = error_text_list          # ← エラーログ（文字列リスト）
labels = error_category_list     # ← カテゴリー（文字列リスト）

vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1,2))
X_tfidf = vectorizer.fit_transform(texts)

svd = TruncatedSVD(n_components=300)
X = svd.fit_transform(X_tfidf)



