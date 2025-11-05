import numpy as np
import pickle

# ==== モデル読み込み ====
with open("error_category_model.pkl", "rb") as f:
    data = pickle.load(f)

log_vectorizer = data["log_vectorizer"]
encoder = data["encoder"]

# パラメータ復元
n_input     = data["n_input"]
n_middle1   = data["n_middle1"]
n_middle2   = data["n_middle2"]
n_output    = data["n_output"]
dropout_ratio = data["dropout_ratio"]

# ==== NN 再構築 ====
class MiddleLayer:
    def __init__(self, n_input, n_output):
        self.w = np.zeros((n_input, n_output))
        self.b = np.zeros(n_output)

    def forward(self, x):
        self.x = x
        self.u = np.dot(x, self.w) + self.b
        self.y = 1 / (1 + np.exp(-self.u))


class OutputLayer:
    def __init__(self, n_input, n_output):
        self.w = np.zeros((n_input, n_output))
        self.b = np.zeros(n_output)

    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        exp_u = np.exp(u - np.max(u, axis=1, keepdims=True))
        self.y = exp_u / np.sum(exp_u, axis=1, keepdims=True)


# ==== 層を再構築 ====
layer1 = MiddleLayer(n_input, n_middle1)
layer2 = MiddleLayer(n_middle1, n_middle2)
output = OutputLayer(n_middle2, n_output)

# ==== 保存された重みをセット ====
layer1.w, layer1.b = data["layer1_w"], data["layer1_b"]
layer2.w, layer2.b = data["layer2_w"], data["layer2_b"]
output.w, output.b = data["output_w"], data["output_b"]


# ==== 推論関数 ====
def predict(text):
    vec = log_vectorizer.transform([text]).toarray()
    layer1.forward(vec)
    layer2.forward(layer1.y)
    output.forward(layer2.y)

    idx = np.argmax(output.y)
    print("\n=== 推測結果 ===")
    print("推定カテゴリ:", encoder.categories_[0][idx])


# テスト例
predict("DBに接続できません TimeOutしました")