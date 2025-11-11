import numpy as np
import matplotlib.pyplot as plt

# ============================================
# ハイパーパラメータ設定
# ============================================
n_in = 800     # 入力次元（TF-IDF）
n_mid = 600    # 中間層
n_out = 20     # カテゴリ数
dropout_ratio = 0.6

wb_width = 0.1
eta = 0.005
epoch = 300
batch_size = 32
interval = 20

L2_lambda = 1e-4          # L2 正則化
patience = 20             # Early Stopping
min_delta = 1e-5
best_val_loss = np.inf
wait = 0


# ============================================
# ★★★ あなたの TF-IDF 結果をここに入れる ★★★
# input_train, correct_train
# input_test, correct_test
# ============================================

# ※ ダミー（あなたのデータに置き換えてください）
n_train = 800
n_test = 200
input_train = np.random.randn(n_train, n_in)
input_test = np.random.randn(n_test, n_in)
correct_train = np.eye(n_out)[np.random.choice(n_out, n_train)]
correct_test  = np.eye(n_out)[np.random.choice(n_out, n_test)]


# ============================================
# 層の定義
# ============================================
class BaseLayer:
    def __init__(self, n_upper, n):
        self.w = wb_width * np.random.randn(n_upper, n)
        self.b = wb_width * np.random.randn(n)

        self.h_w = np.zeros((n_upper, n)) + 1e-8
        self.h_b = np.zeros(n) + 1e-8

    def update(self, eta):
        self.h_w += self.grad_w * self.grad_w
        self.w -= (eta / np.sqrt(self.h_w)) * self.grad_w

        self.h_b += self.grad_b * self.grad_b
        self.b -= (eta / np.sqrt(self.h_b)) * self.grad_b


class MiddleLayer(BaseLayer):
    def forward(self, x):
        self.x = x
        self.u = np.dot(x, self.w) + self.b
        self.y = np.maximum(0, self.u)  # ReLU

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

    def forward(self, x, is_training):
        if is_training:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            self.y = x * self.mask
        else:
            self.y = x * (1 - self.dropout_ratio)

    def backward(self, grad_y):
        self.grad_x = grad_y * self.mask


# ============================================
# ネットワーク構成
# ============================================
layer1 = MiddleLayer(n_in, n_mid)
drop1  = Dropout(dropout_ratio)
layer2 = MiddleLayer(n_mid, n_mid)
drop2  = Dropout(dropout_ratio)
output = OutputLayer(n_mid, n_out)


# ============================================
# Forward / Backprop 処理
# ============================================
def forward(x, is_train):
    layer1.forward(x)
    drop1.forward(layer1.y, is_train)
    layer2.forward(drop1.y)
    drop2.forward(layer2.y, is_train)
    output.forward(drop2.y)

def backward(t):
    output.backward(t)
    drop2.backward(output.grad_x)
    layer2.backward(drop2.grad_x)
    drop1.backward(layer2.grad_x)
    layer1.backward(drop1.grad_x)

def update_weight():
    layer1.grad_w += L2_lambda * layer1.w
    layer2.grad_w += L2_lambda * layer2.w
    output.grad_w += L2_lambda * output.w

    layer1.update(eta)
    layer2.update(eta)
    output.update(eta)

def get_loss(t, batch_size):
    return -np.sum(t * np.log(output.y + 1e-7)) / batch_size


# ============================================
# モデル保存 / 復元
# ============================================

def save_model(filename="model_weights.npz"):
    np.savez(filename,
             l1_w=layer1.w, l1_b=layer1.b,
             l2_w=layer2.w, l2_b=layer2.b,
             o_w=output.w, o_b=output.b)
    print(f"✅ モデル保存しました → {filename}")

def load_model(filename="model_weights.npz"):
    data = np.load(filename)
    layer1.w, layer1.b = data["l1_w"], data["l1_b"]
    layer2.w, layer2.b = data["l2_w"], data["l2_b"]
    output.w, output.b = data["o_w"], data["o_b"]
    print(f"✅ モデルを読み込みました → {filename}")


# ============================================
# 学習ループ
# ============================================
train_loss_hist = []
test_loss_hist = []

best_model = None

for ep in range(1, epoch + 1):

    perm = np.random.permutation(n_train)

    for i in range(0, n_train, batch_size):
        batch = perm[i:i + batch_size]
        forward(input_train[batch], True)
        backward(correct_train[batch])
        update_weight()

    forward(input_train, False)
    train_loss = get_loss(correct_train, n_train)

    forward(input_test, False)
    test_loss = get_loss(correct_test, n_test)

    train_loss_hist.append(train_loss)
    test_loss_hist.append(test_loss)

    # EarlyStopping
    if test_loss + min_delta < best_val_loss:
        best_val_loss = test_loss
        wait = 0
        save_model()    # ★ここで保存
    else:
        wait += 1

    if ep % interval == 0:
        print(f"Epoch {ep}/{epoch} | Train Loss {train_loss:.4f} | Test Loss {test_loss:.4f}")

    if wait >= patience:
        print("⛔ Early stopping")
        break


# ============================================
# グラフ表示
# ============================================
plt.plot(train_loss_hist, label="Train Loss")
plt.plot(test_loss_hist, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()