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