import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 1. データ読み込み
print("=== Loading Data ===")
df = pd.read_parquet('data/sales_df.parquet')
df = df.rename(columns={'日付': 'date', '書店コード': 'store_id', '書名': 'book_id', 'POS販売冊数': 'sales'})

# 2. 平均と分散を計算（店舗×商品ごと）
print("=== Calculating Mean-Variance ===")
stats = df.groupby(['store_id', 'book_id'])['sales'].agg(['mean', 'var']).reset_index()

# 対数を取るため、0より大きいデータに限定
stats = stats[(stats['mean'] > 0) & (stats['var'] > 0)]
stats['log_mean'] = np.log(stats['mean'])
stats['log_var'] = np.log(stats['var'])

# 3. 回帰分析で p を推定
X = stats[['log_mean']]
y = stats['log_var']
model = LinearRegression()
model.fit(X, y)

best_p = model.coef_[0]
print(f"\nEstimated Optimal p: {best_p:.4f}")

# 4. 可視化
plt.figure(figsize=(8, 6))
plt.scatter(stats['log_mean'], stats['log_var'], alpha=0.1, s=10)
x_range = np.linspace(stats['log_mean'].min(), stats['log_mean'].max(), 100)
plt.plot(x_range, model.predict(x_range.reshape(-1, 1)), 'r-', label=f'p={best_p:.2f}')
plt.xlabel('log(Mean)')
plt.ylabel('log(Variance)')
plt.title(f'Mean-Variance Plot (p={best_p:.2f})')
plt.legend()
plt.savefig('mean_variance_plot.png')
print("Saved plot.")