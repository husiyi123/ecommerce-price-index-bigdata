"""
数据可视化模块 — 日度 CPI 价格指数趋势图绘制
--------------------------------------------
读取 ClickHouse 计算模块的输出结果，生成精美的 CPI 趋势可视化图表。

使用说明:
    python src/visualization/draw_cpi_trend.py

上游依赖: src/clickhouse_process/ck_price_calc.py（需先计算价格指数）
输出: data/result/ 目录下的 CPI 趋势图 PNG/SVG
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# 读取ClickHouse输出的指数数据
df = pd.read_csv("./price_index_result.csv", parse_dates=["price_date"])

# 设置画布
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 中文显示
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(14, 7))

# 按分类分组绘制折线
category_list = df["category_name"].unique()
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

for idx, cate in enumerate(category_list):
    sub_df = df[df["category_name"] == cate].sort_values("price_date")
    ax.plot(sub_df["price_date"], sub_df["weighted_avg_price"], label=cate, color=colors[idx], linewidth=1.2)

# 图表美化
ax.set_title("电商日度分类加权价格指数趋势图（高频CPI监测）", fontsize=16, pad=15)
ax.set_xlabel("日期", fontsize=12)
ax.set_ylabel("销量加权平均价格（指数基准）", fontsize=12)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=30)
ax.legend(loc="upper right")
ax.grid(alpha=0.3)

# 保存图片
plt.tight_layout()
plt.savefig("./cpi_price_trend.png", dpi=300)
plt.show()
print("✅ 趋势图已保存 cpi_price_trend.png")
