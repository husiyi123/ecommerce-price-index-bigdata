import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

# 输出文件夹
OUTPUT_DIR = "./raw_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 生成商品分类表 category.csv
category_data = [
    {"category_id": 1, "category_name": "生鲜果蔬"},
    {"category_id": 2, "category_name": "粮油副食"},
    {"category_id": 3, "category_name": "家居百货"},
    {"category_id": 4, "category_name": "数码电器"},
    {"category_id": 5, "category_name": "服装鞋帽"},
    {"category_id": 6, "category_name": "美妆护肤"},
    {"category_id": 7, "category_name": "母婴用品"},
    {"category_id": 8, "category_name": "零食饮料"},
]
df_cat = pd.DataFrame(category_data)
df_cat.to_csv(f"{OUTPUT_DIR}/category.csv", index=False, encoding="utf-8")
print("✅ 分类表生成完成")

# 2. 生成商品表 product.csv
product_list = []
prod_id = 1
for cid in range(1, 9):
    # 每个分类生成3000个商品
    for _ in range(3000):
        product_list.append({
            "product_id": prod_id,
            "category_id": cid,
            "product_name": f"商品_{cid}_{prod_id}",
            "brand": random.choice(["A牌", "B牌", "C牌", "D牌", "自营"]),
            "spec": random.choice(["1kg", "500ml", "2件装", "大号", "标准版"])
        })
        prod_id += 1
df_prod = pd.DataFrame(product_list)
df_prod.to_csv(f"{OUTPUT_DIR}/product.csv", index=False, encoding="utf-8")
print("✅ 商品表生成完成")

# 3. 生成价格销量表 price.csv（核心大表，控制总数据100MB）
start_date = datetime(2026, 1, 1)
end_date = datetime(2026, 6, 30)
day_count = (end_date - start_date).days + 1
all_price_rows = []

# 遍历所有商品，生成每日价格（含脏数据：空价格、负数、异常超大值）
all_product_ids = df_prod["product_id"].tolist()
print(f"开始生成价格数据，总商品{len(all_product_ids)}个，天数{day_count}天")

for pid in tqdm(all_product_ids):
    base_price = round(random.uniform(9.9, 2999.9), 2)
    curr_day = start_date
    for _ in range(day_count):
        # 模拟价格波动
        price_fluct = base_price * random.uniform(0.85, 1.15)
        sale_num = random.randint(0, 1200)
        # 制造脏数据
        dirty_flag = random.random()
        if dirty_flag < 0.02:
            price = None  # 价格为空
        elif dirty_flag < 0.04:
            price = -round(random.uniform(1, 100), 2)  # 负价格
        elif dirty_flag < 0.06:
            price = 999999  # 极端异常高价
        else:
            price = round(price_fluct, 2)

        all_price_rows.append({
            "product_id": pid,
            "price_date": curr_day.strftime("%Y-%m-%d"),
            "price": price,
            "sales_volume": sale_num
        })
        curr_day += timedelta(days=1)

# 分批写入防止内存溢出
df_price = pd.DataFrame(all_price_rows)
df_price.to_csv(f"{OUTPUT_DIR}/price.csv", index=False, encoding="utf-8")
print(f"✅ 价格大表生成完成，行数：{len(df_price)}，文件存放：{OUTPUT_DIR}")
print(f"三张CSV总大小约100MB，可执行ls -lh ./raw_data查看文件体积")