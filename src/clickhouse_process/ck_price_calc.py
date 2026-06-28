"""
ClickHouse 数据清洗与价格指数计算模块
--------------------------------------
功能:
  1. 从阿里云 OSS 下载原始 CSV 数据
  2. 连接 ClickHouse，创建电商表并导入数据
  3. 脏数据过滤（空值、负数、异常超大值）
  4. 计算日度分类价格指数（销量加权平均）

使用说明:
    python src/clickhouse_process/ck_price_calc.py

依赖: 需先运行 oss_upload.py 将数据上传至 OSS
"""
import os
import sys
import io
import clickhouse_connect
import pandas as pd
import oss2
from dotenv import load_dotenv

# 显式指定 .env 路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
    print(f"✅ 已加载配置文件: {ENV_PATH}")
else:
    print(f"⚠️  未找到 .env 文件: {ENV_PATH}")

# ============================================================
# 读取环境变量
# ============================================================
CK_HOST = os.getenv("CK_HOST")
CK_PORT = int(os.getenv("CK_PORT", "9000"))
CK_USER = os.getenv("CK_USER")
CK_PWD = os.getenv("CK_PASSWORD")
CK_DB = os.getenv("CK_DATABASE")

OSS_AK = os.getenv("OSS_ACCESS_KEY_ID")
OSS_SK = os.getenv("OSS_ACCESS_KEY_SECRET")
OSS_ENDPOINT = os.getenv("OSS_ENDPOINT")
OSS_BUCKET = os.getenv("OSS_BUCKET_NAME")

# ============================================================
# 全局变量：缓存下载的 DataFrame
# ============================================================
dfs = {}  # {"category": df, "product": df, "price": df}


def check_env():
    """校验必需环境变量。"""
    ck_vars = {"CK_HOST": CK_HOST, "CK_USER": CK_USER,
               "CK_PASSWORD": CK_PWD, "CK_DATABASE": CK_DB}
    oss_vars = {"OSS_ACCESS_KEY_ID": OSS_AK, "OSS_ACCESS_KEY_SECRET": OSS_SK,
                "OSS_ENDPOINT": OSS_ENDPOINT, "OSS_BUCKET": OSS_BUCKET}

    ck_missing = [k for k, v in ck_vars.items() if not v]
    oss_missing = [k for k, v in oss_vars.items() if not v]

    if ck_missing or oss_missing:
        print(f"\n❌ 环境变量未配置:")
        if ck_missing:
            print(f"   ClickHouse: {', '.join(ck_missing)}")
        if oss_missing:
            print(f"   OSS: {', '.join(oss_missing)}")
        print("   请编辑 .env 填入真实凭据后重试")
        sys.exit(1)


# ============================================================
# Step 1: 从 OSS 下载 CSV 到内存
# ============================================================
def download_from_oss():
    """使用 oss2 从 OSS Bucket 下载三个 CSV 文件到 pandas DataFrame。"""
    print("\n[1/5] 从 OSS 下载原始数据...")

    auth = oss2.Auth(OSS_AK, OSS_SK)
    bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET)

    files = {
        "category": "raw/category.csv",
        "product": "raw/product.csv",
        "price": "raw/price.csv",
    }

    for name, oss_path in files.items():
        try:
            obj = bucket.get_object(oss_path)
            df = pd.read_csv(io.BytesIO(obj.read()))
            dfs[name] = df
            print(f"  ✓ {name}: {len(df):,} 行, {list(df.columns)}")
        except Exception as e:
            print(f"  ✗ {name} 下载失败: {e}")
            print(f"    请确认 OSS 路径 oss://{OSS_BUCKET}/{oss_path} 存在")
            sys.exit(1)

    print(f"✅ OSS 数据下载完成 ({sum(len(d) for d in dfs.values()):,} 行)")


# ============================================================
# Step 2: 连接 ClickHouse
# ============================================================
def connect_ck():
    """连接 ClickHouse 并创建数据库。"""
    print("\n[2/5] 连接 ClickHouse...")
    try:
        client = clickhouse_connect.get_client(
            host=CK_HOST, port=CK_PORT,
            username=CK_USER, password=CK_PWD,
        )
        client.command(f"CREATE DATABASE IF NOT EXISTS {CK_DB}")
        client.command(f"USE {CK_DB}")
        print(f"✅ 已连接 {CK_HOST}:{CK_PORT}, 数据库 {CK_DB}")
        return client
    except Exception as e:
        print(f"❌ ClickHouse 连接失败: {e}")
        sys.exit(1)


# ============================================================
# Step 3: 建表 & 导入数据
# ============================================================
def create_tables_and_load(client):
    """创建本地表并从 DataFrame 导入数据。"""
    print("\n[3/5] 建表 & 导入数据...")

    # --- 3a. 建表 ---
    tables_ddl = {
        "dim_category": """
            CREATE TABLE IF NOT EXISTS dim_category (
                category_id Int32,
                category_name String
            ) ENGINE = MergeTree() ORDER BY category_id
        """,
        "dim_product": """
            CREATE TABLE IF NOT EXISTS dim_product (
                product_id Int64,
                category_id Int32,
                product_name String,
                brand String,
                spec String
            ) ENGINE = MergeTree() ORDER BY product_id
        """,
        "fact_price_clean": """
            CREATE TABLE IF NOT EXISTS fact_price_clean (
                product_id Int64,
                category_id Int32,
                price_date Date,
                price Float64,
                sales_volume Int64
            ) ENGINE = MergeTree() PARTITION BY toYYYYMM(price_date)
            ORDER BY (category_id, price_date)
        """,
    }

    for name, ddl in tables_ddl.items():
        client.command(ddl)
        print(f"  ✓ 建表: {name}")

    # --- 3b. 导入维度表 ---
    client.command("TRUNCATE TABLE dim_category")
    client.insert("dim_category", dfs["category"].values.tolist(),
                  column_names=list(dfs["category"].columns))
    print(f"  ✓ dim_category: {len(dfs['category'])} 行")

    client.command("TRUNCATE TABLE dim_product")
    client.insert("dim_product", dfs["product"].values.tolist(),
                  column_names=list(dfs["product"].columns))
    print(f"  ✓ dim_product: {len(dfs['product'])} 行")

    print("✅ 建表与维度数据导入完成")


# ============================================================
# Step 4: 清洗价格数据 & 计算指数
# ============================================================
def clean_and_calc(client):
    """清洗脏数据并计算加权价格指数。"""
    print("\n[4/5] 清洗数据 & 计算价格指数...")

    price_df = dfs["price"]

    # --- 4a. 清洗规则（与 data_generator 的脏数据注入对齐）---
    # price IS NOT NULL AND price > 0 AND price < 5000
    valid_mask = (
        price_df["price"].notna() &
        (price_df["price"] > 0) &
        (price_df["price"] < 5000)
    )
    clean_df = price_df[valid_mask].copy()
    total = len(price_df)
    clean_n = len(clean_df)

    print(f"  原始价格行数: {total:,}")
    print(f"  空值剔除: {(price_df['price'].isna().sum()):,}")
    print(f"  非正数剔除: {((price_df['price'].notna()) & (price_df['price'] <= 0)).sum():,}")
    print(f"  极端值剔除(≥5000): {((price_df['price'].notna()) & (price_df['price'] >= 5000)).sum():,}")
    print(f"  清洗后: {clean_n:,} 行 ({clean_n/total*100:.2f}%)")

    if clean_n == 0:
        print("❌ 清洗后无有效数据，请检查数据生成逻辑")
        sys.exit(1)

    # --- 4b. LEFT JOIN 获取 category_id ---
    prod_df = dfs["product"]
    merged = clean_df.merge(prod_df[["product_id", "category_id"]], on="product_id", how="left")
    # 过滤 orphan（商品表中不存在的 product_id）
    before_orphan = len(merged)
    merged = merged[merged["category_id"].notna()]
    if len(merged) < before_orphan:
        print(f"  orphan 商品剔除: {before_orphan - len(merged)} 行（product 表中不存在）")

    # --- 4c. 写入 fact_price_clean ---
    client.command("TRUNCATE TABLE fact_price_clean")
    # ⚠️ price_date 必须转为 datetime.date，CK Date 列不接受字符串
    merged["price_date"] = pd.to_datetime(merged["price_date"]).dt.date
    cols = ["product_id", "category_id", "price_date", "price", "sales_volume"]
    insert_data = merged[cols].values.tolist()
    client.insert("fact_price_clean", insert_data, column_names=cols)
    print(f"  ✓ fact_price_clean: {len(merged):,} 行")

    # --- 4d. 计算加权价格指数 ---
    # SUM(price * sales_volume) / SUM(sales_volume)
    client.command("DROP TABLE IF EXISTS daily_category_price_index")
    client.command("""
        CREATE TABLE daily_category_price_index (
            category_id Int32,
            category_name String,
            price_date Date,
            weighted_avg_price Float64,
            total_sales Int64,
            price_index Float64
        ) ENGINE = MergeTree() PARTITION BY toYYYYMM(price_date)
        ORDER BY (category_id, price_date)
    """)
    client.command("""
        INSERT INTO daily_category_price_index
        SELECT
            cat.category_id,
            cat.category_name,
            f.price_date,
            SUM(f.price * f.sales_volume) / SUM(f.sales_volume) AS weighted_avg_price,
            SUM(f.sales_volume) AS total_sales,
            round(
                (SUM(f.price * f.sales_volume) / SUM(f.sales_volume))
                / any(base.base_price) * 100,
                2
            ) AS price_index
        FROM fact_price_clean f
        LEFT JOIN dim_category cat ON f.category_id = cat.category_id
        LEFT JOIN (
            SELECT
                category_id,
                SUM(price * sales_volume) / SUM(sales_volume) AS base_price
            FROM fact_price_clean
            WHERE price_date = '2026-01-01'
            GROUP BY category_id
        ) base ON f.category_id = base.category_id
        GROUP BY cat.category_id, cat.category_name, f.price_date
    """)
    print("✅ 加权价格指数计算完成")


# ============================================================
# Step 5: 导出结果
# ============================================================
def export_results(client):
    """导出指数结果为 CSV，供可视化模块使用。"""
    print("\n[5/5] 导出结果...")

    df_index = client.query_df(
        "SELECT * FROM daily_category_price_index ORDER BY category_id, price_date"
    )
    print(f"  指数数据集: {len(df_index):,} 行")
    print(f"  分类数: {df_index['category_name'].nunique()}")
    print(f"  日期范围: {df_index['price_date'].min()} ~ {df_index['price_date'].max()}")

    # 统计摘要
    print(f"\n  各分类加权均价均值 & 价格指数均值（基期 2026-01-01 = 100）:")
    summary_price = df_index.groupby("category_name")["weighted_avg_price"].mean().sort_values(ascending=False)
    summary_index = df_index.groupby("category_name")["price_index"].mean().sort_values(ascending=False)
    for cat in summary_price.index:
        print(f"    {cat}: ¥{summary_price[cat]:.2f} | 指数均值: {summary_index[cat]:.2f}")

    output_path = os.path.join(PROJECT_ROOT, "price_index_result.csv")
    df_index.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n✅ 结果已保存: {output_path}")


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("ClickHouse 数据清洗 & 价格指数计算")
    print("=" * 60)

    check_env()
    download_from_oss()
    client = connect_ck()
    create_tables_and_load(client)
    clean_and_calc(client)
    export_results(client)

    print("\n" + "=" * 60)
    print("全流程执行完毕!")
    print("=" * 60)


if __name__ == "__main__":
    main()
