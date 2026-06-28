"""
阿里云 OSS 对象存储上传模块
将本地生成的原始 CSV 数据批量上传至阿里云 OSS Bucket。

使用说明:
    1. cp config/.env.example .env  # 复制模板
    2. 编辑 .env 填写真实的阿里云 OSS 凭据
    3. python src/oss_operation/oss_upload.py

上游依赖: src/data_generation/data_generator.py（需先生成原始数据）
下游消费: src/clickhouse_process/ck_price_calc.py（从 OSS 导入）
"""
import os
import sys
import oss2
from dotenv import load_dotenv

# 显式指定 .env 路径（解决跨目录运行的路径问题）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
    print(f"✅ 已加载配置文件: {ENV_PATH}")
else:
    print(f"⚠️  未找到 .env 文件: {ENV_PATH}")
    print("   请执行: cp config/.env.example .env")
    print("   然后编辑 .env 填入真实的阿里云 OSS 凭据")

# OSS 配置
AK = os.getenv("OSS_ACCESS_KEY_ID")
SK = os.getenv("OSS_ACCESS_KEY_SECRET")
ENDPOINT = os.getenv("OSS_ENDPOINT")
BUCKET_NAME = os.getenv("OSS_BUCKET_NAME")

# ============================================================
# 凭据校验（友好报错）
# ============================================================
missing = []
if not AK:
    missing.append("OSS_ACCESS_KEY_ID")
if not SK:
    missing.append("OSS_ACCESS_KEY_SECRET")
if not ENDPOINT:
    missing.append("OSS_ENDPOINT")
if not BUCKET_NAME:
    missing.append("OSS_BUCKET_NAME")

if missing:
    print(f"\n❌ 错误: 以下环境变量未配置: {', '.join(missing)}")
    print("   请按以下步骤操作:")
    print("   1. cp config/.env.example .env")
    print("   2. 编辑 .env 填入真实的阿里云 OSS 凭据")
    print("   3. 重新运行: python src/oss_operation/oss_upload.py")
    sys.exit(1)

# 初始化 OSS 客户端
auth = oss2.Auth(AK, SK)
bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)
print(f"✅ OSS 客户端初始化成功: oss://{BUCKET_NAME}")

# 本地文件路径 & OSS 远程路径映射
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw_data")
upload_mapping = [
    (os.path.join(RAW_DATA_DIR, "category.csv"), "raw/category.csv"),
    (os.path.join(RAW_DATA_DIR, "product.csv"), "raw/product.csv"),
    (os.path.join(RAW_DATA_DIR, "price.csv"), "raw/price.csv"),
]


def upload_file(local_path, oss_path):
    """上传单个文件到 OSS，含完整性校验。"""
    if not os.path.exists(local_path):
        print(f"⚠️  跳过: 本地文件不存在 {local_path}")
        return False

    file_size = os.path.getsize(local_path)
    print(f"↑ 上传: {os.path.basename(local_path)} ({file_size / 1024:.1f} KB) → oss://{BUCKET_NAME}/{oss_path}")

    bucket.put_object_from_file(oss_path, local_path)

    # 验证上传
    try:
        head = bucket.head_object(oss_path)
        print(f"✅ {oss_path} 上传成功 (ETag: {head.etag})")
        return True
    except Exception as e:
        print(f"❌ {oss_path} 上传验证失败: {e}")
        return False


if __name__ == "__main__":
    success_count = 0
    for local, remote in upload_mapping:
        if upload_file(local, remote):
            success_count += 1

    print(f"\n=== 上传完成: {success_count}/{len(upload_mapping)} 个文件 ===")
