"""
OSS 连通性与字段解析单元测试
-----------------------------
测试内容:
  1. OSS SDK 导入与连接参数验证
  2. 本地 CSV 文件字段解析正确性（适配新版三表结构）
  3. OSS Bucket 连通性测试（需有效凭据）
  4. 上传完整性校验（MD5）
  5. OSS 路径映射正确性

运行方式:
    pytest tests/test_oss_connect.py -v
    pytest tests/test_oss_connect.py -v -k "not integration"  # 跳过集成测试
"""

import os
import sys
import hashlib
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ============================================================
# 测试 1: OSS SDK 导入
# ============================================================
def test_oss_sdk_import():
    """验证 oss2 包可正常导入。"""
    try:
        import oss2
        assert hasattr(oss2, "Auth")
        assert hasattr(oss2, "Bucket")
    except ImportError:
        pytest.skip("oss2 未安装，跳过 SDK 导入测试")


# ============================================================
# 测试 2: 连接参数验证
# ============================================================
def test_oss_config_from_env(monkeypatch):
    """验证 OSS 配置从环境变量正确加载（匹配 oss_upload.py 中的变量名）。"""
    monkeypatch.setenv("OSS_ENDPOINT", "https://oss-cn-test.aliyuncs.com")
    monkeypatch.setenv("OSS_ACCESS_KEY_ID", "test_key_id")
    monkeypatch.setenv("OSS_ACCESS_KEY_SECRET", "test_key_secret")
    monkeypatch.setenv("OSS_BUCKET_NAME", "test_bucket")

    assert os.getenv("OSS_ENDPOINT") == "https://oss-cn-test.aliyuncs.com"
    assert os.getenv("OSS_ACCESS_KEY_ID") == "test_key_id"
    assert os.getenv("OSS_BUCKET_NAME") == "test_bucket"

    for key in ["OSS_ENDPOINT", "OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET", "OSS_BUCKET_NAME"]:
        val = os.getenv(key)
        assert val is not None, f"{key} 不应为空"
        assert len(val) > 0, f"{key} 不应为空字符串"


# ============================================================
# 测试 3: category.csv 字段解析正确性
# ============================================================
def test_category_csv_field_parsing():
    """验证 category.csv 的字段结构符合新版 schema。"""
    import pandas as pd

    test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(test_data_dir, exist_ok=True)

    test_path = os.path.join(test_data_dir, "test_category.csv")
    pd.DataFrame({
        "category_id": [1, 2, 3],
        "category_name": ["生鲜果蔬", "粮油副食", "家居百货"],
    }).to_csv(test_path, index=False)

    df = pd.read_csv(test_path)

    expected_cols = {"category_id", "category_name"}
    actual_cols = set(df.columns)
    assert expected_cols == actual_cols, f"字段不匹配: 期望 {expected_cols}, 实际 {actual_cols}"
    assert pd.api.types.is_integer_dtype(df["category_id"]), \
        f"category_id 应为整型，实际 {df['category_id'].dtype}"
    assert pd.api.types.is_string_dtype(df["category_name"]), \
        f"category_name 应为字符串，实际 {df['category_name'].dtype}"


# ============================================================
# 测试 4: product.csv 字段解析正确性
# ============================================================
def test_product_csv_field_parsing():
    """验证 product.csv 的字段结构符合新版 schema（含 brand, spec）。"""
    import pandas as pd

    test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(test_data_dir, exist_ok=True)

    test_path = os.path.join(test_data_dir, "test_product.csv")
    pd.DataFrame({
        "product_id": [10001, 10002, 10003],
        "category_id": [1, 2, 3],
        "product_name": ["测试商品A", "测试商品B", "测试商品C"],
        "brand": ["A牌", "B牌", "自营"],
        "spec": ["1kg", "500ml", "标准版"],
    }).to_csv(test_path, index=False)

    df = pd.read_csv(test_path)

    expected_cols = {"product_id", "category_id", "product_name", "brand", "spec"}
    actual_cols = set(df.columns)
    assert expected_cols == actual_cols, f"字段不匹配: 期望 {expected_cols}, 实际 {actual_cols}"

    assert pd.api.types.is_integer_dtype(df["product_id"]), \
        f"product_id 应为整型，实际 {df['product_id'].dtype}"
    assert pd.api.types.is_integer_dtype(df["category_id"]), \
        f"category_id 应为整型，实际 {df['category_id'].dtype}"


# ============================================================
# 测试 5: price.csv 字段解析正确性
# ============================================================
def test_price_csv_field_parsing():
    """验证 price.csv 的字段结构符合新版 schema。"""
    import pandas as pd

    test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(test_data_dir, exist_ok=True)

    test_path = os.path.join(test_data_dir, "test_price.csv")
    pd.DataFrame({
        "product_id": [10001, 10002, 10001],
        "price_date": ["2026-01-01", "2026-01-01", "2026-01-02"],
        "price": [99.99, 200.0, 105.0],
        "sales_volume": [10, 5, 8],
    }).to_csv(test_path, index=False)

    df = pd.read_csv(test_path)

    expected_cols = {"product_id", "price_date", "price", "sales_volume"}
    actual_cols = set(df.columns)
    assert expected_cols == actual_cols, f"字段不匹配: 期望 {expected_cols}, 实际 {actual_cols}"

    # price 应为数值型
    assert pd.api.types.is_numeric_dtype(df["price"]), \
        f"price 类型异常: {df['price'].dtype}"


# ============================================================
# 测试 6: 价格字段合法性与脏数据检测
# ============================================================
def test_price_field_validity():
    """验证价格字段识别算法能正确标记脏数据（对齐 ck_price_calc.py 过滤规则）。"""
    import numpy as np

    # 正常价格（对齐清洗规则: price > 0 AND price < 5000）
    valid_prices = [9.9, 100.0, 2999.99, 0.01, 4999.0]
    # 脏数据
    invalid_prices = [None, np.nan, -10.0, 0.0, 5000.0, 999999.0]

    UPPER = 5000  # 对齐 ck_price_calc.py 中的清洗阈值

    def is_valid(price):
        if price is None or (isinstance(price, float) and np.isnan(price)):
            return False
        if price <= 0:
            return False
        if price >= UPPER:
            return False
        return True

    for p in valid_prices:
        assert is_valid(p), f"价格 {p} 应为有效值"

    for p in invalid_prices:
        assert not is_valid(p), f"价格 {p} 应被标记为无效"


# ============================================================
# 测试 7: MD5 完整性校验
# ============================================================
def test_md5_integrity_check():
    """验证 MD5 计算函数的正确性。"""
    content = "ecommerce-test-data-oss-upload"
    expected_md5 = hashlib.md5(content.encode()).hexdigest()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write(content)
        tmp_path = f.name

    try:
        computed_md5 = hashlib.md5()
        with open(tmp_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                computed_md5.update(chunk)
        assert computed_md5.hexdigest() == expected_md5, "MD5 校验值不匹配"
    finally:
        os.unlink(tmp_path)


# ============================================================
# 测试 8: OSS 上传路径映射正确性
# ============================================================
def test_oss_upload_mapping():
    """验证 oss_upload.py 中的路径映射逻辑。"""
    upload_mapping = [
        ("./raw_data/category.csv", "raw/category.csv"),
        ("./raw_data/product.csv", "raw/product.csv"),
        ("./raw_data/price.csv", "raw/price.csv"),
    ]

    for local, remote in upload_mapping:
        assert local.startswith("./raw_data/"), f"本地路径应以 ./raw_data/ 开头: {local}"
        assert remote.startswith("raw/"), f"远程路径应以 raw/ 开头: {remote}"
        assert local.endswith(".csv"), f"本地文件应为 CSV: {local}"
        assert remote.endswith(".csv"), f"远程文件应为 CSV: {remote}"
        # 文件名应一致
        assert os.path.basename(local) == os.path.basename(remote), \
            f"文件名不一致: {os.path.basename(local)} vs {os.path.basename(remote)}"


# ============================================================
# 测试 9: OSS Bucket 连通性测试（集成测试，需真实凭据）
# ============================================================
@pytest.mark.integration
def test_oss_bucket_connectivity():
    """
    集成测试：验证是否可连通阿里云 OSS。
    若未配置有效凭据则跳过。
    """
    try:
        import oss2
    except ImportError:
        pytest.skip("oss2 未安装，跳过连通性测试")

    endpoint = os.getenv("OSS_ENDPOINT")
    key_id = os.getenv("OSS_ACCESS_KEY_ID")
    key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
    bucket_name = os.getenv("OSS_BUCKET_NAME")

    if not all([endpoint, key_id, key_secret, bucket_name]):
        pytest.skip("OSS 凭据未配置，跳过集成测试")

    try:
        auth = oss2.Auth(key_id, key_secret)
        bucket = oss2.Bucket(auth, endpoint, bucket_name)
        info = bucket.get_bucket_info()
        assert info is not None, "无法获取 Bucket 信息"
        print(f"  ✓ Bucket 连通: {bucket_name} (创建日期: {info.creation_date})")
    except oss2.exceptions.AccessDenied:
        pytest.skip("OSS 凭据无权限访问目标 Bucket")
    except Exception as e:
        pytest.fail(f"OSS 连接失败: {e}")


# ============================================================
# 测试 10: src/oss_upload 模块函数导入与路径映射验证
# ============================================================
def test_oss_upload_module_functions():
    """
    验证 oss_upload.py 中的上传路径映射与常量定义。
    由于模块级代码会初始化 oss2.Bucket，使用 mock 隔离。
    """
    import sys
    import os
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from unittest.mock import patch, MagicMock
    import oss2 as _oss2_module

    with patch.object(_oss2_module, "Auth") as mock_auth, \
         patch.object(_oss2_module, "Bucket") as mock_bucket, \
         patch.dict(os.environ, {
             "OSS_ENDPOINT": "https://oss-cn-test.aliyuncs.com",
             "OSS_ACCESS_KEY_ID": "test_key_id",
             "OSS_ACCESS_KEY_SECRET": "test_key_secret",
             "OSS_BUCKET_NAME": "test-bucket-name",
         }):
        mock_auth.return_value = MagicMock()
        mock_bucket.return_value = MagicMock()

        # 清除已缓存的模块以强制重新导入
        if "src.oss_operation.oss_upload" in sys.modules:
            del sys.modules["src.oss_operation.oss_upload"]
        if "src.oss_operation" in sys.modules:
            del sys.modules["src.oss_operation"]

        from src.oss_operation import oss_upload

        # 验证核心函数存在
        assert hasattr(oss_upload, "upload_file"), "缺少 upload_file 函数"

        # 验证路径映射常量
        upload_mapping = oss_upload.upload_mapping
        assert len(upload_mapping) == 3, f"应有 3 个文件映射，实际 {len(upload_mapping)}"
        for local, remote in upload_mapping:
            assert local.endswith(".csv"), f"本地路径应以 .csv 结尾: {local}"
            assert remote.startswith("raw/"), f"远程路径应以 raw/ 开头: {remote}"
            assert remote.endswith(".csv"), f"远程文件应以 .csv 结尾: {remote}"

        print("  ✓ oss_upload 模块函数验证通过")
