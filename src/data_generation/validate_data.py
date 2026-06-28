"""
数据格式校验与字段类型转换模块
------------------------------
对生成的电商三表 CSV 进行:
  1. 字段完整性校验（列名、列数）
  2. 数据类型校验与自动转换
  3. 业务逻辑校验（外键引用、值域范围）
  4. 脏数据统计与报告生成

使用说明:
    python src/data_generation/validate_data.py

输入: raw_data/*.csv（由 data_generator.py 生成）
输出: raw_data/validation_report.txt（校验报告）
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np

# ============================================================
# 配置
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "raw_data")
REPORT_PATH = os.path.join(DATA_DIR, "validation_report.txt")

# 期望的字段定义
EXPECTED_SCHEMA = {
    "category.csv": {
        "columns": ["category_id", "category_name"],
        "types": {"category_id": "int64", "category_name": "object"},
        "not_null": ["category_id", "category_name"],
        "pk": "category_id",
    },
    "product.csv": {
        "columns": ["product_id", "category_id", "product_name", "brand", "spec"],
        "types": {"product_id": "int64", "category_id": "int64", "product_name": "object",
                  "brand": "object", "spec": "object"},
        "not_null": ["product_id", "category_id", "product_name"],
        "pk": "product_id",
        "fk": {"category_id": "category.csv:category_id"},
    },
    "price.csv": {
        "columns": ["product_id", "price_date", "price", "sales_volume"],
        "types": {"product_id": "int64", "price_date": "object", "price": "float64",
                  "sales_volume": "int64"},
        "not_null": ["product_id", "price_date"],
        "fk": {"product_id": "product.csv:product_id"},
    },
}


class ValidationReport:
    """校验报告收集器。"""

    def __init__(self, report_path: str):
        self.report_path = report_path
        self.lines = []
        self.errors = 0
        self.warnings = 0
        self.passes = 0

    def add(self, level: str, msg: str):
        prefix = {"PASS": "✅", "WARN": "⚠️", "ERROR": "❌", "INFO": "ℹ️"}
        line = f"[{level}] {msg}"
        self.lines.append(line)
        print(f"  {prefix.get(level, '')} {line}")
        if level == "ERROR":
            self.errors += 1
        elif level == "WARN":
            self.warnings += 1
        elif level == "PASS":
            self.passes += 1

    def save(self):
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(f"数据格式校验报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"通过: {self.passes}  警告: {self.warnings}  错误: {self.errors}\n")
            f.write(f"{'=' * 60}\n\n")
            for line in self.lines:
                f.write(line + "\n")
        print(f"\n📄 校验报告已保存: {self.report_path}")


def validate_file_exists(report: ValidationReport) -> dict:
    """验证三个必需 CSV 文件是否存在。"""
    report.add("INFO", f"数据目录: {os.path.abspath(DATA_DIR)}")
    dfs = {}
    for fname in ["category.csv", "product.csv", "price.csv"]:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            report.add("ERROR", f"文件缺失: {fname}")
        else:
            size_kb = os.path.getsize(fpath) / 1024
            report.add("PASS", f"文件存在: {fname} ({size_kb:.2f} KB)")
            dfs[fname] = pd.read_csv(fpath, low_memory=False)
    return dfs


def validate_schema(dfs: dict, report: ValidationReport):
    """字段完整性校验：列名、列数、缺失字段。"""
    for fname, expected in EXPECTED_SCHEMA.items():
        if fname not in dfs:
            continue
        df = dfs[fname]
        expected_cols = set(expected["columns"])
        actual_cols = set(df.columns)

        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols

        if not missing and not extra:
            report.add("PASS", f"{fname}: 字段完整匹配 ({len(actual_cols)} 列)")
        else:
            if missing:
                report.add("ERROR", f"{fname}: 缺失字段 {missing}")
            if extra:
                report.add("WARN", f"{fname}: 冗余字段 {extra}")

        # 详细列名列表
        report.add("INFO", f"{fname}: 列名 = {list(df.columns)}")


def validate_types_and_convert(dfs: dict, report: ValidationReport):
    """类型校验与自动转换。"""
    for fname, expected in EXPECTED_SCHEMA.items():
        if fname not in dfs:
            continue
        df = dfs[fname]
        expected_types = expected["types"]

        for col, exp_dtype in expected_types.items():
            if col not in df.columns:
                continue
            actual_dtype = str(df[col].dtype)

            # 自动类型转换
            try:
                if exp_dtype in ("int64", "int32"):
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                    report.add("PASS", f"{fname}.{col}: 类型转换 int → Int64 (nullable)")
                elif exp_dtype == "float64":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    report.add("PASS", f"{fname}.{col}: 类型转换 → float64")
                elif exp_dtype == "object" and "date" in col.lower():
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    report.add("PASS", f"{fname}.{col}: 类型转换 → datetime")
            except Exception as e:
                report.add("ERROR", f"{fname}.{col}: 类型转换失败 - {e}")


def validate_not_null(dfs: dict, report: ValidationReport):
    """非空约束校验。"""
    for fname, expected in EXPECTED_SCHEMA.items():
        if fname not in dfs:
            continue
        df = dfs[fname]
        for col in expected.get("not_null", []):
            if col not in df.columns:
                continue
            null_count = df[col].isna().sum()
            if null_count == 0:
                report.add("PASS", f"{fname}.{col}: 非空约束通过")
            else:
                report.add("WARN", f"{fname}.{col}: 存在 {null_count} 个空值 ({null_count/len(df)*100:.2f}%)")


def validate_foreign_keys(dfs: dict, report: ValidationReport):
    """外键引用完整性校验。"""
    for fname, expected in EXPECTED_SCHEMA.items():
        if fname not in dfs or "fk" not in expected:
            continue
        df = dfs[fname]
        for fk_col, ref in expected["fk"].items():
            ref_file, ref_col = ref.split(":")
            if ref_file not in dfs:
                report.add("WARN", f"{fname}.{fk_col}: 无法验证外键（{ref_file} 未加载）")
                continue

            ref_vals = set(dfs[ref_file][ref_col].dropna().unique())
            fk_vals = set(df[fk_col].dropna().unique())
            orphan = fk_vals - ref_vals

            if not orphan:
                report.add("PASS", f"{fname}.{fk_col} → {ref_file}.{ref_col}: 外键完整")
            else:
                report.add("ERROR", f"{fname}.{fk_col} → {ref_file}.{ref_col}: {len(orphan)} 个孤儿引用")


def validate_price_data(dfs: dict, report: ValidationReport):
    """价格业务逻辑校验：脏数据统计。"""
    if "price.csv" not in dfs:
        return
    df = dfs["price.csv"]
    total = len(df)
    stats = {}

    # 空价格
    null_p = df["price"].isna().sum()
    stats["空价格"] = null_p

    # 负价格
    neg_p = (df["price"] < 0).sum()
    stats["负价格"] = neg_p

    # 极端异常值 (>= 5000，对应生成脚本的 999999)
    extreme_p = (df["price"] >= 5000).sum()
    stats["极端异常值(≥5000)"] = extreme_p

    # 有效价格
    valid_mask = df["price"].notna() & (df["price"] > 0) & (df["price"] < 5000)
    stats["有效价格"] = valid_mask.sum()

    report.add("INFO", f"price.csv: 总行数 = {total:,}")
    for label, count in stats.items():
        pct = count / total * 100 if total > 0 else 0
        level = "PASS" if label == "有效价格" else "INFO"
        report.add(level, f"  {label}: {count:,} ({pct:.2f}%)")

    # 日期范围
    if "price_date" in df.columns:
        df["price_date_parsed"] = pd.to_datetime(df["price_date"], errors="coerce")
        valid_dates = df["price_date_parsed"].dropna()
        if len(valid_dates) > 0:
            report.add("INFO", f"  日期范围: {valid_dates.min().strftime('%Y-%m-%d')} ~ {valid_dates.max().strftime('%Y-%m-%d')}")
            report.add("INFO", f"  覆盖天数: {valid_dates.nunique()} 天")


def validate_product_count(dfs: dict, report: ValidationReport):
    """验证商品-分类分布合理性。"""
    if "product.csv" not in dfs or "category.csv" not in dfs:
        return
    prod_df = dfs["product.csv"]
    cat_df = dfs["category.csv"]

    cat_counts = prod_df.groupby("category_id").size()
    report.add("INFO", f"商品总数: {len(prod_df)}, 分类数: {len(cat_df)}")
    for cid, cnt in cat_counts.items():
        cat_name = cat_df.loc[cat_df["category_id"] == cid, "category_name"].values
        cat_name = cat_name[0] if len(cat_name) > 0 else f"未知分类{cid}"
        report.add("INFO", f"  分类 [{cid}] {cat_name}: {cnt} 个商品")


def main():
    """数据校验主流程。"""
    print("=" * 60)
    print("[校验] 电商数据格式校验与字段类型转换")
    print("=" * 60)

    report = ValidationReport(REPORT_PATH)
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. 文件存在性校验
    print("\n[1/6] 文件存在性校验")
    dfs = validate_file_exists(report)

    if len(dfs) < 3:
        report.add("ERROR", "必需文件不完整，终止后续校验")
        report.save()
        sys.exit(1)

    # 2. 字段完整性校验
    print("\n[2/6] 字段完整性校验")
    validate_schema(dfs, report)

    # 3. 类型校验与转换
    print("\n[3/6] 类型校验与自动转换")
    validate_types_and_convert(dfs, report)

    # 4. 非空约束校验
    print("\n[4/6] 非空约束校验")
    validate_not_null(dfs, report)

    # 5. 外键引用完整性
    print("\n[5/6] 外键引用完整性校验")
    validate_foreign_keys(dfs, report)

    # 6. 价格业务逻辑校验
    print("\n[6/6] 价格数据业务逻辑校验")
    validate_price_data(dfs, report)
    validate_product_count(dfs, report)

    # 保存报告
    report.save()

    # 退出码
    if report.errors > 0:
        print(f"\n❌ 校验未通过: {report.errors} 个错误")
        sys.exit(1)
    else:
        print(f"\n✅ 全部校验通过! (PASS={report.passes}, WARN={report.warnings})")


if __name__ == "__main__":
    main()
