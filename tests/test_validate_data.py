"""
validate_data.py 模块全面单元测试
----------------------------------
对 src/data_generation/validate_data.py 所有公开函数进行直接测试。
该模块为纯逻辑（无外部服务依赖），可达到高覆盖率。
"""
import os
import sys
import tempfile

import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_generation.validate_data import (
    ValidationReport,
    validate_file_exists,
    validate_schema,
    validate_types_and_convert,
    validate_not_null,
    validate_foreign_keys,
    validate_price_data,
    validate_product_count,
)


class TestValidationReport:
    """ValidationReport 类测试。"""

    def test_init(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        assert report.errors == 0
        assert report.warnings == 0
        assert report.passes == 0
        assert report.report_path == "/tmp/_vr_test.txt"

    def test_add_pass(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        report.add("PASS", "测试通过")
        assert report.passes == 1
        assert report.errors == 0
        assert report.warnings == 0
        assert len(report.lines) == 1
        assert "测试通过" in report.lines[0]
        assert "[PASS]" in report.lines[0]

    def test_add_warn(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        report.add("WARN", "测试警告")
        assert report.warnings == 1
        assert report.passes == 0
        assert report.errors == 0

    def test_add_error(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        report.add("ERROR", "测试错误")
        assert report.errors == 1
        assert report.passes == 0
        assert report.warnings == 0

    def test_add_info(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        report.add("INFO", "测试信息")
        assert report.errors == 0
        assert report.passes == 0
        assert report.warnings == 0

    def test_add_mixed(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        report.add("PASS", "a")
        report.add("PASS", "b")
        report.add("WARN", "c")
        report.add("ERROR", "d")
        report.add("ERROR", "e")
        report.add("INFO", "f")
        assert report.passes == 2
        assert report.warnings == 1
        assert report.errors == 2

    def test_save(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            tmp_path = f.name
        try:
            report = ValidationReport(tmp_path)
            report.add("PASS", "通过项目")
            report.add("ERROR", "错误项目")
            report.add("INFO", "信息项目")
            report.save()
            assert os.path.exists(tmp_path)
            with open(tmp_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "数据格式校验报告" in content
            assert "通过: 1" in content
            assert "错误: 1" in content
            assert "通过项目" in content
            assert "错误项目" in content
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestValidateFileExists:
    """文件存在性校验测试。"""

    def test_all_files_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = ValidationReport(os.path.join(tmpdir, "report.txt"))
            # 临时替换 DATA_DIR（需要 monkeypatch）
            import src.data_generation.validate_data as vd
            orig_dir = vd.DATA_DIR
            try:
                vd.DATA_DIR = tmpdir
                for fname in ["category.csv", "product.csv", "price.csv"]:
                    pd.DataFrame({"dummy": []}).to_csv(os.path.join(tmpdir, fname), index=False)
                dfs = validate_file_exists(report)
                assert len(dfs) == 3
                assert report.errors == 0
            finally:
                vd.DATA_DIR = orig_dir

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = ValidationReport(os.path.join(tmpdir, "report.txt"))
            import src.data_generation.validate_data as vd
            orig_dir = vd.DATA_DIR
            try:
                vd.DATA_DIR = tmpdir
                # 只创建 2 个文件，缺少 price.csv
                pd.DataFrame({"dummy": []}).to_csv(os.path.join(tmpdir, "category.csv"), index=False)
                pd.DataFrame({"dummy": []}).to_csv(os.path.join(tmpdir, "product.csv"), index=False)
                dfs = validate_file_exists(report)
                assert len(dfs) == 2
                assert report.errors >= 1
            finally:
                vd.DATA_DIR = orig_dir

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = ValidationReport(os.path.join(tmpdir, "report.txt"))
            import src.data_generation.validate_data as vd
            orig_dir = vd.DATA_DIR
            try:
                vd.DATA_DIR = tmpdir
                dfs = validate_file_exists(report)
                assert len(dfs) == 0
                assert report.errors == 3  # 3 个文件全部缺失
            finally:
                vd.DATA_DIR = orig_dir


class TestValidateSchema:
    """字段完整性校验测试。"""

    def test_perfect_match(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        dfs = {
            "category.csv": pd.DataFrame({"category_id": [1], "category_name": ["test"]}),
            "product.csv": pd.DataFrame({
                "product_id": [1], "category_id": [1], "product_name": ["p"],
                "brand": ["b"], "spec": ["s"],
            }),
            "price.csv": pd.DataFrame({
                "product_id": [1], "price_date": ["2026-01-01"],
                "price": [10.0], "sales_volume": [5],
            }),
        }
        validate_schema(dfs, report)
        assert report.errors == 0

    def test_missing_column(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        dfs = {
            "category.csv": pd.DataFrame({"category_id": [1]}),  # 缺少 category_name
        }
        validate_schema(dfs, report)
        assert report.errors >= 1

    def test_extra_column(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        dfs = {
            "category.csv": pd.DataFrame({
                "category_id": [1], "category_name": ["test"],
                "extra_col": ["unexpected"],  # 冗余字段
            }),
        }
        validate_schema(dfs, report)
        assert report.warnings >= 1


class TestValidateTypesAndConvert:
    """类型校验与转换测试。"""

    def test_int_conversion(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        dfs = {
            "category.csv": pd.DataFrame({
                "category_id": ["1", "2", "3"],  # 字符串 → Int64
                "category_name": ["a", "b", "c"],
            }),
        }
        validate_types_and_convert(dfs, report)
        assert pd.api.types.is_integer_dtype(dfs["category.csv"]["category_id"])

    def test_float_conversion(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        dfs = {
            "price.csv": pd.DataFrame({
                "product_id": [1, 2],
                "price_date": ["2026-01-01", "2026-01-02"],
                "price": ["10.5", "20.3"],  # 字符串 → float64
                "sales_volume": [5, 3],
            }),
        }
        validate_types_and_convert(dfs, report)
        assert pd.api.types.is_float_dtype(dfs["price.csv"]["price"])

    def test_date_conversion(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        dfs = {
            "price.csv": pd.DataFrame({
                "product_id": [1, 2],
                "price_date": ["2026-01-01", "2026-06-15"],
                "price": [10.0, 20.0], "sales_volume": [5, 3],
            }),
        }
        validate_types_and_convert(dfs, report)
        # price_date 应被转为 datetime
        assert pd.api.types.is_datetime64_any_dtype(dfs["price.csv"]["price_date"])

    def test_invalid_data_coerced_to_null(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        dfs = {
            "category.csv": pd.DataFrame({
                "category_id": ["abc", "def"],  # 无效 → NaN
                "category_name": ["a", "b"],
            }),
        }
        validate_types_and_convert(dfs, report)
        assert dfs["category.csv"]["category_id"].isna().all()


class TestValidateNotNull:
    """非空约束测试。"""

    def test_all_not_null(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        dfs = {
            "category.csv": pd.DataFrame({
                "category_id": [1, 2], "category_name": ["a", "b"],
            }),
        }
        validate_not_null(dfs, report)
        assert report.warnings == 0

    def test_has_null(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        dfs = {
            "category.csv": pd.DataFrame({
                "category_id": [1, None],
                "category_name": ["a", None],
            }),
        }
        validate_not_null(dfs, report)
        # category_id 和 category_name 都标记为 not_null，各有一个空值
        assert report.warnings >= 2


class TestValidateForeignKeys:
    """外键引用完整性测试。"""

    def test_fk_complete(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        dfs = {
            "category.csv": pd.DataFrame({
                "category_id": [1, 2], "category_name": ["a", "b"],
            }),
            "product.csv": pd.DataFrame({
                "product_id": [101, 102], "category_id": [1, 2],
                "product_name": ["p1", "p2"], "brand": ["b1", "b2"],
                "spec": ["s1", "s2"],
            }),
        }
        validate_foreign_keys(dfs, report)
        assert report.errors == 0

    def test_orphan_fk(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        dfs = {
            "category.csv": pd.DataFrame({
                "category_id": [1], "category_name": ["a"],
            }),
            "product.csv": pd.DataFrame({
                "product_id": [101, 102], "category_id": [1, 999],  # 999 是孤儿
                "product_name": ["p1", "p2"], "brand": ["b1", "b2"],
                "spec": ["s1", "s2"],
            }),
        }
        validate_foreign_keys(dfs, report)
        assert report.errors >= 1

    def test_fk_ref_not_loaded(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        dfs = {
            "product.csv": pd.DataFrame({
                "product_id": [101], "category_id": [1],
                "product_name": ["p1"], "brand": ["b1"], "spec": ["s1"],
            }),
        }
        # category.csv 未加载，应产生 WARN
        validate_foreign_keys(dfs, report)
        assert report.warnings >= 1


class TestValidatePriceData:
    """价格业务逻辑校验测试。"""

    def test_all_valid_prices(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        dfs = {
            "price.csv": pd.DataFrame({
                "product_id": [1, 2],
                "price_date": ["2026-01-01", "2026-01-02"],
                "price": [100.0, 200.0],
                "sales_volume": [5, 10],
            }),
        }
        validate_price_data(dfs, report)
        # 所有数据有效，不应有错误
        assert report.errors == 0

    def test_mixed_prices(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        dfs = {
            "price.csv": pd.DataFrame({
                "product_id": [1, 2, 3, 4, 5],
                "price_date": ["2026-01-01"] * 5,
                "price": [100.0, None, -10.0, 999999.0, 200.0],
                "sales_volume": [5, 10, 3, 8, 12],
            }),
        }
        validate_price_data(dfs, report)
        # 报告统计行在 INFO 行中
        report_text = "\n".join(report.lines)
        assert "空价格" in report_text
        assert "负价格" in report_text
        assert "极端异常值" in report_text

    def test_price_csv_not_loaded(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        dfs = {}  # price.csv 未加载
        validate_price_data(dfs, report)
        # 不应报错，直接返回
        assert report.errors == 0

    def test_date_range_detection(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        dfs = {
            "price.csv": pd.DataFrame({
                "product_id": [1, 1, 1],
                "price_date": ["2026-01-01", "2026-01-10", "2026-01-20"],
                "price": [10.0, 12.0, 11.0],
                "sales_volume": [5, 6, 5],
            }),
        }
        validate_price_data(dfs, report)
        report_text = "\n".join(report.lines)
        assert "覆盖天数" in report_text


class TestValidateProductCount:
    """商品-分类分布测试。"""

    def test_distribution(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        dfs = {
            "category.csv": pd.DataFrame({
                "category_id": [1, 2], "category_name": ["食品", "饮料"],
            }),
            "product.csv": pd.DataFrame({
                "product_id": [1, 2, 3, 4, 5],
                "category_id": [1, 1, 2, 2, 2],
                "product_name": [f"p{i}" for i in range(1, 6)],
                "brand": ["b"] * 5, "spec": ["s"] * 5,
            }),
        }
        validate_product_count(dfs, report)
        report_text = "\n".join(report.lines)
        assert "商品总数" in report_text
        assert "分类数" in report_text

    def test_missing_dfs(self):
        report = ValidationReport("/tmp/_vr_test.txt")
        dfs = {}
        validate_product_count(dfs, report)
        # 不报错，直接返回
        assert report.errors == 0


class TestMainFunction:
    """validate_data.main() 函数测试。"""

    def test_main_success_flow(self, monkeypatch):
        """完整 main() 流程：无错误应正常完成。"""
        import src.data_generation.validate_data as vd

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setattr(vd, "DATA_DIR", tmpdir)
            monkeypatch.setattr(vd, "REPORT_PATH", os.path.join(tmpdir, "validation_report.txt"))

            # 创建完整的三表数据
            pd.DataFrame({
                "category_id": [1, 2], "category_name": ["食品", "饮料"],
            }).to_csv(os.path.join(tmpdir, "category.csv"), index=False)
            pd.DataFrame({
                "product_id": [101, 102], "category_id": [1, 2],
                "product_name": ["苹果", "可乐"], "brand": ["A", "B"], "spec": ["1kg", "330ml"],
            }).to_csv(os.path.join(tmpdir, "product.csv"), index=False)
            pd.DataFrame({
                "product_id": [101, 102], "price_date": ["2026-01-01", "2026-01-02"],
                "price": [10.0, 5.0], "sales_volume": [100, 200],
            }).to_csv(os.path.join(tmpdir, "price.csv"), index=False)

            # 直接调用 main()
            vd.main()
            assert os.path.exists(vd.REPORT_PATH)

    def test_main_missing_files_exits(self, monkeypatch):
        """文件缺失时 main() 应 sys.exit(1)。"""
        import src.data_generation.validate_data as vd

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setattr(vd, "DATA_DIR", tmpdir)
            monkeypatch.setattr(vd, "REPORT_PATH", os.path.join(tmpdir, "validation_report.txt"))

            # 目录为空，无 CSV 文件
            with pytest.raises(SystemExit) as exc_info:
                vd.main()
            assert exc_info.value.code == 1

    def test_main_with_errors_exits(self, monkeypatch):
        """外键错误时 main() 应 sys.exit(1)。"""
        import src.data_generation.validate_data as vd

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setattr(vd, "DATA_DIR", tmpdir)
            monkeypatch.setattr(vd, "REPORT_PATH", os.path.join(tmpdir, "validation_report.txt"))

            pd.DataFrame({
                "category_id": [1], "category_name": ["食品"],
            }).to_csv(os.path.join(tmpdir, "category.csv"), index=False)
            pd.DataFrame({
                "product_id": [101], "category_id": [999],  # 孤儿外键！
                "product_name": ["苹果"], "brand": ["A"], "spec": ["1kg"],
            }).to_csv(os.path.join(tmpdir, "product.csv"), index=False)
            pd.DataFrame({
                "product_id": [101], "price_date": ["2026-01-01"],
                "price": [10.0], "sales_volume": [100],
            }).to_csv(os.path.join(tmpdir, "price.csv"), index=False)

            with pytest.raises(SystemExit) as exc_info:
                vd.main()
            assert exc_info.value.code == 1


class TestIntegrationEndToEnd:
    """端到端校验流程测试。"""

    def test_full_validation_flow(self):
        """模拟完整的 main() 流程（不含 sys.exit）。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            import src.data_generation.validate_data as vd
            orig_dir = vd.DATA_DIR
            orig_report = vd.REPORT_PATH
            try:
                vd.DATA_DIR = tmpdir
                vd.REPORT_PATH = os.path.join(tmpdir, "validation_report.txt")

                # 创建完整的三表数据
                pd.DataFrame({
                    "category_id": [1, 2], "category_name": ["食品", "饮料"],
                }).to_csv(os.path.join(tmpdir, "category.csv"), index=False)

                pd.DataFrame({
                    "product_id": [101, 102, 103],
                    "category_id": [1, 1, 2],
                    "product_name": ["苹果", "香蕉", "可乐"],
                    "brand": ["A牌", "B牌", "C牌"],
                    "spec": ["1kg", "500g", "330ml"],
                }).to_csv(os.path.join(tmpdir, "product.csv"), index=False)

                pd.DataFrame({
                    "product_id": [101, 102, 103],
                    "price_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                    "price": [9.9, 5.0, 3.5],
                    "sales_volume": [100, 200, 300],
                }).to_csv(os.path.join(tmpdir, "price.csv"), index=False)

                report = ValidationReport(vd.REPORT_PATH)

                # 执行全部 6 步校验
                dfs = validate_file_exists(report)
                assert len(dfs) == 3

                validate_schema(dfs, report)
                validate_types_and_convert(dfs, report)
                validate_not_null(dfs, report)
                validate_foreign_keys(dfs, report)
                validate_price_data(dfs, report)
                validate_product_count(dfs, report)

                report.save()

                # 验证报告中无 ERROR
                assert report.errors == 0, \
                    f"全流程应有 0 错误，实际 {report.errors} 个: {[l for l in report.lines if 'ERROR' in l]}"
                assert os.path.exists(vd.REPORT_PATH)

            finally:
                vd.DATA_DIR = orig_dir
                vd.REPORT_PATH = orig_report

    def test_validation_with_anomalies(self):
        """包含异常数据的完整校验流程。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            import src.data_generation.validate_data as vd
            orig_dir = vd.DATA_DIR
            orig_report = vd.REPORT_PATH
            try:
                vd.DATA_DIR = tmpdir
                vd.REPORT_PATH = os.path.join(tmpdir, "validation_report.txt")

                # 包含问题的数据：孤儿外键、脏价格、缺失字段
                pd.DataFrame({
                    "category_id": [1], "category_name": ["食品"],
                }).to_csv(os.path.join(tmpdir, "category.csv"), index=False)

                pd.DataFrame({
                    "product_id": [101, 102],
                    "category_id": [1, 999],  # 999 是孤儿引用
                    "product_name": ["苹果", "未知"],
                    "brand": ["A牌", "X牌"],
                    "spec": ["1kg", "?"],
                }).to_csv(os.path.join(tmpdir, "product.csv"), index=False)

                pd.DataFrame({
                    "product_id": [101, 102, 103],  # 103 在 product 表中不存在
                    "price_date": ["2026-01-01"] * 3,
                    "price": [9.9, None, 100.0],
                    "sales_volume": [100, 0, 50],
                }).to_csv(os.path.join(tmpdir, "price.csv"), index=False)

                report = ValidationReport(vd.REPORT_PATH)

                dfs = validate_file_exists(report)
                validate_schema(dfs, report)
                validate_types_and_convert(dfs, report)
                validate_not_null(dfs, report)
                validate_foreign_keys(dfs, report)
                validate_price_data(dfs, report)
                validate_product_count(dfs, report)

                report.save()

                # 应有孤儿外键 ERROR
                assert report.errors >= 1, f"应有至少 1 个错误（孤儿外键），实际 {report.errors}"
                # 应有空值 WARN
                assert report.warnings >= 1, f"应有至少 1 个警告（空值），实际 {report.warnings}"

            finally:
                vd.DATA_DIR = orig_dir
                vd.REPORT_PATH = orig_report
