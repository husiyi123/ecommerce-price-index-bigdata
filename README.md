# 电商价格指数大数据分析系统

> **课程设计大作业** —— 基于阿里云 OSS + ClickHouse 的电商 CPI 价格指数计算与可视化平台

## 📁 项目结构

```
ecommerce-price-index-bigdata/
├── .github/                    # CI/CD 自动化流水线配置
│   └── workflows/
│       └── ci_cd_pipeline.yml  # GitHub Actions 全流程工作流
├── config/                     # 配置文件目录（密钥严禁提交）
│   ├── .env.example            # 环境变量示例模板
│   └── config.yaml.template    # 配置模板
├── src/                        # 核心源代码目录
│   ├── data_generation/        # 模拟数据生成模块
│   │   └── data_generator.py   # 100MB 电商三表数据集生成脚本
│   ├── oss_operation/          # OSS 对象存储操作模块
│   │   └── oss_upload.py       # 本地数据上传至阿里云 OSS
│   ├── clickhouse_process/     # ClickHouse 清洗与计算模块
│   │   └── ck_price_calc.py    # 建表、脏数据过滤、价格指数核心计算
│   └── visualization/          # 数据可视化模块
│       └── draw_cpi_trend.py   # 日度 CPI 价格指数趋势图绘制
├── tests/                      # 单元测试 & 集成测试
│   ├── test_data/              # 手工构造的小规模测试 CSV
│   ├── test_oss_connect.py     # OSS 连通性与字段解析单元测试
│   └── test_ck_sql.py          # ClickHouse SQL 逻辑正确性验证
├── data/                       # 数据目录（大文件默认不提交）
│   ├── raw/                    # 生成的原始 CSV 数据（约 100MB）
│   └── result/                 # 指数计算结果、导出 CSV、可视化图片
├── docs/                       # 课程设计交付文档
│   ├── 设计文档.md             # 需求分析、技术选型、表设计、架构图
│   ├── 实践总结报告.md         # 完整实验报告（权重 40%）
│   ├── 实践工作日志.md         # 分阶段开发记录（权重 20%）
│   └── assets/                 # 报告配图：运行截图、架构图、结果图
├── requirements.txt            # Python 全量依赖清单
├── .gitignore                  # Git 忽略规则
└── README.md                   # 本文件
```

## 🚀 环境搭建

### 1. 前置条件

- **Python** ≥ 3.10
- **Git** ≥ 2.30
- **阿里云 OSS 账号**（已开通 OSS 服务）
- **ClickHouse** 服务（本地开发可用 Docker 快速启动）

### 2. 安装依赖

```bash
# 克隆仓库
git clone <your-repo-url>
cd ecommerce-price-index-bigdata

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置密钥

```bash
# 复制环境变量模板
cp config/.env.example .env

# 编辑 .env，填入真实的阿里云 OSS 和 ClickHouse 凭据
vim .env
```

### 4. 启动 ClickHouse（Docker 方式）

```bash
docker run -d --name clickhouse-server \
  -p 8123:8123 -p 9000:9000 \
  -e CLICKHOUSE_DB=ecommerce \
  clickhouse/clickhouse-server:latest
```

## 🔧 运行步骤

### Step 1: 生成模拟数据

```bash
python src/data_generation/data_generator.py
# 输出: data/raw/ 目录下生成 ~100MB 的 CSV 文件
```

### Step 2: 上传至阿里云 OSS

```bash
python src/oss_operation/oss_upload.py
# 将 data/raw/ 中的数据上传至配置的 OSS Bucket
```

### Step 3: ClickHouse 清洗与计算

```bash
python src/clickhouse_process/ck_price_calc.py
# 1. 从 OSS 导入数据到 ClickHouse
# 2. 创建电商三表（商品信息表、价格明细表、分类维度表）
# 3. 执行脏数据过滤
# 4. 计算日度 CPI 价格指数
# 5. 导出结果至 data/result/
```

### Step 4: 生成可视化趋势图

```bash
python src/visualization/draw_cpi_trend.py
# 输出: data/result/ 目录下的 CPI 趋势图 PNG/SVG
```

### 一键运行

```bash
# 顺序执行全流程
python src/data_generation/data_generator.py && \
python src/oss_operation/oss_upload.py && \
python src/clickhouse_process/ck_price_calc.py && \
python src/visualization/draw_cpi_trend.py
```

## 🧪 运行测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行指定测试文件
pytest tests/test_oss_connect.py -v
pytest tests/test_ck_sql.py -v

# 生成覆盖率报告
pytest tests/ --cov=src --cov-report=html
```

## 📊 CI/CD 流水线

本项目配置了 GitHub Actions 四阶段流水线：

| 阶段 | 说明 | 触发条件 |
|------|------|----------|
| **阶段 1** | 代码检查 + 单元测试 | push / PR |
| **阶段 2** | 数据生成 + OSS 上传 | 阶段 1 通过后 |
| **阶段 3** | ClickHouse 清洗 + CPI 计算 | 阶段 2 通过后 |
| **阶段 4** | 可视化 + 产物归档 | 阶段 3 通过后 |

流水线产物保留 7-30 天，可在 Actions → Artifacts 中下载。

## 📐 技术架构

```
┌─────────────────────────────────────────────────────┐
│                    数据源层                            │
│  data_generator.py → 100MB 模拟 CSV (商品/价格/分类)   │
└────────────────────┬────────────────────────────────┘
                     │ oss_upload.py
                     ▼
┌─────────────────────────────────────────────────────┐
│                 阿里云 OSS 存储层                       │
│  原始 CSV 文件存储 → 触发 ClickHouse 导入              │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│               ClickHouse 计算引擎层                    │
│  ck_price_calc.py                                    │
│  ├── 建表: product_info / price_detail / category_dim │
│  ├── 清洗: 范围过滤 / 空值处理 / 去重                 │
│  └── 计算: Laspeyres / Paasche / Fisher 指数          │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│                   可视化展示层                         │
│  draw_cpi_trend.py → matplotlib/plotly 趋势图        │
└─────────────────────────────────────────────────────┘
```

## 📝 评分项对应

| 评分项 | 权重 | 对应交付物 |
|--------|------|-----------|
| 实践总结报告 | 40% | `docs/实践总结报告.md` |
| 实践工作日志 | 20% | `docs/实践工作日志.md` |
| 设计文档 | 15% | `docs/设计文档.md` |
| 代码与实现 | 25% | `src/` 全部源代码 |

## ⚠️ 注意事项

- **密钥安全**: `.env` 和 `config/config.yaml` 已被 `.gitignore` 忽略，严禁提交到 Git
- **数据量**: 生成的 CSV 约 100MB，确保磁盘空间充足（建议 ≥ 500MB 余量）
- **网络**: OSS 上传速度取决于带宽，建议在云服务器上运行

## 📄 License

本项目仅用于课程设计学习目的。
