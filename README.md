# Gold Price Prediction | 黄金价格预测

> 基于传统机器学习与深度学习（LSTM / GRU）的黄金价格时间序列预测项目

---

## 项目简介

本项目通过分析黄金价格与美元指数、原油价格、标普 500 指数等宏观经济指标之间的关系，
构建多种预测模型（线性回归、多项式回归、Ridge 回归、LSTM、GRU），并对各模型的预测效果进行对比分析。

**数据来源：** Yahoo Finance（`yfinance`）  
**时间范围：** 2015-01-01 ~ 2026-01-01  
**预测目标：** 黄金收盘价（归一化后）

---

## 快速开始

### 1. 创建并激活虚拟环境（推荐）

虚拟环境可以保证每个项目的依赖互不干扰，强烈建议使用。

```bash
# 在项目根目录下创建虚拟环境
python -m venv .venv

# 激活虚拟环境（Windows）
.venv\Scripts\activate
```

激活成功后，终端最左边会出现 `(.venv)` 前缀。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> 如果下载慢，使用国内镜像：
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

### 3. 注册 Jupyter 内核（只需做一次）

让 Cursor / Jupyter 能识别到这个虚拟环境：

```bash
python -m ipykernel install --user --name=gold_venv --display-name "Gold Price Project"
```

### 4. 在 Cursor 里打开 Notebook 并选择内核

打开 `notebooks/gold_price_prediction_full.ipynb`，点击右上角的内核选择按钮，
选择 **Gold Price Project**，然后点击 **Run All** 运行全部单元格即可。

### 5. 分步运行（可选）

如果只想运行某个模块，先激活虚拟环境，再执行 `src/` 下的脚本：

```bash
.venv\Scripts\activate
cd src

# 步骤1：采集真实数据（需要联网）
python data_collection.py

# 步骤1（备用）：生成模拟数据（无需联网）
python generate_sample_data.py

# 步骤2：数据预处理
python data_preprocessing.py

# 步骤3：训练基线模型
python train_baseline.py
```

---

## 项目结构

```
Gold_Price_Prediction/
│
├── notebooks/
│   ├── gold_price_prediction_full.ipynb  ← 主文件，完整项目流程（推荐从这里开始）
│   └── 01_eda.ipynb                      ← 单独的 EDA 探索笔记本
│
├── src/                                  ← 独立运行的 Python 脚本
│   ├── data_collection.py                ← 从 Yahoo Finance 下载数据
│   ├── generate_sample_data.py           ← 网络受限时生成模拟数据
│   ├── data_preprocessing.py             ← 数据清洗、归一化、划分
│   └── train_baseline.py                 ← 训练三个基线模型
│
├── models/
│   └── deep_learning/
│       └── lstm_model.py                 ← LSTM 模型定义（占位，完整实现在 Notebook 中）
│
├── streamlit_app/
│   └── app.py                            ← Web 前端（待实现）
│
├── data/
│   ├── raw/
│   │   └── gold_data_raw.csv             ← 原始数据（运行后生成）
│   └── processed/
│       ├── train_data.csv                ← 训练集（运行后生成）
│       ├── val_data.csv                  ← 验证集（运行后生成）
│       ├── test_data.csv                 ← 测试集（运行后生成）
│       └── scaler.pkl                    ← MinMax 归一化器（运行后生成）
│
├── outputs/
│   ├── figures/                          ← 所有生成的图表
│   └── models/                           ← 保存的模型文件
│
├── utils/                                ← 通用工具函数（待扩展）
│
├── requirements.txt                      ← Python 依赖清单
└── README.md                             ← 本文件
```

---

## 主 Notebook 内容（gold_price_prediction_full.ipynb）

| 章节 | 内容 | 输出 |
|------|------|------|
| **第一部分** | 环境配置、库导入 | — |
| **第二部分** | 数据采集（Yahoo Finance / 模拟数据） | `data/raw/gold_data_raw.csv` |
| **第三部分** | 数据预处理（缺失值处理、归一化、划分） | `data/processed/*.csv` |
| **第四部分** | 探索性数据分析（趋势图、热力图、分布图） | `outputs/figures/eda_*.png` |
| **第五部分** | 基线模型（线性/多项式/Ridge 回归） | `outputs/models/*.pkl` |
| **第六部分** | 深度学习模型（LSTM、GRU） | `outputs/models/*.pth` |
| **第七部分** | 模型对比与综合分析 | `outputs/figures/final_comparison.png` |

---

## 技术栈

| 类别 | 工具 |
|------|------|
| 语言 | Python 3.8+ |
| 数据处理 | `pandas`, `numpy` |
| 数据获取 | `yfinance` |
| 传统机器学习 | `scikit-learn` |
| 深度学习 | `PyTorch` |
| 可视化 | `matplotlib`, `seaborn` |
| Web 前端（待实现） | `streamlit` |
| 开发环境 | Jupyter Notebook |

---

## 模型说明

### 基线模型

| 模型 | 说明 |
|------|------|
| 线性回归 | 假设目标与特征间存在线性关系，作为最简单的基准 |
| 多项式回归 | 加入特征的高次项，可捕捉非线性关系（degree=2）|
| Ridge 回归 | 线性回归 + L2 正则化，防止过拟合 |

### 深度学习模型

| 模型 | 说明 |
|------|------|
| LSTM | 长短期记忆网络，通过三个门控机制学习时间序列的长期依赖 |
| GRU | LSTM 的简化版本，参数更少，训练更快，效果相近 |

**时间窗口设计：** 用过去 **30 天**的所有特征预测**第 31 天**的黄金价格。

---

## 评估指标

| 指标 | 全称 | 说明 |
|------|------|------|
| MAE | Mean Absolute Error | 平均绝对误差，越小越好 |
| RMSE | Root Mean Square Error | 均方根误差，对大误差更敏感，越小越好 |
| MAPE | Mean Absolute Percentage Error | 误差占真实值的百分比，越小越好 |
| R² | Coefficient of Determination | 决定系数，越接近 1 越好 |

---

## 常见问题

**Q1：运行 data_collection.py 报错 "Too Many Requests"**  
Yahoo Finance 对请求频率有限制。
解决方法：改用 `generate_sample_data.py` 生成模拟数据，或稍后再试。

**Q2：图表中文显示为方框**  
Windows 系统通常自带 SimHei 字体，如仍有问题，将代码中的字体改为 `'Microsoft YaHei'`。

**Q3：PyTorch 未安装**  
先激活虚拟环境，再安装 PyTorch：
```bash
.venv\Scripts\activate
pip install torch torchvision torchaudio
```
CPU 版（无 GPU 也能运行，只是较慢）：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Q4：训练太慢**  
- 基线模型通常在 1 分钟内完成
- 深度学习模型（LSTM/GRU）在 CPU 上约需 5~15 分钟
- 如有 NVIDIA GPU，程序会自动使用 CUDA 加速

---

## 当前进度

- [x] 项目结构搭建
- [x] 数据采集（Yahoo Finance + 模拟数据备用方案）
- [x] 数据预处理（缺失值、归一化、数据集划分）
- [x] 探索性数据分析（EDA）
- [x] 基线模型（线性回归、多项式回归、Ridge 回归）
- [x] 深度学习模型（LSTM、GRU）
- [x] 模型对比与可视化
- [ ] Streamlit Web 前端（待实现）

---

## 团队成员与分工

| 角色 | 负责内容 |
|------|---------|
| 成员 A | 数据采集、数据预处理、基线模型 |
| 成员 B | 深度学习模型（LSTM/GRU）、Streamlit 集成 |
| 成员 C | 可视化图表、PPT 制作 |
| 成员 D | 演示视频、演讲组织 |
| 成员 E | 项目报告（3500 字） |

---

## 参考资料

- [yfinance 文档](https://pypi.org/project/yfinance/)
- [PyTorch LSTM 文档](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [scikit-learn 文档](https://scikit-learn.org/stable/)
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*.
- Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder. *EMNLP*.
