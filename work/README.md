# Iris Machine Learning Experiment (鸢尾花机器学习实验)

本项目基于鸢尾花（Iris）数据集，包含数据预处理、可视化、KNN与决策树分类方法的手写实现、模型评估与对比等模块，代码和结果均可复现。

---

## 主要功能模块
- 数据预处理：数据读取、特征标准化、标签编码、缺失值处理、训练测试集划分
- 可视化分析：单变量/多变量分布、相关性热图、分组箱线图、pairplot等
- KNN分类：纯Python实现KNN，支持参数调优、评估与可视化
- 决策树分类：纯Python实现决策树（基尼指数），支持深度调优、评估与可视化
- 模型对比分析：自动汇总KNN与决策树各项指标，生成对比表格和可视化
- 实验报告：各阶段自动生成Markdown报告，便于查阅和引用

---

## 目录结构
```
work/
├── 1_data_preprocess/         # 数据预处理相关代码
│   └── code/
│       ├── 1.1_data_overview.py
│       ├── 1.2_data_preprocess.py
│       └── 1.3_data_split.py
├── 2_visualization/           # 可视化分析相关代码
│   ├── 2.1_feature_distribution/
│   │   ├── feature_distribution.py
│   │   └── result/            # 特征分布可视化结果（图片、csv等）
│   └── 2.2_class_distribution/
│       ├── class_distribution.py
│       └── result/            # 类别分布可视化结果（图片、csv等）
├── 3_KNN/                     # KNN分类实验
│   ├── code/
│   │   ├── knn_core.py
│   │   ├── knn_main.py
│   │   └── knn_思路说明.md
│   └── result/                # KNN实验结果（图片、csv、分析报告等）
├── 4_TREE/                    # 决策树分类实验
│   ├── code/
│   │   ├── tree_core.py
│   │   ├── tree_main.py
│   │   └── tree_思路说明.md
│   └── result/                # 决策树实验结果（图片、csv、分析报告等）
├── 5_COMPARE/                 # 模型对比分析
│   ├── code/
│   │   └── compare_models.py
│   └── result/                # 对比分析结果（图片、csv、分析报告等）
├── data/                      # 数据文件
│   └── iris.csv
├── LICENSE                    # 开源许可证
├── README.md                  # 项目说明文档
└── 思路.txt                   # 项目开发思路与笔记
```

---

## 运行环境与依赖
- Python 3.7+
- 依赖：pandas, numpy, matplotlib, seaborn
- 安装依赖：
```bash
pip install pandas numpy matplotlib seaborn
```

---

## 快速开始
1. 克隆仓库并进入目录：
   ```bash
   git clone https://github.com/yuhong2024/Learn_AI.git
   cd Learn_AI/work
   ```
2. 运行各阶段主控脚本：
   ```bash
   python 3_KNN/code/knn_main.py
   python 4_TREE/code/tree_main.py
   python 5_COMPARE/code/compare_models.py
   ```
3. 所有结果自动保存到对应result文件夹。

---

## 结果说明
- 所有评估指标、混淆矩阵、对比表格、可视化图片均自动输出
- Markdown实验报告自动生成，便于查阅和引用

---

## 贡献
如有建议或改进，欢迎通过Issue或PR反馈。

---

## 作者
- Y.Hong
- 邮箱：wyhstar@email.swu.edu.cn
- GitHub: https://github.com/yuhong2024

---

## License
本项目采用 GPL-2.0 License，详见 LICENSE 文件。 