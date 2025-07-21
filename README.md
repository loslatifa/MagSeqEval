# MagLocEval

## 地磁序列定位模型评价框架

MagLocEval 是一个专为 **地磁序列输入的室内定位模型**设计的系统化评价与错误分析框架。

### 功能特性
- 计算 **MAE、RMSE、百分位误差、CDF** 等准确率指标
- 自动筛选 **高误差路径及路径内关键高误差点**
- 在其他路径中寻找 **预测准确的同位置样本**进行对比分析
- 可视化 **CDF 曲线及楼层平面误差热力图**
- 预留 **特征空间-物理空间对齐度**、**区域准确率**接口，便于科研扩展

### 文件结构
- `run_magloceval.py` – 主执行程序，加载数据、执行完整评价及可视化
- `magloceval_utils.py` – 核心指标计算与可视化函数模块

### 安装依赖
```bash
pip install numpy pandas matplotlib scikit-learn
```

### 使用方法
1️⃣ 将预测结果 CSV 放入工作目录，示例格式：
```csv
path_id,timestamp,mag_x,mag_y,mag_z,pred_x,pred_y,true_x,true_y
```

2️⃣ 执行评价：
```bash
python run_magloceval.py --input your_test_results.csv
```

### 输出结果
- 终端输出 MAE、RMSE、百分位误差、CDF 等指标
- 自动列出 **平均误差最大的路径及关键高误差点**
- 可视化生成：
    - CDF 曲线图
    - 平面误差热力图

---
如需持续扩展或科研论文结果可视化，请联系助手继续协助完成。