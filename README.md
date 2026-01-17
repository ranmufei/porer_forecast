# 风电站功率预测系统

基于GFS气象数据的风电站发电功率预测系统，使用XGBoost机器学习模型进行训练和预测。

## 系统概述

本系统通过分析历史气象数据（温度、风速、风向）和历史发电功率，训练机器学习模型，实现基于GFS气象预测数据的电站发电功率预测。

## 数据说明

### 输入数据

1. **GFS气象数据** (`data_gfs_forecast.csv`)
   - 包含字段：
     - `timestamp`: 时间戳（15分钟粒度）
     - `gfs_temp`: 温度 (°C)
     - `gfs_wind_speed`: 风速 (m/s)
     - `gfs_wind_direction`: 风向 (度)

2. **历史功率数据** (`data_history_power.csv`)
   - 包含字段：
     - `timestamp`: 时间戳（15分钟粒度）
     - `power`: 实际发电功率 (kW)

### 输出数据

**预测结果** (`predictions.csv`)
- 包含字段：
  - `timestamp`: 时间戳
  - `predicted_power`: 预测发电功率 (kW)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

使用历史数据训练预测模型：

```bash
python train_power_forecast.py
```

训练过程包括：
- 加载和合并数据
- 特征工程（时间特征、风向编码、风速立方等）
- 模型训练（XGBoost回归）
- 模型评估（MAE、RMSE、MAPE、R²）
- 生成可视化图表（预测对比图、特征重要性图）
- 保存训练好的模型 (`power_forecast_model.pkl`)

### 2. 功率预测

使用训练好的模型进行预测：

```bash
python predict_power.py
```

预测过程包括：
- 加载训练好的模型
- 读取GFS气象数据
- 准备特征
- 进行预测
- 保存预测结果到CSV文件

### 3. 自定义预测

在代码中使用模型进行预测：

```python
from predict_power import load_model, predict_from_dict

# 加载模型
model = load_model('power_forecast_model.pkl')

# 准备GFS数据
gfs_data = {
    'timestamp': '2024-08-01 12:00:00',
    'gfs_temp': 15.5,
    'gfs_wind_speed': 8.3,
    'gfs_wind_direction': 180
}

# 预测
predicted_power = predict_from_dict(model, gfs_data)
print(f"预测功率: {predicted_power:.2f} kW")
```

## 模型架构

### 特征工程

系统使用以下特征：

1. **基础气象特征**
   - 温度 (`gfs_temp`)
   - 风速 (`gfs_wind_speed`)
   - 风向 (`gfs_wind_direction`)

2. **时间特征**
   - 小时 (`hour`)
   - 日期 (`day`)
   - 月份 (`month`)
   - 星期几 (`day_of_week`)
   - 一年中的第几天 (`day_of_year`)

3. **周期性特征编码**
   - 小时的正弦/余弦编码 (`hour_sin`, `hour_cos`)
   - 风向的正弦/余弦编码 (`wind_dir_sin`, `wind_dir_cos`)

4. **物理特征**
   - 风速的三次方 (`wind_speed_cube`) - 基于风功率公式 P ∝ v³

### 模型参数

- **算法**: XGBoost Regressor
- **最大深度**: 6
- **学习率**: 0.1
- **树的数量**: 1000
- **早停轮数**: 50

## 评估指标

模型使用以下指标进行评估：

- **MAE** (平均绝对误差): 预测值与真实值的平均绝对差
- **RMSE** (均方根误差): 预测值与真实值的均方根差
- **MAPE** (平均绝对百分比误差): 预测误差的百分比
- **R²** (决定系数): 模型解释的方差比例

## 输出文件

训练后会生成以下文件：

1. `power_forecast_model.pkl` - 训练好的模型文件
2. `prediction_results.png` - 预测结果对比图
3. `feature_importance.png` - 特征重要性图
4. `predictions.csv` - 预测结果（运行预测脚本后）

## 系统优势

1. **准确性**: 使用XGBoost算法，在表格数据上表现优异
2. **可解释性**: 提供特征重要性分析，理解关键影响因素
3. **高效性**: 训练速度快，预测延迟低
4. **易用性**: 简单的API接口，易于集成到生产环境
5. **可视化**: 自动生成图表，便于理解模型性能

## 注意事项

1. 确保输入数据的格式正确，时间戳需要与训练数据格式一致
2. GFS气象数据的时间粒度应为15分钟
3. 模型会保存在当前目录下，请确保有写入权限
4. 首次使用需要先训练模型，之后可以直接加载模型进行预测

## 技术栈

- Python 3.8+
- Pandas - 数据处理
- NumPy - 数值计算
- XGBoost - 机器学习模型
- Scikit-learn - 模型评估
- Matplotlib/Seaborn - 数据可视化

## 许可证

本项目仅供学习和研究使用。
