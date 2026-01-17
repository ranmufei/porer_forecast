# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个基于GFS气象数据的风电站发电功率预测系统，使用机器学习模型（XGBoost和集成学习）进行15分钟粒度的功率预测。项目包含基础版本和优化版本两套完整方案。

## 常用命令

### 依赖安装
```bash
pip install -r requirements.txt
```

### 模型训练
```bash
# 基础模型训练（XGBoost单模型）
python train_power_forecast.py

# 优化模型训练（集成学习：XGBoost + Random Forest + Gradient Boosting）
python train_power_forecast_optimized.py
```

### 功率预测
```bash
# 使用基础模型预测
python predict_power.py

# 使用优化模型预测
python predict_power_optimized.py
```

### 启动API服务
```bash
python api.py
# 服务地址: http://localhost:5000
# API文档: http://localhost:5000/info
```

### 测试和评估
```bash
# 优化效果测试和对比
python test_optimization.py

# 功率分段分析
python analyze_power_segments.py

# 综合对比分析
python analyze_final_comparison.py
```

## 系统架构

### 数据流程

1. **输入数据**
   - `data_history_power.csv`: 历史实际发电功率（15分钟粒度）
   - `data_gfs_forecast.csv`: GFS气象预报数据（温度、风速、风向）

2. **数据处理** ([`train_power_forecast_optimized.py:60-115`](train_power_forecast_optimized.py#L60-L115))
   - 数据合并：按时间戳内连接气象数据和功率数据
   - 数据清理：移除负值功率，填充缺失值
   - 特征工程：生成时间特征、周期性编码、物理特征、约束特征

3. **模型架构**
   - **基础版本** ([`train_power_forecast.py:127-161`](train_power_forecast.py#L127-L161)): 单一XGBoost模型
   - **优化版本** ([`train_power_forecast_optimized.py:151-208`](train_power_forecast_optimized.py#L151-L208)): 集成学习
     - XGBoost（权重50%）- 主力模型
     - Random Forest（权重30%）- 稳定性
     - Gradient Boosting（权重20%）- 补充

4. **后处理优化** ([`train_power_forecast_optimized.py:236-249`](train_power_forecast_optimized.py#L236-L249))
   - 负值截断：确保预测功率 >= 0
   - 移动平均平滑：3点窗口减少波动
   - 极值修正：限制异常预测

5. **输出**
   - 模型文件（.pkl）
   - 预测结果（.csv）
   - 可视化图表（.png）

### 核心特征工程

系统使用以下特征类型：

1. **基础气象特征**: gfs_temp, gfs_wind_speed, gfs_wind_direction
2. **时间特征**: hour, day, month, day_of_week, day_of_year
3. **周期性编码**: hour_sin/cos, wind_dir_sin/cos - 处理周期性问题
4. **物理特征**: wind_speed_cube - 基于风功率公式 P ∝ v³
5. **约束特征** (优化版):
   - wind_below_cutin: 风速 < 3m/s（切入风速）
   - wind_above_cutoff: 风速 > 25m/s（切出风速）
   - wind_in_ideal_range: 10-15m/s（理想范围）
6. **交互特征** (优化版):
   - estimated_max_power: 基于风速估算理论功率上限
   - wind_temp_interaction: 风速×温度交互

### API接口

[`api.py`](api.py) 提供RESTful API：

- `POST /predict`: 单点预测
- `POST /batch_predict`: 批量预测
- `GET /health`: 健康检查
- `GET /info`: API信息

## 重要技术细节

### 数据划分策略
- 按时间顺序划分（非随机划分）以避免未来数据泄露
- 80%训练，20%测试
- 训练集中再划分80%训练+20%验证用于早停

### 特征工程关键点
- **周期性编码**：使用sin/cos编码处理小时（0-23）和风向（0-360°）的周期性，避免0和23/360不连续的问题
- **风速立方**：基于物理原理，风功率与风速三次方成正比
- **约束特征**：模拟风电机组的实际运行区间（切入/额定/切出风速）

### 模型参数
- XGBoost: max_depth=6, learning_rate=0.1, n_estimators=500-1000
- Random Forest: n_estimators=100, max_depth=10
- Gradient Boosting: n_estimators=200, max_depth=5

### 评估指标
- **MAE** (平均绝对误差): 主要评估指标
- **RMSE** (均方根误差): 惩罚大误差
- **MAPE** (平均绝对百分比误差): 相对误差
- **R²** (决定系数): 拟合优度

### 优化版改进
根据 [`优化报告.md`](优化报告.md)，优化版相比基础版：
- MAE降低21.32%（7178→5648 kW）
- 完全消除负值预测
- 零功率预测改善（0-5k区间MAE降低64.29%）
- R²从0.4142提升至0.6099

## 数据文件说明

- **data_history_power.csv**: 历史功率数据，格式为 `timestamp, power`（需跳过第一行空行）
- **data_gfs_forecast.csv**: GFS气象数据，格式为 `timestamp, gfs_temp, gfs_wind_speed, gfs_wind_direction`
- **对比数据/**: 包含用于模型评估的独立测试数据（6月1日-8日）

## 模型文件

训练后会生成以下模型文件：
- `power_forecast_model.pkl`: 基础XGBoost模型
- `power_forecast_model_optimized_*.pkl`: 优化版集成模型（xgboost, random_forest, gradient_boosting）
- `power_forecast_model_optimized_info.pkl`: 模型元信息

## 注意事项

1. **数据时序性**: 必须保持时间顺序，不能随机shuffle，否则会导致数据泄露
2. **滞后特征限制**: 滚动统计和滞后特征仅用于分析，实际预测时不可使用
3. **负值处理**: 预测值必须截断为 >= 0，因为功率不能为负
4. **特征一致性**: 预测时的特征工程必须与训练时完全一致
5. **数据量**: 优化版本使用完整一年数据（35041条），相比部分数据（5824条）性能显著提升

## 性能基准

根据 [`项目总结报告.md`](项目总结报告.md)，优化模型在对比数据上的表现：
- **整体**: MAE=6,518.7 kW, R²=0.52
- **低功率 (<5k)**: MAE=5,739.4 kW
- **中高功率 (15-30k)**: MAE=5,445.5 kW（最佳区间）
- **高功率 (>=30k)**: MAE=10,034.3 kW（挑战最大）

## 中文支持

项目默认使用中文显示图表，已配置字体：
```python
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
```
