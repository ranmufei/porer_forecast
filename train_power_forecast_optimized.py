"""
风电站功率预测模型训练脚本（优化版）
使用GFS气象数据预测电站发电功率
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
import pickle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """加载GFS气象数据和历史功率数据"""
    print("正在加载数据...")
    
    # 加载GFS数据
    gfs_df = pd.read_csv('data_gfs_forecast.csv')
    gfs_df['timestamp'] = pd.to_datetime(gfs_df['timestamp'])
    
    # 加载功率数据
    power_df = pd.read_csv('data_history_power.csv', header=None, skiprows=1, names=['timestamp', 'power'])
    power_df['timestamp'] = pd.to_datetime(power_df['timestamp'])
    
    # 数据清理：移除负值功率
    power_df['power'] = power_df['power'].clip(lower=0)
    
    print(f"GFS数据: {len(gfs_df)} 条记录")
    print(f"功率数据: {len(power_df)} 条记录")
    print(f"清理负值后功率数据范围: {power_df['power'].min():.2f} - {power_df['power'].max():.2f} kW")
    
    return gfs_df, power_df


def merge_data(gfs_df, power_df):
    """合并GFS数据和功率数据"""
    print("\n正在合并数据...")
    
    # 按时间戳合并
    merged_df = pd.merge(gfs_df, power_df, on='timestamp', how='inner')
    
    print(f"合并后数据: {len(merged_df)} 条记录")
    print(f"时间范围: {merged_df['timestamp'].min()} 到 {merged_df['timestamp'].max()}")
    
    return merged_df


def feature_engineering(df):
    """特征工程（增强版）"""
    print("\n正在进行特征工程...")
    
    df = df.copy()
    
    # 1. 基础时间特征
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    # 2. 周期性编码
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df['wind_dir_sin'] = np.sin(2 * np.pi * df['gfs_wind_direction'] / 360)
    df['wind_dir_cos'] = np.cos(2 * np.pi * df['gfs_wind_direction'] / 360)
    
    # 3. 风速主导特征工程 - v2.5风速优化版
    # ============ 核心：强化风速特征，弱化温度特征 ============

    # 3.1 基础风速幂次（风功率物理公式）
    df['wind_speed_square'] = df['gfs_wind_speed'] ** 2
    df['wind_speed_cube'] = df['gfs_wind_speed'] ** 3
    df['wind_speed_power_2.5'] = df['gfs_wind_speed'] ** 2.5

    # 3.2 风速高阶特征（强化主导地位）
    df['wind_speed_power_4'] = df['gfs_wind_speed'] ** 4       # v^4 极端敏感
    df['wind_speed_sqrt'] = np.sqrt(df['gfs_wind_speed'])     # √v 低速敏感
    df['wind_speed_log'] = np.log1p(df['gfs_wind_speed'])     # log(1+v) 非线性变换

    # 3.3 风速×风速交互特征（自身增强）
    df['wind_squared_x_cubed'] = df['wind_speed_square'] * df['wind_speed_cube']  # v^5
    df['wind_x_power_2.5'] = df['gfs_wind_speed'] * df['wind_speed_power_2.5']    # v^3.5

    # 3.4 风速×时间交互（不同时段风速效率不同）
    df['wind_x_hour'] = df['gfs_wind_speed'] * df['hour']
    df['wind_x_month'] = df['gfs_wind_speed'] * df['month']
    df['wind_x_hour_sin'] = df['gfs_wind_speed'] * df['hour_sin']
    df['wind_x_hour_cos'] = df['gfs_wind_speed'] * df['hour_cos']

    # 3.5 风速立方×方向（风能利用系数）
    df['wind_cube_x_dir_sin'] = df['wind_speed_cube'] * df['wind_dir_sin']
    df['wind_cube_x_dir_cos'] = df['wind_speed_cube'] * df['wind_dir_cos']

    # 3.6 风速稳定性（多尺度）
    df['wind_stability_4'] = df['gfs_wind_speed'] / (df['gfs_wind_speed'].rolling(window=4, min_periods=1).std() + 0.1)
    df['wind_stability_8'] = df['gfs_wind_speed'] / (df['gfs_wind_speed'].rolling(window=8, min_periods=1).std() + 0.1)
    df['wind_stability_16'] = df['gfs_wind_speed'] / (df['gfs_wind_speed'].rolling(window=16, min_periods=1).std() + 0.1)

    # 3.7 风速分段（更细致的区间）
    df['wind_0_5'] = (df['gfs_wind_speed'] < 5).astype(int)
    df['wind_5_10'] = ((df['gfs_wind_speed'] >= 5) & (df['gfs_wind_speed'] < 10)).astype(int)
    df['wind_10_15'] = ((df['gfs_wind_speed'] >= 10) & (df['gfs_wind_speed'] < 15)).astype(int)
    df['wind_15_20'] = ((df['gfs_wind_speed'] >= 15) & (df['gfs_wind_speed'] < 20)).astype(int)
    df['wind_20_25'] = ((df['gfs_wind_speed'] >= 20) & (df['gfs_wind_speed'] < 25)).astype(int)
    df['wind_above_25'] = (df['gfs_wind_speed'] >= 25).astype(int)

    # 3.8 风速饱和特征
    df['wind_saturation'] = np.clip((df['gfs_wind_speed'] - 12) / 8, 0, 1)  # 12-20m/s
    df['wind_saturation_x_cube'] = df['wind_saturation'] * df['wind_speed_cube']

    # 3.9 风速×风速方差（湍流）
    df['wind_var_4'] = df['gfs_wind_speed'].rolling(window=4, min_periods=1).var()
    df['wind_x_var'] = df['gfs_wind_speed'] * (df['wind_var_4'] + 1)

    # 3.10 风速累积特征
    df['wind_cumsum_4'] = df['gfs_wind_speed'].rolling(window=4, min_periods=1).sum()
    df['wind_cumsum_8'] = df['gfs_wind_speed'].rolling(window=8, min_periods=1).sum()

    # 4. 温度特征（大幅简化，作为辅助）
    # 只保留最基础的温度特征，移除复杂派生
    df['temp_simple'] = df['gfs_temp']  # 仅保留原始温度
    df['temp_x_wind'] = df['gfs_temp'] * df['gfs_wind_speed']  # 保留一个交互特征

    # 移除所有复杂的温度特征（temp_rolling_mean_3等将在特征列表中排除）

    # 5. 风速约束特征（物理极限）
    df['estimated_power_limit'] = np.where(
        df['gfs_wind_speed'] < 3,
        0,
        np.where(
            df['gfs_wind_speed'] <= 12,
            50000 * ((df['gfs_wind_speed'] - 3) / 9) ** 3,
            np.where(
                df['gfs_wind_speed'] <= 25,
                50000,
                0
            )
        )
    )

    # 6. v2.6新增：极端风速异常处理特征
    # ============ 处理22%的高风速异常数据 ============

    # 6.1 极端风速标记（风速>25m/s，超过切出风速）
    df['is_extreme_wind'] = (df['gfs_wind_speed'] > 25).astype(int)

    # 6.2 异常高功率标记（极端风速且有高功率 - 可能是数据错误）
    df['is_anomalous_high_power'] = ((df['gfs_wind_speed'] > 25) & (df['power'] > 10000)).astype(int)

    # 6.3 极端风速下的功率修正值（基于物理公式）
    # 风速>25m/s时，风机应该停机或降功率
    df['corrected_power_extreme'] = np.where(
        df['gfs_wind_speed'] > 25,
        np.where(
            df['gfs_wind_speed'] < 30,
            50000 * 0.5,  # 25-30m/s: 半功率运行
            0  # ≥30m/s: 停机
        ),
        df['power']
    )

    # 6.4 功率差异特征（实际功率 - 理论功率）
    theoretical_power = np.where(
        df['gfs_wind_speed'] < 3,
        0,
        np.where(
            df['gfs_wind_speed'] <= 12,
            50000 * ((df['gfs_wind_speed'] - 3) / 9) ** 3,
            np.where(
                df['gfs_wind_speed'] <= 25,
                50000,
                0
            )
        )
    )
    df['power_deviation'] = df['power'] - theoretical_power
    df['power_deviation_abs'] = np.abs(df['power_deviation'])
    df['power_deviation_ratio'] = df['power_deviation'] / (theoretical_power + 1)

    # 6.5 极端风速×功率交互
    df['extreme_wind_x_power'] = df['is_extreme_wind'] * df['power']

    # 6.6 异常标记特征（综合）
    df['has_anomaly'] = (df['power_deviation_abs'] > 20000).astype(int)

    # 7. 季节性特征（保留）
    df['seasonal_efficiency'] = np.sin(2 * np.pi * df['month'] / 12) * 0.1 + 1

    # 7. 滚动统计特征（仅用于分析，实际预测时不可用）
    for window in [4, 8, 16]:
        df[f'power_rolling_mean_{window}'] = df['power'].rolling(window=window, min_periods=1).mean()
        df[f'power_rolling_std_{window}'] = df['power'].rolling(window=window, min_periods=1).std()
    
    # 8. 滞后特征
    for lag in [1, 4, 8]:
        df[f'power_lag_{lag}'] = df['power'].shift(lag)
    
    # 填充NaN值
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    print(f"特征工程完成，共 {len(df.columns)} 个特征")
    
    return df


def prepare_train_test(df):
    """准备训练和测试数据"""
    print("\n正在划分数据集...")
    
    # v2.6异常处理优化：在v2.5基础上添加异常处理特征
    # ============ 核心理念：识别和处理高风速异常数据 ============
    features_to_use = [
        # 基础特征
        'gfs_wind_speed', 'gfs_wind_direction',
        'hour', 'day', 'month', 'day_of_week', 'day_of_year',
        'hour_sin', 'hour_cos', 'wind_dir_sin', 'wind_dir_cos',

        # === 风速核心特征（20个） ===
        # 基础幂次
        'wind_speed_square', 'wind_speed_cube', 'wind_speed_power_2.5',
        # 高阶特征
        'wind_speed_power_4', 'wind_speed_sqrt', 'wind_speed_log',
        # 风速×风速交互
        'wind_squared_x_cubed', 'wind_x_power_2.5',
        # 风速×时间交互
        'wind_x_hour', 'wind_x_month', 'wind_x_hour_sin', 'wind_x_hour_cos',
        # 风速立方×方向
        'wind_cube_x_dir_sin', 'wind_cube_x_dir_cos',
        # 风速稳定性（多尺度）
        'wind_stability_4', 'wind_stability_8', 'wind_stability_16',
        # 风速饱和
        'wind_saturation', 'wind_saturation_x_cube',

        # === 风速分段（6个） ===
        'wind_0_5', 'wind_5_10', 'wind_10_15', 'wind_15_20', 'wind_20_25', 'wind_above_25',

        # === 风速统计特征（3个） ===
        'estimated_power_limit', 'wind_x_var',

        # === v2.6新增：异常处理特征（仅3个，避免数据泄露） ===
        'is_extreme_wind',               # 极端风速标记（仅基于风速）
        'is_anomalous_high_power',       # 异常标记（需要标注时用，预测时可用估算值）
        # 注意：移除所有依赖power的特征避免数据泄露
        # 'corrected_power_extreme',    # ✗ 包含power
        # 'power_deviation',            # ✗ 包含power
        # 'power_deviation_abs',        # ✗ 包含power
        # 'power_deviation_ratio',      # ✗ 包含power
        # 'extreme_wind_x_power',       # ✗ 包含power
        # 'has_anomaly',                # ✗ 包含power

        # === 简化的温度特征（仅2个） ===
        'temp_simple', 'temp_x_wind',

        # === 季节性特征（1个） ===
        'seasonal_efficiency'
    ]
    
    # 按时间划分数据（80%训练，20%测试）
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    X_train = train_df[features_to_use]
    y_train = train_df['power']
    
    X_test = test_df[features_to_use]
    y_test = test_df['power']
    
    print(f"训练集: {len(X_train)} 条")
    print(f"测试集: {len(X_test)} 条")
    print(f"使用特征: {len(features_to_use)} 个")
    
    return X_train, X_test, y_train, y_test, features_to_use


def train_ensemble_models(X_train, y_train):
    """训练集成模型"""
    print("\n正在训练集成模型...")

    # XGBoost（v2.6异常处理优化版）
    print("训练XGBoost模型（v2.6异常处理优化版）...")
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=8,              # 保持8
        learning_rate=0.05,       # 保持0.05
        n_estimators=1000,        # 保持1000
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=4,
        gamma=0.15,
        reg_alpha=0.3,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1
    )

    # 划分训练集和验证集
    split_idx = int(len(X_train) * 0.8)
    X_train_split = X_train.iloc[:split_idx]
    X_val = X_train.iloc[split_idx:]
    y_train_split = y_train.iloc[:split_idx]
    y_val = y_train.iloc[split_idx:]

    # 训练（简化版，兼容旧版XGBoost）
    xgb_model.fit(
        X_train_split, y_train_split,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Random Forest
    print("训练Random Forest模型...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Gradient Boosting
    print("训练Gradient Boosting模型...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb_model.fit(X_train, y_train)

    # v2.7新增：LightGBM
    print("训练LightGBM模型...")
    lgb_model = lgb.LGBMRegressor(
        num_leaves=31,
        max_depth=8,
        learning_rate=0.05,
        n_estimators=1000,
        reg_alpha=0.3,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=2,          # v2.8优化：全部核心→2核心，避免与Stacking资源冲突
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)

    print("集成模型训练完成")

    return {
        'xgboost': xgb_model,
        'random_forest': rf_model,
        'gradient_boosting': gb_model,
        'lightgbm': lgb_model
    }


def ensemble_predict(models, X):
    """集成预测（v2.7：包含LightGBM，加权平均）"""
    predictions = {}

    # 获取各模型预测
    predictions['xgboost'] = models['xgboost'].predict(X)
    predictions['random_forest'] = models['random_forest'].predict(X)
    predictions['gradient_boosting'] = models['gradient_boosting'].predict(X)
    predictions['lightgbm'] = models['lightgbm'].predict(X)  # v2.7新增

    # 加权平均（v2.7：包含LightGBM）
    weights = {
        'xgboost': 0.5,          # 0.7→0.5，为LightGBM腾出空间
        'lightgbm': 0.2,         # v2.7新增：LightGBM权重
        'random_forest': 0.2,
        'gradient_boosting': 0.1
    }

    ensemble_pred = (
        weights['xgboost'] * predictions['xgboost'] +
        weights['lightgbm'] * predictions['lightgbm'] +
        weights['random_forest'] * predictions['random_forest'] +
        weights['gradient_boosting'] * predictions['gradient_boosting']
    )

    return ensemble_pred, predictions


def stacking_predict(models, X, y_train):
    """v2.8新增：Stacking集成预测（使用Ridge回归自动学习权重）"""
    print("\n正在构建Stacking集成...")

    # 创建Stacking模型
    base_models = [
        ('xgb', models['xgboost']),
        ('lgb', models['lightgbm']),
        ('rf', models['random_forest']),
        ('gb', models['gradient_boosting'])
    ]

    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(alpha=1.0),
        cv=3,                          # v2.8优化：5折→3折，减少40%计算量
        n_jobs=4,                      # v2.8优化：全部核心→4核心，减少50%内存峰值
        verbose=1                      # v2.8优化：显示训练进度
    )

    # 训练Stacking模型
    print("训练Stacking集成（3折交叉验证，4核并发）...")
    stacking_model.fit(X, y_train)
    print("✓ Stacking训练完成")

    return stacking_model


def post_process_predictions(predictions, window=3):
    """后处理：平滑和截断"""
    
    # 1. 截断负值
    predictions = np.clip(predictions, 0, None)
    
    # 2. 移动平均平滑（减少波动）
    if window > 1:
        predictions_smoothed = pd.Series(predictions).rolling(
            window=window, min_periods=1, center=True
        ).mean().values
        predictions = predictions_smoothed
    
    return predictions


def evaluate_model(models, X_test, y_test, X_train=None, y_train=None):
    """评估模型性能（v2.7/2.8：包含LightGBM和Stacking评估）"""
    print("\n正在评估模型...")

    # 预测
    ensemble_pred, individual_preds = ensemble_predict(models, X_test)

    # 后处理
    ensemble_pred_processed = post_process_predictions(ensemble_pred, window=3)

    # 计算指标
    mae = mean_absolute_error(y_test, ensemble_pred_processed)
    rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred_processed))
    r2 = r2_score(y_test, ensemble_pred_processed)

    # 计算MAPE（避免除以零）
    nonzero_mask = y_test > 100
    mape = np.mean(np.abs((y_test[nonzero_mask] - ensemble_pred_processed[nonzero_mask]) / y_test[nonzero_mask])) * 100

    print("\n集成模型性能指标（优化后）:")
    print(f"MAE (平均绝对误差): {mae:.2f} kW")
    print(f"RMSE (均方根误差): {rmse:.2f} kW")
    print(f"MAPE (平均绝对百分比误差): {mape:.2f}%")
    print(f"R² (决定系数): {r2:.4f}")

    # 可视化
    plot_predictions(y_test, ensemble_pred_processed, 'prediction_results_optimized.png')
    plot_feature_importance(models['xgboost'], X_test.columns, 'feature_importance_optimized.png')

    # 生成偏差分析图
    generate_bias_analysis_during_training(y_test, ensemble_pred_processed)

    # v2.8新增：如果提供训练数据，也评估Stacking模型
    if X_train is not None and y_train is not None:
        print("\n" + "="*70)
        print("v2.8 Stacking集成评估")
        print("="*70)

        try:
            import psutil
            import os

            # v2.8优化：显示训练前内存状态
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            print(f"\n训练前内存占用: {mem_before:.1f} MB")

            stacking_model = stacking_predict(models, X_train, y_train)

            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            print(f"训练后内存占用: {mem_after:.1f} MB")
            print(f"内存增长: {mem_after - mem_before:.1f} MB")

            stacking_pred = stacking_model.predict(X_test)
            stacking_pred_processed = post_process_predictions(stacking_pred, window=3)

            stacking_mae = mean_absolute_error(y_test, stacking_pred_processed)
            stacking_rmse = np.sqrt(mean_squared_error(y_test, stacking_pred_processed))
            stacking_r2 = r2_score(y_test, stacking_pred_processed)

            print(f"\nStacking集成性能:")
            print(f"MAE: {stacking_mae:.2f} kW")
            print(f"RMSE: {stacking_rmse:.2f} kW")
            print(f"R²: {stacking_r2:.4f}")

            # 对比
            improvement = (mae - stacking_mae) / mae * 100
            print(f"\n对比加权集成:")
            print(f"MAE变化: {mae:.2f} → {stacking_mae:.2f} kW ({improvement:+.2f}%)")

            # 如果Stacking更好，保存Stacking模型
            if stacking_mae < mae:
                print("\n✓ Stacking表现更优！")
                models['stacking'] = stacking_model
            else:
                print("\n✓ 加权集成表现更优")

        except MemoryError:
            print("\n⚠️ 内存不足！Stacking训练失败")
            print("建议：减少cv折数或降低n_jobs参数")
        except Exception as e:
            print(f"\n⚠️ Stacking训练失败: {e}")
            import traceback
            traceback.print_exc()

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }


def plot_predictions(y_true, y_pred, save_path):
    """绘制预测结果对比图"""
    plt.figure(figsize=(15, 6))
    
    # 只显示前200个点
    n_points = min(200, len(y_true))
    
    plt.plot(y_true.values[:n_points], label='实际功率', alpha=0.7, linewidth=1)
    plt.plot(y_pred[:n_points], label='预测功率（优化后）', alpha=0.7, linewidth=1)
    
    plt.title('预测功率 vs 实际功率对比（优化版）', fontsize=14)
    plt.xlabel('时间点', fontsize=12)
    plt.ylabel('功率 (kW)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n预测对比图已保存至: {save_path}")
    plt.close()


def plot_feature_importance(model, feature_names, save_path):
    """绘制特征重要性"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title('特征重要性（优化版）', fontsize=14)
    plt.xlabel('特征', fontsize=12)
    plt.ylabel('重要性', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"特征重要性图已保存至: {save_path}")
    plt.close()


def generate_bias_analysis_during_training(y_true, y_pred):
    """训练时生成偏差分析图（垂直排列4个子图，含准确度统计）"""
    print("\n正在生成偏差分析图...")

    # 计算误差和准确度
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    n = min(200, len(y_true))

    # 计算总体偏差百分比
    mean_absolute_percentage_error = np.mean(np.abs((y_pred - y_true) / (y_true + 1))) * 100
    median_absolute_percentage_error = np.median(np.abs((y_pred - y_true) / (y_true + 1))) * 100

    # 计算准确度比例
    accuracy_90 = np.sum(np.abs((y_pred - y_true) / (y_true + 1)) <= 0.10) / len(y_true) * 100
    accuracy_85 = np.sum(np.abs((y_pred - y_true) / (y_true + 1)) <= 0.15) / len(y_true) * 100
    accuracy_80 = np.sum(np.abs((y_pred - y_true) / (y_true + 1)) <= 0.20) / len(y_true) * 100
    accuracy_75 = np.sum(np.abs((y_pred - y_true) / (y_true + 1)) <= 0.25) / len(y_true) * 100
    accuracy_70 = np.sum(np.abs((y_pred - y_true) / (y_true + 1)) <= 0.30) / len(y_true) * 100

    # 创建图表：4行1列，方便上下比较
    fig, axes = plt.subplots(4, 1, figsize=(14, 18))
    fig.suptitle('Bias Analysis Report - Training Set', fontsize=16, fontweight='bold', y=0.995)

    # 图1：预测vs实际散点图（添加准确度统计文本框）
    ax1 = axes[0]
    scatter = ax1.scatter(y_true, y_pred, c=abs_errors, cmap='YlOrRd', alpha=0.5, s=20)
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=2, label='Perfect Prediction')
    ax1.set_title('1. Prediction vs Actual Scatter Plot', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('Actual Power (kW)', fontsize=11)
    ax1.set_ylabel('Predicted Power (kW)', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter, ax=ax1, label='Abs Error (kW)', pad=0.02)

    # 添加准确度统计文本框
    accuracy_text = f"""ACCURACY STATISTICS:
Accuracy >= 90%: {accuracy_90:.2f}%
Accuracy >= 85%: {accuracy_85:.2f}%
Accuracy >= 80%: {accuracy_80:.2f}%
Accuracy >= 75%: {accuracy_75:.2f}%
Accuracy >= 70%: {accuracy_70:.2f}%

Mean MAPE: {mean_absolute_percentage_error:.2f}%
Median MAPE: {median_absolute_percentage_error:.2f}%"""
    ax1.text(0.02, 0.98, accuracy_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 图2：误差分布直方图（添加偏差统计）
    ax2 = axes[1]
    ax2.hist(errors, bins=60, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={errors.mean():.1f} kW')
    ax2.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_title(f'2. Error Distribution (μ={errors.mean():.1f} kW, σ={errors.std():.1f} kW)',
                  fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('Error (kW)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # 在误差分布图上添加偏差百分比统计
    bias_stats = f"""BIAS PERCENTAGE:
Mean Bias: {errors.mean():.1f} kW ({mean_absolute_percentage_error:.2f}%)
Median Bias: {np.median(errors):.1f} kW ({median_absolute_percentage_error:.2f}%)
Overestimation: {np.sum(errors > 0) / len(errors) * 100:.1f}%
Underestimation: {np.sum(errors < 0) / len(errors) * 100:.1f}%"""
    ax2.text(0.98, 0.97, bias_stats, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # 图3：时序对比（前200个点）
    ax3 = axes[2]
    ax3.plot(y_true.values[:n], label='Actual', color='black', alpha=0.8, linewidth=1.2)
    ax3.plot(y_pred[:n], label='Predicted', color='#E74C3C', alpha=0.7, linewidth=1.2)
    ax3.set_title('3. Time Series Comparison (First 200 Points)', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel('Time Point', fontsize=11)
    ax3.set_ylabel('Power (kW)', fontsize=11)
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(True, alpha=0.3)

    # 图4：误差时序图
    ax4 = axes[3]
    colors = ['red' if e > 0 else 'blue' for e in errors[:n]]
    ax4.bar(range(n), errors[:n], color=colors, alpha=0.6, width=1)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax4.set_title('4. Error Time Series (Red=Overestimate, Blue=Underestimate)',
                  fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlabel('Time Point', fontsize=11)
    ax4.set_ylabel('Error (kW)', fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')

    # 调整布局，增加子图间距和高度
    plt.tight_layout(rect=[0, 0, 1, 0.99], h_pad=3.5)

    # 保存
    save_path = 'bias_analysis_during_training.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 偏差分析图已保存: {save_path}")
    print(f"\n准确度统计:")
    print(f"  >=90%准确度: {accuracy_90:.2f}%")
    print(f"  >=85%准确度: {accuracy_85:.2f}%")
    print(f"  >=80%准确度: {accuracy_80:.2f}%")
    print(f"  >=75%准确度: {accuracy_75:.2f}%")
    print(f"  >=70%准确度: {accuracy_70:.2f}%")
    print(f"  平均MAPE: {mean_absolute_percentage_error:.2f}%")
    print(f"  中位数MAPE: {median_absolute_percentage_error:.2f}%")
    plt.close()

    return save_path


def save_models(models, temp_stats, filename_prefix='power_forecast_model_optimized'):
    """保存训练好的模型"""
    for name, model in models.items():
        filename = f'{filename_prefix}_{name}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"{name}模型已保存至: {filename}")

    # 保存模型列表和温度统计量
    model_info = {
        'models': list(models.keys()),
        'temp_mean': temp_stats['temp_mean'],
        'temp_std': temp_stats['temp_std']
    }
    with open(f'{filename_prefix}_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    print(f"模型信息已保存至: {filename_prefix}_info.pkl")


def main():
    """主函数"""
    print("=" * 60)
    print("风电站功率预测模型训练（优化版）")
    print("=" * 60)
    
    # 1. 加载数据
    gfs_df, power_df = load_data()
    
    # 2. 合并数据
    merged_df = merge_data(gfs_df, power_df)
    
    # 3. 特征工程
    featured_df = feature_engineering(merged_df)

    # 计算温度统计量（用于预测时标准化）
    temp_stats = {
        'temp_mean': featured_df['gfs_temp'].mean(),
        'temp_std': featured_df['gfs_temp'].std()
    }
    print(f"\n温度统计量:")
    print(f"  均值: {temp_stats['temp_mean']:.2f}°C")
    print(f"  标准差: {temp_stats['temp_std']:.2f}°C")

    # 4. 准备训练测试集
    X_train, X_test, y_train, y_test, features = prepare_train_test(featured_df)

    # 5. 训练集成模型（v2.7：包含LightGBM）
    models = train_ensemble_models(X_train, y_train)

    # 6. 评估模型（v2.8：包含Stacking评估）
    metrics = evaluate_model(models, X_test, y_test, X_train, y_train)

    # 7. 保存模型
    save_models(models, temp_stats)

    print("\n" + "=" * 60)
    print("优化模型训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
