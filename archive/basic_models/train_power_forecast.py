"""
风电站功率预测模型训练脚本
使用GFS气象数据预测电站发电功率
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
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
    
    # 加载功率数据（跳过第一行空行）
    power_df = pd.read_csv('data_history_power.csv', header=None, skiprows=1, names=['timestamp', 'power'])
    power_df['timestamp'] = pd.to_datetime(power_df['timestamp'])
    
    print(f"GFS数据: {len(gfs_df)} 条记录")
    print(f"功率数据: {len(power_df)} 条记录")
    
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
    """特征工程"""
    print("\n正在进行特征工程...")
    
    df = df.copy()
    
    # 1. 时间特征
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    # 小时的正弦余弦编码（处理周期性）
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # 2. 风向的正弦余弦编码（处理周期性）
    df['wind_dir_sin'] = np.sin(2 * np.pi * df['gfs_wind_direction'] / 360)
    df['wind_dir_cos'] = np.cos(2 * np.pi * df['gfs_wind_direction'] / 360)
    
    # 3. 风速的三次方（功率与风速的三次方成正比）
    df['wind_speed_cube'] = df['gfs_wind_speed'] ** 3
    
    # 4. 滚动统计特征（使用过去的功率数据）
    # 注意：实际预测时这些特征不可用，所以只用于特征重要性分析
    for window in [4, 8, 16]:  # 1小时、2小时、4小时
        df[f'power_rolling_mean_{window}'] = df['power'].rolling(window=window, min_periods=1).mean()
        df[f'power_rolling_std_{window}'] = df['power'].rolling(window=window, min_periods=1).std()
    
    # 5. 滞后特征（同样，实际预测时不可用）
    for lag in [1, 4, 8]:
        df[f'power_lag_{lag}'] = df['power'].shift(lag)
    
    # 填充NaN值
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    print(f"特征工程完成，共 {len(df.columns)} 个特征")
    
    return df


def prepare_train_test(df):
    """准备训练和测试数据"""
    print("\n正在划分数据集...")
    
    # 不使用滞后和滚动特征（实际预测时不可用）
    features_to_use = [
        'gfs_temp', 'gfs_wind_speed', 'gfs_wind_direction',
        'hour', 'day', 'month', 'day_of_week', 'day_of_year',
        'hour_sin', 'hour_cos', 'wind_dir_sin', 'wind_dir_cos',
        'wind_speed_cube'
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


def train_model(X_train, y_train):
    """训练XGBoost模型"""
    print("\n正在训练XGBoost模型...")
    
    # 划分训练集和验证集
    split_idx = int(len(X_train) * 0.8)
    X_train_split = X_train.iloc[:split_idx]
    X_val = X_train.iloc[split_idx:]
    y_train_split = y_train.iloc[:split_idx]
    y_val = y_train.iloc[split_idx:]
    
    # XGBoost参数
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBRegressor(**params)
    
    # 训练模型（使用callbacks实现早停）
    model.fit(
        X_train_split, y_train_split,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print("模型训练完成")
    return model


def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    print("\n正在评估模型...")
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算指标
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print("\n模型性能指标:")
    print(f"MAE (平均绝对误差): {mae:.2f} kW")
    print(f"RMSE (均方根误差): {rmse:.2f} kW")
    print(f"MAPE (平均绝对百分比误差): {mape:.2f}%")
    print(f"R² (决定系数): {r2:.4f}")
    
    # 可视化预测结果
    plot_predictions(y_test, y_pred)
    
    # 特征重要性
    plot_feature_importance(model, X_test.columns)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }


def plot_predictions(y_true, y_pred, save_path='prediction_results.png'):
    """绘制预测结果对比图"""
    plt.figure(figsize=(15, 6))
    
    # 只显示前200个点以避免图像过于密集
    n_points = min(200, len(y_true))
    
    plt.plot(y_true.values[:n_points], label='实际功率', alpha=0.7, linewidth=1)
    plt.plot(y_pred[:n_points], label='预测功率', alpha=0.7, linewidth=1)
    
    plt.title('预测功率 vs 实际功率对比', fontsize=14)
    plt.xlabel('时间点', fontsize=12)
    plt.ylabel('功率 (kW)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n预测对比图已保存至: {save_path}")
    plt.close()


def plot_feature_importance(model, feature_names, save_path='feature_importance.png'):
    """绘制特征重要性"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title('特征重要性', fontsize=14)
    plt.xlabel('特征', fontsize=12)
    plt.ylabel('重要性', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"特征重要性图已保存至: {save_path}")
    plt.close()


def save_model(model, filename='power_forecast_model.pkl'):
    """保存训练好的模型"""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n模型已保存至: {filename}")


def main():
    """主函数"""
    print("=" * 60)
    print("风电站功率预测模型训练")
    print("=" * 60)
    
    # 1. 加载数据
    gfs_df, power_df = load_data()
    
    # 2. 合并数据
    merged_df = merge_data(gfs_df, power_df)
    
    # 3. 特征工程
    featured_df = feature_engineering(merged_df)
    
    # 4. 准备训练测试集
    X_train, X_test, y_train, y_test, features = prepare_train_test(featured_df)
    
    # 5. 训练模型
    model = train_model(X_train, y_train)
    
    # 6. 评估模型
    metrics = evaluate_model(model, X_test, y_test)
    
    # 7. 保存模型
    save_model(model)
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
