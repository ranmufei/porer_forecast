"""
分层模型训练脚本
针对不同功率段训练专门模型
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_and_prepare_data():
    """加载和准备数据"""
    print("正在加载数据...")
    
    # 加载GFS数据
    gfs_df = pd.read_csv('data_gfs_forecast.csv')
    gfs_df['timestamp'] = pd.to_datetime(gfs_df['timestamp'])
    
    # 加载功率数据
    power_df = pd.read_csv('data_history_power.csv', header=None, skiprows=1, 
                         names=['timestamp', 'power'])
    power_df['timestamp'] = pd.to_datetime(power_df['timestamp'])
    power_df['power'] = power_df['power'].clip(lower=0)
    
    # 合并数据
    merged_df = pd.merge(gfs_df, power_df, on='timestamp', how='inner')
    
    print(f"合并后数据: {len(merged_df)} 条记录")
    
    return merged_df


def feature_engineering(df):
    """特征工程"""
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
    
    # 3. 物理特征
    df['wind_speed_cube'] = df['gfs_wind_speed'] ** 3
    
    # 4. 气象约束特征
    df['wind_below_cutin'] = (df['gfs_wind_speed'] < 3).astype(int)
    df['wind_above_cutoff'] = (df['gfs_wind_speed'] > 25).astype(int)
    df['wind_in_ideal_range'] = ((df['gfs_wind_speed'] >= 10) & 
                                 (df['gfs_wind_speed'] <= 15)).astype(int)
    
    # 5. 功率分段特征
    df['estimated_max_power'] = 50000 * (df['gfs_wind_speed'] / 12) ** 3
    df['estimated_max_power'] = df['estimated_max_power'].clip(lower=0, upper=50000)
    
    # 6. 交互特征
    df['wind_temp_interaction'] = df['gfs_wind_speed'] * df['gfs_temp']
    
    # 7. 功率段标签（用于分类）
    df['power_segment'] = pd.cut(
        df['power'],
        bins=[-np.inf, 5000, 15000, 30000, np.inf],
        labels=['low', 'medium_low', 'medium_high', 'high']
    )
    
    return df


def define_power_segments():
    """定义功率段"""
    return {
        'zero': {
            'name': '零功率段',
            'threshold': 100,  # <100 kW视为零功率
            'features': ['gfs_temp', 'gfs_wind_speed', 'gfs_wind_direction',
                       'hour', 'day', 'month', 'day_of_week', 'day_of_year',
                       'hour_sin', 'hour_cos', 'wind_dir_sin', 'wind_dir_cos',
                       'wind_speed_cube', 'wind_below_cutin', 'wind_above_cutoff',
                       'wind_in_ideal_range', 'estimated_max_power',
                       'wind_temp_interaction']
        },
        'low': {
            'name': '低功率段',
            'range': (100, 5000),
            'features': ['gfs_temp', 'gfs_wind_speed', 'gfs_wind_direction',
                       'hour', 'day', 'month', 'day_of_week', 'day_of_year',
                       'hour_sin', 'hour_cos', 'wind_dir_sin', 'wind_dir_cos',
                       'wind_speed_cube', 'wind_below_cutin', 'wind_above_cutoff',
                       'wind_in_ideal_range', 'estimated_max_power',
                       'wind_temp_interaction']
        },
        'medium_low': {
            'name': '中低功率段',
            'range': (5000, 15000),
            'features': ['gfs_temp', 'gfs_wind_speed', 'gfs_wind_direction',
                       'hour', 'day', 'month', 'day_of_week', 'day_of_year',
                       'hour_sin', 'hour_cos', 'wind_dir_sin', 'wind_dir_cos',
                       'wind_speed_cube', 'wind_below_cutin', 'wind_above_cutoff',
                       'wind_in_ideal_range', 'estimated_max_power',
                       'wind_temp_interaction']
        },
        'medium_high': {
            'name': '中高功率段',
            'range': (15000, 30000),
            'features': ['gfs_temp', 'gfs_wind_speed', 'gfs_wind_direction',
                       'hour', 'day', 'month', 'day_of_week', 'day_of_year',
                       'hour_sin', 'hour_cos', 'wind_dir_sin', 'wind_dir_cos',
                       'wind_speed_cube', 'wind_below_cutin', 'wind_above_cutoff',
                       'wind_in_ideal_range', 'estimated_max_power',
                       'wind_temp_interaction']
        },
        'high': {
            'name': '高功率段',
            'range': (30000, 50000),
            'features': ['gfs_temp', 'gfs_wind_speed', 'gfs_wind_direction',
                       'hour', 'day', 'month', 'day_of_week', 'day_of_year',
                       'hour_sin', 'hour_cos', 'wind_dir_sin', 'wind_dir_cos',
                       'wind_speed_cube', 'wind_below_cutin', 'wind_above_cutoff',
                       'wind_in_ideal_range', 'estimated_max_power',
                       'wind_temp_interaction']
        }
    }


def train_segment_model(df, segment_key, segment_info, test_ratio=0.2):
    """训练单个功率段模型"""
    segment_name = segment_info['name']
    
    print(f"\n{'='*60}")
    print(f"训练 {segment_name} 模型")
    print(f"{'='*60}")
    
    # 筛选数据
    if segment_key == 'zero':
        # 零功率段：功率<100 kW
        segment_df = df[df['power'] < segment_info['threshold']].copy()
    else:
        # 其他功率段
        min_power, max_power = segment_info['range']
        segment_df = df[(df['power'] >= min_power) & 
                       (df['power'] < max_power)].copy()
    
    if len(segment_df) == 0:
        print(f"警告: {segment_name} 没有数据")
        return None
    
    print(f"数据量: {len(segment_df)} 条")
    
    # 准备特征
    X = segment_df[segment_info['features']]
    y = segment_df['power']
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42, shuffle=False
    )
    
    print(f"训练集: {len(X_train)} 条")
    print(f"测试集: {len(X_test)} 条")
    
    # 训练XGBoost模型
    print(f"正在训练XGBoost模型...")
    
    # 所有功率段都使用回归器
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # 评估
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # 计算MAPE
    nonzero_mask = y_test > 100
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((y_test[nonzero_mask] - y_pred[nonzero_mask]) / 
                               y_test[nonzero_mask])) * 100
    else:
        mape = np.nan
    
    print(f"MAE:  {mae:.2f} kW")
    print(f"RMSE: {rmse:.2f} kW")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²:   {r2:.4f}")
    
    metrics = {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}
    
    return model, metrics


def train_all_models(df):
    """训练所有功率段模型"""
    print(f"\n{'='*60}")
    print("训练分层模型")
    print(f"{'='*60}")
    
    segments = define_power_segments()
    models = {}
    all_metrics = []
    
    # 划分全局训练/测试集
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # 对每个功率段训练模型
    for segment_key, segment_info in segments.items():
        model, metrics = train_segment_model(
            train_df, segment_key, segment_info, test_ratio=0.2
        )
        
        if model is not None:
            models[segment_key] = {
                'model': model,
                'info': segment_info
            }
            metrics['segment'] = segment_key
            metrics['segment_name'] = segment_info['name']
            all_metrics.append(metrics)
    
    return models, all_metrics, test_df


def save_models(models):
    """保存所有模型"""
    print(f"\n{'='*60}")
    print("保存模型")
    print(f"{'='*60}")
    
    for segment_key, model_data in models.items():
        filename = f'hierarchical_model_{segment_key}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"{model_data['info']['name']} 模型已保存至: {filename}")
    
    # 保存模型信息
    model_info = {
        'segments': list(models.keys()),
        'segment_info': {k: v['info'] for k, v in models.items()}
    }
    with open('hierarchical_model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    print("模型信息已保存至: hierarchical_model_info.pkl")


def plot_metrics_comparison(all_metrics):
    """绘制各功率段性能对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 提取回归模型的指标
    regression_metrics = [m for m in all_metrics if 'mae' in m]
    
    if len(regression_metrics) > 0:
        df_metrics = pd.DataFrame(regression_metrics)
        
        # 子图1：MAE对比
        axes[0, 0].bar(df_metrics['segment_name'], df_metrics['mae'], 
                       color='skyblue', alpha=0.8)
        axes[0, 0].set_title('各功率段MAE对比', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('MAE (kW)', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=15, ha='right')
        
        # 子图2：RMSE对比
        axes[0, 1].bar(df_metrics['segment_name'], df_metrics['rmse'], 
                       color='lightgreen', alpha=0.8)
        axes[0, 1].set_title('各功率段RMSE对比', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('RMSE (kW)', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=15, ha='right')
        
        # 子图3：MAPE对比
        mape_data = df_metrics[['segment_name', 'mape']].dropna()
        if len(mape_data) > 0:
            axes[1, 0].bar(mape_data['segment_name'], mape_data['mape'], 
                           color='orange', alpha=0.8)
            axes[1, 0].set_title('各功率段MAPE对比', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylabel('MAPE (%)', fontsize=12)
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=15, ha='right')
        
        # 子图4：R²对比
        axes[1, 1].bar(df_metrics['segment_name'], df_metrics['r2'], 
                       color='lightcoral', alpha=0.8)
        axes[1, 1].set_title('各功率段R²对比', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('R²', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig('分层模型性能对比.png', dpi=300, bbox_inches='tight')
    print("\n性能对比图已保存至: 分层模型性能对比.png")
    plt.close()


def main():
    """主函数"""
    print("=" * 60)
    print("分层模型训练")
    print("=" * 60)
    
    # 1. 加载和准备数据
    df = load_and_prepare_data()
    
    # 2. 特征工程
    print("\n正在进行特征工程...")
    df = feature_engineering(df)
    print(f"特征工程完成，共 {len(df.columns)} 个特征")
    
    # 3. 训练所有模型
    models, all_metrics, test_df = train_all_models(df)
    
    # 4. 保存模型
    save_models(models)
    
    # 5. 绘制性能对比
    plot_metrics_comparison(all_metrics)
    
    # 6. 保存性能指标
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv('分层模型性能指标.csv', index=False)
    print("\n性能指标已保存至: 分层模型性能指标.csv")
    
    print("\n" + "=" * 60)
    print("分层模型训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
