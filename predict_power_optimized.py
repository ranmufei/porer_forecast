"""
风电站功率预测脚本（优化版）
使用集成模型进行功率预测
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, Union


def load_models(model_prefix='power_forecast_model_optimized'):
    """加载所有集成模型和温度统计量"""
    models = {}

    try:
        # 加载模型信息和温度统计量
        with open(f'{model_prefix}_info.pkl', 'rb') as f:
            model_info = pickle.load(f)

        # 加载各个模型
        for model_name in model_info['models']:
            filename = f'{model_prefix}_{model_name}.pkl'
            with open(filename, 'rb') as f:
                models[model_name] = pickle.load(f)

        # 提取温度统计量
        temp_stats = {
            'temp_mean': model_info.get('temp_mean', 15.0),  # 默认值15°C
            'temp_std': model_info.get('temp_std', 8.0)      # 默认值8°C
        }

        print(f"成功加载 {len(models)} 个模型")
        print(f"温度统计量: 均值={temp_stats['temp_mean']:.2f}°C, 标准差={temp_stats['temp_std']:.2f}°C")

        return models, temp_stats

    except FileNotFoundError:
        print("错误：找不到优化模型文件")
        print("请先运行 train_power_forecast_optimized.py 训练模型")
        return None, None


def prepare_features(gfs_data: Union[pd.DataFrame, Dict], temp_stats: Dict = None) -> pd.DataFrame:
    """准备特征数据"""

    # 如果temp_stats未提供，使用默认值
    if temp_stats is None:
        temp_stats = {'temp_mean': 15.0, 'temp_std': 8.0}

    # 如果是字典，转换为DataFrame
    if isinstance(gfs_data, dict):
        df = pd.DataFrame([gfs_data])
    else:
        df = gfs_data.copy()
    
    # 确保timestamp是datetime类型
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
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
    df['wind_in_ideal_range'] = ((df['gfs_wind_speed'] >= 10) & (df['gfs_wind_speed'] <= 15)).astype(int)
    
    # 5. 功率分段特征
    df['estimated_max_power'] = 50000 * (df['gfs_wind_speed'] / 12) ** 3
    df['estimated_max_power'] = df['estimated_max_power'].clip(lower=0, upper=50000)
    
    # 6. 交互特征
    df['wind_temp_interaction'] = df['gfs_wind_speed'] * df['gfs_temp']

    # 6.5 新增：温度高级特征（基于深度学习论文研究）
    # 温度变化率（捕捉温度趋势）
    df['temp_change'] = df['gfs_temp'].diff()
    df['temp_change_abs'] = np.abs(df['temp_change'])

    # 温度移动平均（平滑短期波动）
    df['temp_rolling_mean_3'] = df['gfs_temp'].rolling(window=3, min_periods=1).mean()

    # 温度标准化（使用训练集的统计量）
    df['gfs_temp_normalized'] = (df['gfs_temp'] - temp_stats['temp_mean']) / temp_stats['temp_std']

    # 温度分段特征（不同温度范围影响不同）
    df['temp_below_0'] = (df['gfs_temp'] < 0).astype(int)       # 低温（<0°C）
    df['temp_0_15'] = ((df['gfs_temp'] >= 0) & (df['gfs_temp'] < 15)).astype(int)  # 常温（0-15°C）
    df['temp_15_25'] = ((df['gfs_temp'] >= 15) & (df['gfs_temp'] < 25)).astype(int) # 温和（15-25°C）
    df['temp_above_25'] = (df['gfs_temp'] >= 25).astype(int)    # 高温（≥25°C）

    # 填充NaN值（diff()和rolling()会产生NaN）
    df = df.fillna(method='bfill').fillna(method='ffill')

    return df


def ensemble_predict(models: Dict, X: pd.DataFrame) -> np.ndarray:
    """集成预测（加权平均）"""
    
    # 选择特征
    feature_columns = [
        'gfs_temp', 'gfs_wind_speed', 'gfs_wind_direction',
        'hour', 'day', 'month', 'day_of_week', 'day_of_year',
        'hour_sin', 'hour_cos', 'wind_dir_sin', 'wind_dir_cos',
        'wind_speed_cube',
        'wind_below_cutin', 'wind_above_cutoff', 'wind_in_ideal_range',
        'estimated_max_power', 'wind_temp_interaction',
        # 新增温度高级特征（基于深度学习论文）
        'temp_change', 'temp_change_abs', 'temp_rolling_mean_3',
        'gfs_temp_normalized',
        'temp_below_0', 'temp_0_15', 'temp_15_25', 'temp_above_25'
    ]
    
    X_features = X[feature_columns]
    
    # 获取各模型预测
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X_features)
    
    # 加权平均
    weights = {
        'xgboost': 0.5,
        'random_forest': 0.3,
        'gradient_boosting': 0.2
    }
    
    ensemble_pred = (
        weights['xgboost'] * predictions['xgboost'] +
        weights['random_forest'] * predictions['random_forest'] +
        weights['gradient_boosting'] * predictions['gradient_boosting']
    )
    
    return ensemble_pred


def post_process_predictions(predictions: np.ndarray, 
                         window: int = 3,
                         clip_negative: bool = True,
                         max_power: float = None) -> np.ndarray:
    """
    后处理：平滑和截断
    
    参数:
        predictions: 原始预测值
        window: 移动平均窗口大小
        clip_negative: 是否截断负值
        max_power: 最大功率限制（可选）
    """
    
    # 1. 截断负值
    if clip_negative:
        predictions = np.clip(predictions, 0, None)
    
    # 2. 截断最大值
    if max_power is not None:
        predictions = np.clip(predictions, None, max_power)
    
    # 3. 移动平均平滑
    if window > 1:
        predictions_smoothed = pd.Series(predictions).rolling(
            window=window, min_periods=1, center=True
        ).mean().values
        predictions = predictions_smoothed
    
    return predictions


def predict_from_dict(models, temp_stats, gfs_data: Dict,
                   post_process: bool = True) -> float:
    """
    从字典预测单点功率

    参数:
        models: 集成模型字典
        temp_stats: 温度统计量字典
        gfs_data: GFS数据字典
        post_process: 是否应用后处理

    返回:
        预测功率值
    """
    # 准备特征
    df = prepare_features(gfs_data, temp_stats)

    # 集成预测
    prediction = ensemble_predict(models, df)[0]

    # 后处理
    if post_process:
        prediction = post_process_predictions(np.array([prediction]), window=1)[0]
    
    return prediction


def predict_from_csv(models, temp_stats, csv_path: str,
                   post_process: bool = True,
                   window: int = 3) -> pd.DataFrame:
    """
    从CSV文件批量预测

    参数:
        models: 集成模型字典
        temp_stats: 温度统计量字典
        csv_path: GFS数据CSV文件路径
        post_process: 是否应用后处理
        window: 后处理窗口大小

    返回:
        包含预测结果的DataFrame
    """
    print(f"正在读取GFS数据: {csv_path}")

    # 读取GFS数据
    gfs_df = pd.read_csv(csv_path)

    # 准备特征
    print("正在准备特征...")
    featured_df = prepare_features(gfs_df, temp_stats)
    
    # 集成预测
    print("正在进行集成预测...")
    predictions = ensemble_predict(models, featured_df)
    
    # 后处理
    if post_process:
        print("正在应用后处理...")
        predictions = post_process_predictions(predictions, window=window)
    
    # 创建结果DataFrame
    results = pd.DataFrame({
        'timestamp': gfs_df['timestamp'],
        'predicted_power': predictions
    })
    
    print("预测完成！")
    
    return results


def main():
    """主函数"""
    print("=" * 60)
    print("风电站功率预测（优化版）")
    print("=" * 60)
    
    # 1. 加载模型
    print("\n正在加载模型...")
    models, temp_stats = load_models()

    if models is None:
        return

    print("模型加载完成\n")

    # 2. 读取GFS数据并预测
    results = predict_from_csv(
        models,
        temp_stats,
        'data_gfs_forecast.csv',
        post_process=True,
        window=3
    )
    
    # 3. 保存结果
    output_file = 'predictions_optimized.csv'
    results.to_csv(output_file, index=False)
    print(f"\n预测结果已保存至: {output_file}")
    print(f"共预测 {len(results)} 个时间点")
    
    # 4. 显示前10条结果
    print("\n前10条预测结果:")
    print(results.head(10))
    
    print("\n" + "=" * 60)
    print("预测完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
