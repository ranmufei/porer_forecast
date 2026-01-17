"""
风电站功率预测脚本
使用训练好的模型和GFS气象数据预测电站发电功率
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import sys


def load_model(model_path='power_forecast_model.pkl'):
    """加载训练好的模型"""
    print(f"正在加载模型: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("模型加载完成")
    return model


def prepare_features(gfs_data):
    """
    为GFS数据准备特征
    gfs_data: DataFrame，包含 timestamp, gfs_temp, gfs_wind_speed, gfs_wind_direction
    """
    df = gfs_data.copy()
    
    # 确保timestamp是datetime类型
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. 时间特征
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    # 2. 小时的正弦余弦编码
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # 3. 风向的正弦余弦编码
    df['wind_dir_sin'] = np.sin(2 * np.pi * df['gfs_wind_direction'] / 360)
    df['wind_dir_cos'] = np.cos(2 * np.pi * df['gfs_wind_direction'] / 360)
    
    # 4. 风速的三次方
    df['wind_speed_cube'] = df['gfs_wind_speed'] ** 3
    
    return df


def predict_power(model, gfs_file, output_file='predictions.csv'):
    """
    使用模型预测功率
    
    参数:
        model: 训练好的模型
        gfs_file: GFS气象数据文件路径
        output_file: 预测结果输出文件路径
    """
    print(f"\n正在读取GFS数据: {gfs_file}")
    
    # 读取GFS数据
    gfs_df = pd.read_csv(gfs_file)
    
    # 准备特征
    print("正在准备特征...")
    features_df = prepare_features(gfs_df)
    
    # 选择特征
    feature_columns = [
        'gfs_temp', 'gfs_wind_speed', 'gfs_wind_direction',
        'hour', 'day', 'month', 'day_of_week', 'day_of_year',
        'hour_sin', 'hour_cos', 'wind_dir_sin', 'wind_dir_cos',
        'wind_speed_cube'
    ]
    
    X = features_df[feature_columns]
    
    # 预测
    print("正在进行预测...")
    predictions = model.predict(X)
    
    # 保存结果
    result_df = pd.DataFrame({
        'timestamp': gfs_df['timestamp'],
        'predicted_power': predictions
    })
    
    result_df.to_csv(output_file, index=False)
    print(f"\n预测完成！")
    print(f"预测结果已保存至: {output_file}")
    print(f"共预测 {len(predictions)} 个时间点")
    
    # 显示前10条预测结果
    print("\n前10条预测结果:")
    print(result_df.head(10).to_string(index=False))
    
    return result_df


def predict_from_dict(model, gfs_data):
    """
    从字典数据进行预测（适用于API调用）
    
    参数:
        model: 训练好的模型
        gfs_data: 字典或DataFrame，包含GFS气象数据
        
    返回:
        预测的功率值
    """
    # 如果是字典，转换为DataFrame
    if isinstance(gfs_data, dict):
        gfs_df = pd.DataFrame([gfs_data])
    else:
        gfs_df = gfs_data.copy()
    
    # 准备特征
    features_df = prepare_features(gfs_df)
    
    # 选择特征
    feature_columns = [
        'gfs_temp', 'gfs_wind_speed', 'gfs_wind_direction',
        'hour', 'day', 'month', 'day_of_week', 'day_of_year',
        'hour_sin', 'hour_cos', 'wind_dir_sin', 'wind_dir_cos',
        'wind_speed_cube'
    ]
    
    X = features_df[feature_columns]
    
    # 预测
    prediction = model.predict(X)
    
    return prediction[0] if len(prediction) == 1 else prediction


def main():
    """主函数 - 示例用法"""
    print("=" * 60)
    print("风电站功率预测")
    print("=" * 60)
    
    # 加载模型
    try:
        model = load_model()
    except FileNotFoundError:
        print("\n错误: 找不到训练好的模型文件 'power_forecast_model.pkl'")
        print("请先运行 train_power_forecast.py 训练模型")
        sys.exit(1)
    
    # 预测GFS文件中的所有数据
    gfs_file = 'data_gfs_forecast.csv'
    output_file = 'predictions.csv'
    
    predict_power(model, gfs_file, output_file)
    
    print("\n" + "=" * 60)
    print("预测完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
