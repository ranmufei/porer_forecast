"""
风电站功率预测API服务
提供HTTP接口进行功率预测
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量存储模型
model = None


def load_model():
    """加载模型"""
    global model
    try:
        with open('power_forecast_model.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.info("模型加载成功")
        return True
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return False


def prepare_features(gfs_data):
    """准备特征"""
    df = gfs_data.copy()
    
    # 确保timestamp是datetime类型
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 时间特征
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    # 小时的正弦余弦编码
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # 风向的正弦余弦编码
    df['wind_dir_sin'] = np.sin(2 * np.pi * df['gfs_wind_direction'] / 360)
    df['wind_dir_cos'] = np.cos(2 * np.pi * df['gfs_wind_direction'] / 360)
    
    # 风速的三次方
    df['wind_speed_cube'] = df['gfs_wind_speed'] ** 3
    
    return df


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    单点预测接口
    
    请求体示例:
    {
        "timestamp": "2024-08-01 12:00:00",
        "gfs_temp": 15.5,
        "gfs_wind_speed": 8.3,
        "gfs_wind_direction": 180
    }
    
    返回:
    {
        "predicted_power": 35000.5,
        "timestamp": "2024-08-01 12:00:00"
    }
    """
    try:
        if model is None:
            return jsonify({'error': '模型未加载'}), 500
        
        # 获取请求数据
        data = request.get_json()
        
        # 验证必需字段
        required_fields = ['timestamp', 'gfs_temp', 'gfs_wind_speed', 'gfs_wind_direction']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'缺少必需字段: {field}'}), 400
        
        # 转换为DataFrame
        gfs_df = pd.DataFrame([data])
        
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
        
        return jsonify({
            'predicted_power': float(prediction[0]),
            'timestamp': data['timestamp']
        })
    
    except Exception as e:
        logger.error(f"预测失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    批量预测接口
    
    请求体示例:
    {
        "data": [
            {
                "timestamp": "2024-08-01 12:00:00",
                "gfs_temp": 15.5,
                "gfs_wind_speed": 8.3,
                "gfs_wind_direction": 180
            },
            ...
        ]
    }
    
    返回:
    {
        "predictions": [
            {"predicted_power": 35000.5, "timestamp": "2024-08-01 12:00:00"},
            ...
        ]
    }
    """
    try:
        if model is None:
            return jsonify({'error': '模型未加载'}), 500
        
        # 获取请求数据
        data = request.get_json()
        
        if 'data' not in data:
            return jsonify({'error': '缺少data字段'}), 400
        
        gfs_list = data['data']
        
        # 转换为DataFrame
        gfs_df = pd.DataFrame(gfs_list)
        
        # 验证必需字段
        required_fields = ['timestamp', 'gfs_temp', 'gfs_wind_speed', 'gfs_wind_direction']
        for field in required_fields:
            if field not in gfs_df.columns:
                return jsonify({'error': f'缺少必需字段: {field}'}), 400
        
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
        predictions = model.predict(X)
        
        # 构建返回结果
        results = []
        for i, row in enumerate(gfs_list):
            results.append({
                'predicted_power': float(predictions[i]),
                'timestamp': row['timestamp']
            })
        
        return jsonify({
            'predictions': results,
            'count': len(results)
        })
    
    except Exception as e:
        logger.error(f"批量预测失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/info', methods=['GET'])
def info():
    """API信息"""
    return jsonify({
        'name': '风电站功率预测API',
        'version': '1.0.0',
        'description': '基于GFS气象数据的风电站发电功率预测服务',
        'endpoints': {
            'POST /predict': '单点功率预测',
            'POST /batch_predict': '批量功率预测',
            'GET /health': '健康检查',
            'GET /info': 'API信息'
        },
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    # 启动前加载模型
    if not load_model():
        print("警告: 模型加载失败，请确保已训练模型")
    
    # 启动Flask服务
    print("=" * 60)
    print("风电站功率预测API服务")
    print("=" * 60)
    print("服务地址: http://localhost:5000")
    print("API文档: http://localhost:5000/info")
    print("健康检查: http://localhost:5000/health")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
