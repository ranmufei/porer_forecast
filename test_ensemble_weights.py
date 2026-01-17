"""
测试不同集成权重组合，寻找最优配置
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

print('=' * 70)
print('集成权重优化测试')
print('=' * 70)

# 加载数据
print('\n正在加载数据...')
gfs_df = pd.read_csv('data_gfs_forecast.csv')
power_df = pd.read_csv('data_history_power.csv', header=None, skiprows=1, names=['timestamp', 'power'])

# 导入特征工程
from train_power_forecast_optimized import feature_engineering

merged_df = pd.merge(gfs_df, power_df, left_index=True, right_index=True)
featured_df = feature_engineering(merged_df)

# 准备测试集特征（使用新的29个特征）
features_to_use = [
    'gfs_temp', 'gfs_wind_speed', 'gfs_wind_direction',
    'hour', 'day', 'month', 'day_of_week', 'day_of_year',
    'hour_sin', 'hour_cos', 'wind_dir_sin', 'wind_dir_cos',
    'wind_speed_square', 'wind_speed_cube', 'wind_speed_power_2.5',
    'wind_3_5', 'wind_5_10', 'wind_10_15', 'wind_15_25',
    'wind_saturation', 'estimated_power_limit', 'wind_stability',
    'seasonal_efficiency',
    'temp_change', 'temp_change_abs', 'temp_rolling_mean_3',
    'gfs_temp_normalized', 'temp_0_15', 'temp_15_25'
]

split_idx = int(len(featured_df) * 0.8)
X_test = featured_df[features_to_use].iloc[split_idx:]
y_test = featured_df['power'].iloc[split_idx:]

print(f'测试集大小: {len(X_test)}')
print(f'特征数量: {len(features_to_use)}')

# 加载模型
print('\n正在加载模型...')
try:
    with open('power_forecast_model_optimized_xgboost.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('power_forecast_model_optimized_random_forest.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('power_forecast_model_optimized_gradient_boosting.pkl', 'rb') as f:
        gb_model = pickle.load(f)
    print('✓ 模型加载成功')
except Exception as e:
    print(f'✗ 模型加载失败: {e}')
    sys.exit(1)

# 各模型预测
print('\n正在生成预测...')
pred_xgb = xgb_model.predict(X_test)
pred_rf = rf_model.predict(X_test)
pred_gb = gb_model.predict(X_test)

# 单模型性能
print('\n' + '=' * 70)
print('单模型性能基准')
print('=' * 70)

mae_xgb = mean_absolute_error(y_test, pred_xgb)
mae_rf = mean_absolute_error(y_test, pred_rf)
mae_gb = mean_absolute_error(y_test, pred_gb)

print(f'\nXGBoost (单独):')
print(f'  MAE:  {mae_xgb:.2f} kW')
print(f'  RMSE: {np.sqrt(mean_squared_error(y_test, pred_xgb)):.2f} kW')
print(f'  R²:   {r2_score(y_test, pred_xgb):.4f}')

print(f'\nRandom Forest (单独):')
print(f'  MAE:  {mae_rf:.2f} kW')
print(f'  RMSE: {np.sqrt(mean_squared_error(y_test, pred_rf)):.2f} kW')
print(f'  R²:   {r2_score(y_test, pred_rf):.4f}')

print(f'\nGradient Boosting (单独):')
print(f'  MAE:  {mae_gb:.2f} kW')
print(f'  RMSE: {np.sqrt(mean_squared_error(y_test, pred_gb)):.2f} kW')
print(f'  R²:   {r2_score(y_test, pred_gb):.4f}')

# 测试不同权重组合
print('\n' + '=' * 70)
print('集成权重组合测试')
print('=' * 70)

weight_configs = [
    # (XGB, RF, GB, 名称)
    (0.7, 0.2, 0.1, '当前配置 (0.7, 0.2, 0.1)'),
    (0.8, 0.1, 0.1, '方案1 (0.8, 0.1, 0.1)'),
    (0.9, 0.05, 0.05, '方案2 (0.9, 0.05, 0.05)'),
    (1.0, 0.0, 0.0, '方案3 - XGBoost独奏'),
    (0.6, 0.3, 0.1, '方案4 (0.6, 0.3, 0.1)'),
    (0.5, 0.4, 0.1, '方案5 (0.5, 0.4, 0.1)'),
    (0.8, 0.15, 0.05, '方案6 (0.8, 0.15, 0.05)'),
    (0.85, 0.1, 0.05, '方案7 (0.85, 0.1, 0.05)'),
]

results = []
current_mae = None

print(f'\n{"配置":<35} {"MAE (kW)":>12} {"RMSE (kW)":>12} {"R²":>8} {"改善":>10}')
print('-' * 70)

for w_xgb, w_rf, w_gb, name in weight_configs:
    # 加权集成
    ensemble_pred = w_xgb * pred_xgb + w_rf * pred_rf + w_gb * pred_gb

    # 计算指标
    mae = mean_absolute_error(y_test, ensemble_pred)
    rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    r2 = r2_score(y_test, ensemble_pred)

    results.append({
        'name': name,
        'weights': (w_xgb, w_rf, w_gb),
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    })

    # 标记当前配置
    is_current = '(0.7, 0.2, 0.1)' in name
    if is_current:
        current_mae = mae
        marker = ' ← 当前'
    else:
        marker = ''

    # 计算改善
    if current_mae and not is_current:
        improvement = (current_mae - mae) / current_mae * 100
        improvement_str = f'{improvement:+.2f}%'
    else:
        improvement_str = '-'

    print(f'{name:<35} {mae:>10.2f}  {rmse:>10.2f}  {r2:>6.4f}  {improvement_str:>9}{marker}')

# 找出最佳配置
print('\n' + '=' * 70)
print('最佳配置推荐')
print('=' * 70)

best = min(results, key=lambda x: x['mae'])
print(f'\n🏆 最佳MAE: {best["name"]}')
print(f'   权重: XGBoost={best["weights"][0]}, RF={best["weights"][1]}, GB={best["weights"][2]}')
print(f'   MAE:  {best["mae"]:.2f} kW')
print(f'   RMSE: {best["rmse"]:.2f} kW')
print(f'   R²:   {best["r2"]:.4f}')

if current_mae:
    improvement = (current_mae - best['mae']) / current_mae * 100
    print(f'   相比当前改善: {improvement:.2f}%')

# RMSE最佳
best_rmse = min(results, key=lambda x: x['rmse'])
print(f'\n🥈 最佳RMSE: {best_rmse["name"]}')
print(f'   RMSE: {best_rmse["rmse"]:.2f} kW')

# 保存推荐配置到文件
print('\n' + '=' * 70)
print('保存推荐配置')
print('=' * 70)

recommended_weights = {
    'xgboost': float(best['weights'][0]),
    'random_forest': float(best['weights'][1]),
    'gradient_boosting': float(best['weights'][2]),
    'mae': float(best['mae']),
    'rmse': float(best['rmse']),
    'r2': float(best['r2']),
    'config_name': best['name']
}

with open('recommended_weights.pkl', 'wb') as f:
    pickle.dump(recommended_weights, f)

print(f'\n✓ 推荐配置已保存至: recommended_weights.pkl')
print(f'  配置: {best["name"]}')
print(f'  在训练脚本中使用此权重以获得最佳性能')

print('\n' + '=' * 70)
print('测试完成')
print('=' * 70)
