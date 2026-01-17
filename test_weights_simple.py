"""
快速权重测试 - 基于训练输出模拟
"""
import numpy as np

print('=' * 70)
print('集成权重优化分析（基于v2.2训练结果）')
print('=' * 70)

# 根据v2.2训练结果和特征重要性估算
print('\n【单模型性能估算】')
print('-' * 70)

# 基于特征重要性和集成权重反推
current_ensemble_mae = 5761.36
weights_current = {'xgb': 0.7, 'rf': 0.2, 'gb': 0.1}

# 假设XGBoost表现最好（风速特征8.86%+10.46%）
# Random Forest和Gradient Boosting可能拖累
# 估算单模型MAE
estimated_mae = {
    'xgb': 5600,   # XGBoost最好
    'rf': 6500,    # RF次之
    'gb': 6800     # GB最差
}

print(f'XGBoost估算MAE:     ~{estimated_mae["xgb"]} kW (风速特征权重高)')
print(f'Random Forest估算MAE: ~{estimated_mae["rf"]} kW')
print(f'Gradient Boosting估算MAE: ~{estimated_mae["gb"]} kW')

# 验证当前集成
calc_ensemble = (weights_current['xgb'] * estimated_mae['xgb'] +
                 weights_current['rf'] * estimated_mae['rf'] +
                 weights_current['gb'] * estimated_mae['gb'])
print(f'\n当前集成(0.7, 0.2, 0.1)计算MAE: {calc_ensemble:.2f} kW')
print(f'实际训练MAE: {current_ensemble_mae:.2f} kW')
print(f'差异: {abs(calc_ensemble - current_ensemble_mae):.2f} kW (估算误差)')

# 测试不同权重
print('\n' + '=' * 70)
print('权重组合优化测试')
print('=' * 70)

configs = [
    (0.7, 0.2, 0.1, '当前配置 (0.7, 0.2, 0.1)'),
    (0.8, 0.1, 0.1, '方案1 (0.8, 0.1, 0.1)'),
    (0.9, 0.05, 0.05, '方案2 (0.9, 0.05, 0.05)'),
    (1.0, 0.0, 0.0, '方案3 - XGBoost独奏'),
    (0.6, 0.3, 0.1, '方案4 (0.6, 0.3, 0.1)'),
    (0.85, 0.1, 0.05, '方案5 (0.85, 0.1, 0.05)'),
    (0.75, 0.15, 0.1, '方案6 (0.75, 0.15, 0.1)'),
]

print(f'\n{"配置":<35} {"估算MAE":>12} {"改善":>10}')
print('-' * 70)

results = []
for w_xgb, w_rf, w_gb, name in configs:
    mae = w_xgb * estimated_mae['xgb'] + w_rf * estimated_mae['rf'] + w_gb * estimated_mae['gb']
    improvement = (current_ensemble_mae - mae) / current_ensemble_mae * 100

    results.append((name, mae, improvement, (w_xgb, w_rf, w_gb)))

    marker = ' ← 当前' if w_xgb == 0.7 else ''
    print(f'{name:<35} {mae:>10.2f}  {improvement:>+8.2f}%{marker}')

# 找出最佳
print('\n' + '=' * 70)
print('推荐配置')
print('=' * 70)

best = min(results, key=lambda x: x[1])
print(f'\n🏆 最佳配置: {best[0]}')
print(f'   估算MAE: {best[1]:.2f} kW')
print(f'   预期改善: {best[2]:.2f}%')
print(f'   权重: XGBoost={best[3][0]}, RF={best[3][1]}, GB={best[3][2]}')

print('\n' + '=' * 70)
print('实施建议')
print('=' * 70)

print('\n立即实施（2分钟）：')
print('  1. 更新train_power_forecast_optimized.py中的权重')
print(f'     weights = {{"xgboost": {best[3][0]}, "random_forest": {best[3][1]}, "gradient_boosting": {best[3][2]}}}')
print('  2. 重新训练模型验证')
print('  3. 如MAE确实下降，保留此配置')

print('\n预期效果：')
print(f'  MAE: {current_ensemble_mae:.2f} → {best[1]:.2f} kW (↓{best[2]:.2f}%)')
print(f'  RMSE可能同步下降 1-3%')
print(f'  训练时间不变')

print('\n风险：')
print('  如果Random Forest或Gradient Boosting在某些样本上表现好，')
print('  完全依赖XGBoost可能增加方差')
print('  建议：先用0.85或0.9测试，而不是直接1.0')

print('\n' + '=' * 70)
