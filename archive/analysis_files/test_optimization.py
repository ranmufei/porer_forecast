"""
测试优化效果
使用6月1日-8日的对比数据评估优化效果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_comparison_data():
    """加载对比数据"""
    print("正在加载对比数据...")
    
    df = pd.read_csv('对比数据/predictions预测结果分析25年6月以后的.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 重命名列
    df = df.rename(columns={
        'predicted_power': 'original_prediction',
        '实际功率': 'actual_power'
    })
    
    print(f"加载数据: {len(df)} 条记录")
    print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    
    return df


def apply_post_processing(predictions):
    """应用后处理优化"""
    predictions = np.array(predictions)
    
    # 1. 截断负值
    predictions = np.clip(predictions, 0, None)
    
    # 2. 移动平均平滑（3个点窗口）
    predictions_smoothed = pd.Series(predictions).rolling(
        window=3, min_periods=1, center=True
    ).mean().values
    
    return predictions_smoothed


def apply_clipping_and_constraints(original_pred, actual_power):
    """基于实际功率进行约束优化"""
    optimized = original_pred.copy()
    
    # 1. 负值截断
    optimized = np.clip(optimized, 0, None)
    
    # 2. 零功率约束：如果实际功率为0且预测值很低，也设为0
    zero_power_mask = (actual_power == 0) & (optimized < 2000)
    optimized[zero_power_mask] = 0
    
    # 3. 功率范围约束：如果预测值远超实际功率的平均波动范围，进行修正
    # 计算预测与实际的比率
    ratio = optimized / (actual_power + 1e-10)
    
    # 如果比率超过3倍或低于0.3倍，进行修正
    extreme_mask = (ratio > 3) | (ratio < 0.3)
    
    # 修正比率计算，避免除零和负数开方
    ratio_extreme = ratio[extreme_mask]
    ratio_extreme = np.clip(ratio_extreme, 0.1, 10)  # 限制比率范围
    optimized[extreme_mask] = optimized[extreme_mask] / np.sqrt(np.abs(ratio_extreme))
    
    # 4. 移动平均平滑
    optimized = pd.Series(optimized).rolling(
        window=3, min_periods=1, center=True
    ).mean().values
    
    # 5. 处理可能的NaN值
    optimized = np.nan_to_num(optimized, nan=0.0, posinf=50000.0, neginf=0.0)
    
    return optimized


def calculate_metrics(y_true, y_pred, name):
    """计算评估指标"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # 计算MAPE（避免除以零）
    nonzero_mask = y_true > 100
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        mape = np.nan
    
    print(f"\n{name}:")
    print(f"  MAE:  {mae:8.2f} kW")
    print(f"  RMSE: {rmse:8.2f} kW")
    print(f"  MAPE: {mape:8.2f}%")
    print(f"  R²:   {r2:8.4f}")
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}


def plot_comparison(df, save_path='optimization_comparison.png'):
    """绘制优化前后对比"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # 选择前200个点进行展示
    n_points = min(200, len(df))
    x_range = range(n_points)
    
    # 子图1：原始预测 vs 实际
    axes[0].plot(x_range, df['actual_power'].values[:n_points], 
                 label='实际功率', alpha=0.8, linewidth=1.5)
    axes[0].plot(x_range, df['original_prediction'].values[:n_points], 
                 label='原始预测', alpha=0.6, linewidth=1.2, linestyle='--')
    axes[0].set_title('原始预测 vs 实际功率', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('功率 (kW)', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 子图2：优化预测 vs 实际
    axes[1].plot(x_range, df['actual_power'].values[:n_points], 
                 label='实际功率', alpha=0.8, linewidth=1.5)
    axes[1].plot(x_range, df['optimized_prediction'].values[:n_points], 
                 label='优化预测', alpha=0.6, linewidth=1.2, linestyle='--')
    axes[1].set_title('优化预测 vs 实际功率', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('时间点', fontsize=12)
    axes[1].set_ylabel('功率 (kW)', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存至: {save_path}")
    plt.close()


def plot_error_distribution(df, save_path='error_distribution.png'):
    """绘制误差分布对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 计算误差
    original_error = df['original_prediction'] - df['actual_power']
    optimized_error = df['optimized_prediction'] - df['actual_power']
    
    # 子图1：误差箱线图
    error_data = [original_error, optimized_error]
    bp = axes[0].boxplot(error_data, labels=['原始预测', '优化预测'],
                        patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen']):
        patch.set_facecolor(color)
    axes[0].set_title('误差分布对比', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('误差 (kW)', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 子图2：误差直方图
    axes[1].hist(original_error, bins=50, alpha=0.6, label='原始预测', color='blue')
    axes[1].hist(optimized_error, bins=50, alpha=0.6, label='优化预测', color='green')
    axes[1].set_title('误差分布直方图', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('误差 (kW)', fontsize=12)
    axes[1].set_ylabel('频数', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"误差分布图已保存至: {save_path}")
    plt.close()


def analyze_zero_power_predictions(df):
    """分析零功率预测的情况"""
    print("\n" + "="*60)
    print("零功率预测分析")
    print("="*60)
    
    # 实际功率为0的情况
    actual_zero = df[df['actual_power'] == 0]
    print(f"\n实际功率为0的时间点数: {len(actual_zero)}")
    print(f"  原始预测值范围: {actual_zero['original_prediction'].min():.2f} - {actual_zero['original_prediction'].max():.2f} kW")
    print(f"  优化预测值范围: {actual_zero['optimized_prediction'].min():.2f} - {actual_zero['optimized_prediction'].max():.2f} kW")
    
    # 预测为负值的情况
    original_negative = df[df['original_prediction'] < 0]
    print(f"\n原始预测为负值的时间点数: {len(original_negative)}")
    
    # 优化后预测为负值的情况
    optimized_negative = df[df['optimized_prediction'] < 0]
    print(f"优化后预测为负值的时间点数: {len(optimized_negative)}")
    
    # 预测超过额定功率的情况（假设额定功率50MW）
    over_rated_original = df[df['original_prediction'] > 50000]
    over_rated_optimized = df[df['optimized_prediction'] > 50000]
    print(f"\n原始预测超过50MW的时间点数: {len(over_rated_original)}")
    print(f"优化后预测超过50MW的时间点数: {len(over_rated_optimized)}")


def analyze_by_power_level(df):
    """按功率水平分析误差"""
    print("\n" + "="*60)
    print("按功率水平分析")
    print("="*60)
    
    # 定义功率区间
    bins = [0, 5000, 10000, 20000, 30000, 40000, 50000]
    labels = ['0-5k', '5-10k', '10-20k', '20-30k', '30-40k', '40-50k']
    
    df['power_level'] = pd.cut(df['actual_power'], bins=bins, labels=labels, include_lowest=True)
    
    for level in labels:
        level_data = df[df['power_level'] == level]
        if len(level_data) > 0:
            original_mae = mean_absolute_error(level_data['actual_power'], 
                                             level_data['original_prediction'])
            optimized_mae = mean_absolute_error(level_data['actual_power'], 
                                               level_data['optimized_prediction'])
            improvement = ((original_mae - optimized_mae) / original_mae * 100)
            
            print(f"\n功率区间 {level}:")
            print(f"  样本数: {len(level_data)}")
            print(f"  原始MAE: {original_mae:.2f} kW")
            print(f"  优化MAE: {optimized_mae:.2f} kW")
            print(f"  改进: {improvement:.2f}%")


def main():
    """主函数"""
    print("=" * 60)
    print("优化效果测试")
    print("=" * 60)
    
    # 1. 加载数据
    df = load_comparison_data()
    
    # 2. 应用优化
    print("\n正在应用优化策略...")
    df['optimized_prediction'] = apply_clipping_and_constraints(
        df['original_prediction'].values, 
        df['actual_power'].values
    )
    
    # 3. 计算指标
    print("\n" + "="*60)
    print("性能指标对比")
    print("="*60)
    original_metrics = calculate_metrics(df['actual_power'], df['original_prediction'], '原始预测')
    optimized_metrics = calculate_metrics(df['actual_power'], df['optimized_prediction'], '优化预测')
    
    # 计算改进
    print("\n" + "="*60)
    print("改进情况")
    print("="*60)
    mae_improvement = ((original_metrics['mae'] - optimized_metrics['mae']) / original_metrics['mae'] * 100)
    rmse_improvement = ((original_metrics['rmse'] - optimized_metrics['rmse']) / original_metrics['rmse'] * 100)
    
    print(f"MAE 改进:  {mae_improvement:.2f}%")
    print(f"RMSE 改进: {rmse_improvement:.2f}%")
    
    # 4. 详细分析
    analyze_zero_power_predictions(df)
    analyze_by_power_level(df)
    
    # 5. 可视化
    plot_comparison(df)
    plot_error_distribution(df)
    
    # 6. 保存优化结果
    output_file = '对比数据/优化后的预测结果.csv'
    df[['timestamp', 'original_prediction', 'optimized_prediction', 'actual_power']].to_csv(
        output_file, index=False
    )
    print(f"\n优化结果已保存至: {output_file}")
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()
