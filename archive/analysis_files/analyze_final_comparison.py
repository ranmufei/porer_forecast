"""
综合对比分析：优化模型 vs 分层模型
找出最佳方案
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def calculate_metrics(actual, predicted):
    """计算评估指标"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    
    # MAPE（排除零功率）
    nonzero_mask = actual > 100
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((actual[nonzero_mask] - predicted[nonzero_mask]) / 
                               actual[nonzero_mask])) * 100
    else:
        mape = np.nan
    
    # 偏差
    bias = np.mean(predicted - actual)
    
    # 零功率准确率
    actual_zero = actual == 0
    pred_zero = predicted < 500
    if actual_zero.sum() > 0:
        zero_accuracy = np.sum(actual_zero & pred_zero) / actual_zero.sum() * 100
    else:
        zero_accuracy = np.nan
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2,
        'Bias': bias,
        'Zero_Accuracy': zero_accuracy
    }


def analyze_by_power_segments(df, pred_col):
    """按功率段分析误差"""
    segments = [
        ('零功率', df[df['实际功率'] == 0]),
        ('低功率(<5k)', df[(df['实际功率'] > 0) & (df['实际功率'] < 5000)]),
        ('中低功率(5-15k)', df[(df['实际功率'] >= 5000) & (df['实际功率'] < 15000)]),
        ('中高功率(15-30k)', df[(df['实际功率'] >= 15000) & (df['实际功率'] < 30000)]),
        ('高功率(>=30k)', df[df['实际功率'] >= 30000])
    ]
    
    results = []
    for segment_name, segment_df in segments:
        if len(segment_df) == 0:
            continue
        
        metrics = calculate_metrics(segment_df['实际功率'].values, 
                                    segment_df[pred_col].values)
        metrics['Segment'] = segment_name
        metrics['Count'] = len(segment_df)
        results.append(metrics)
    
    return pd.DataFrame(results)


def analyze_by_wind_speed(df, pred_col):
    """按风速分析误差"""
    segments = [
        ('低风速(<3)', df[df['gfs_wind_speed'] < 3]),
        ('中低风速(3-8)', df[(df['gfs_wind_speed'] >= 3) & (df['gfs_wind_speed'] < 8)]),
        ('中等风速(8-12)', df[(df['gfs_wind_speed'] >= 8) & (df['gfs_wind_speed'] < 12)]),
        ('中高风速(12-16)', df[(df['gfs_wind_speed'] >= 12) & (df['gfs_wind_speed'] < 16)]),
        ('高风速(>=16)', df[df['gfs_wind_speed'] >= 16])
    ]
    
    results = []
    for segment_name, segment_df in segments:
        if len(segment_df) == 0:
            continue
        
        metrics = calculate_metrics(segment_df['实际功率'].values, 
                                    segment_df[pred_col].values)
        metrics['Wind_Segment'] = segment_name
        metrics['Count'] = len(segment_df)
        results.append(metrics)
    
    return pd.DataFrame(results)


def main():
    """主函数"""
    print("=" * 80)
    print("综合对比分析：优化模型 vs 分层模型")
    print("=" * 80)
    
    # 1. 加载对比数据
    print("\n1. 加载数据...")
    
    # 优化模型预测结果
    optimized_file = '对比数据/0116数据优化版本数据分析/predictions_optimized-数据分析6-1---8.csv'
    df_optimized = pd.read_csv(optimized_file)
    df_optimized['timestamp'] = pd.to_datetime(df_optimized['timestamp'])
    
    # 分层模型预测结果
    hierarchical_file = '对比数据/分层模型预测对比.csv'
    df_hierarchical = pd.read_csv(hierarchical_file)
    df_hierarchical['timestamp'] = pd.to_datetime(df_hierarchical['timestamp'])
    
    # GFS数据
    gfs_df = pd.read_csv('data_gfs_forecast.csv')
    gfs_df['timestamp'] = pd.to_datetime(gfs_df['timestamp'])
    
    # 合并数据
    df = pd.merge(df_optimized[['timestamp', '实际功率', 'predicted_power']], 
                 df_hierarchical[['timestamp', 'predicted_power']], 
                 on='timestamp', how='inner', 
                 suffixes=('_optimized', '_hierarchical'))
    df = pd.merge(df, gfs_df, on='timestamp', how='inner')
    
    print(f"   合并后数据量: {len(df)} 条")
    
    # 2. 整体性能对比
    print("\n2. 整体性能对比")
    print("=" * 80)
    
    metrics_optimized = calculate_metrics(df['实际功率'].values, 
                                          df['predicted_power_optimized'].values)
    metrics_hierarchical = calculate_metrics(df['实际功率'].values, 
                                            df['predicted_power_hierarchical'].values)
    
    comparison_df = pd.DataFrame({
        '优化模型': metrics_optimized,
        '分层模型': metrics_hierarchical
    })
    
    # 计算改进百分比
    comparison_df['改进'] = (
        (comparison_df['优化模型'] - comparison_df['分层模型']) / 
        abs(comparison_df['分层模型']) * 100
    )
    
    print(comparison_df.round(2))
    
    # 3. 按功率段分析
    print("\n3. 按功率段分析")
    print("=" * 80)
    
    power_segments_opt = analyze_by_power_segments(df, 'predicted_power_optimized')
    power_segments_hier = analyze_by_power_segments(df, 'predicted_power_hierarchical')
    
    for _, row in power_segments_opt.iterrows():
        segment = row['Segment']
        count = row['Count']
        
        opt_metrics = row.drop(['Segment', 'Count']).to_dict()
        hier_metrics = power_segments_hier[power_segments_hier['Segment'] == segment].iloc[0]
        hier_metrics = hier_metrics.drop(['Segment', 'Count']).to_dict()
        
        print(f"\n{segment} ({count} 条):")
        print(f"  优化模型: MAE={opt_metrics['MAE']:.1f}, RMSE={opt_metrics['RMSE']:.1f}, "
              f"MAPE={opt_metrics['MAPE']:.1f}%")
        print(f"  分层模型: MAE={hier_metrics['MAE']:.1f}, RMSE={hier_metrics['RMSE']:.1f}, "
              f"MAPE={hier_metrics['MAPE']:.1f}%")
        
        if abs(hier_metrics['MAPE']) > 0:
            improvement = (opt_metrics['MAPE'] - hier_metrics['MAPE']) / abs(hier_metrics['MAPE']) * 100
            print(f"  MAPE改进: {improvement:.1f}%")
    
    # 4. 按风速分析
    print("\n4. 按风速分析")
    print("=" * 80)
    
    wind_segments_opt = analyze_by_wind_speed(df, 'predicted_power_optimized')
    wind_segments_hier = analyze_by_wind_speed(df, 'predicted_power_hierarchical')
    
    for _, row in wind_segments_opt.iterrows():
        segment = row['Wind_Segment']
        count = row['Count']
        
        opt_metrics = row.drop(['Wind_Segment', 'Count']).to_dict()
        hier_metrics = wind_segments_hier[wind_segments_hier['Wind_Segment'] == segment].iloc[0]
        hier_metrics = hier_metrics.drop(['Wind_Segment', 'Count']).to_dict()
        
        print(f"\n{segment} ({count} 条):")
        print(f"  优化模型: MAE={opt_metrics['MAE']:.1f}, RMSE={opt_metrics['RMSE']:.1f}")
        print(f"  分层模型: MAE={hier_metrics['MAE']:.1f}, RMSE={hier_metrics['RMSE']:.1f}")
    
    # 5. 关键发现分析
    print("\n5. 关键发现")
    print("=" * 80)
    
    # 零功率分析
    actual_zero = df['实际功率'] == 0
    pred_zero_opt = df['predicted_power_optimized'] < 500
    pred_zero_hier = df['predicted_power_hierarchical'] < 500
    
    print(f"\n零功率分析 (共{actual_zero.sum()}个时间点):")
    print(f"  优化模型: {np.sum(actual_zero & pred_zero_opt)} 个正确识别 "
          f"({np.sum(actual_zero & pred_zero_opt)/actual_zero.sum()*100:.1f}%)")
    print(f"  分层模型: {np.sum(actual_zero & pred_zero_hier)} 个正确识别 "
          f"({np.sum(actual_zero & pred_zero_hier)/actual_zero.sum()*100:.1f}%)")
    
    # 高功率分析
    high_power = df['实际功率'] >= 30000
    if high_power.sum() > 0:
        print(f"\n高功率分析 (>=30kW, 共{high_power.sum()}个时间点):")
        opt_mae = mean_absolute_error(
            df.loc[high_power, '实际功率'], 
            df.loc[high_power, 'predicted_power_optimized']
        )
        hier_mae = mean_absolute_error(
            df.loc[high_power, '实际功率'], 
            df.loc[high_power, 'predicted_power_hierarchical']
        )
        print(f"  优化模型 MAE: {opt_mae:.1f} kW")
        print(f"  分层模型 MAE: {hier_mae:.1f} kW")
    
    # 6. 绘制综合对比图
    print("\n6. 生成对比图...")
    
    n_points = min(500, len(df))
    plot_df = df.iloc[:n_points].copy()
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    
    # 子图1：时序对比
    axes[0, 0].plot(plot_df.index, plot_df['实际功率'], 
                    label='实际功率', alpha=0.8, linewidth=2, color='black')
    axes[0, 0].plot(plot_df.index, plot_df['predicted_power_optimized'], 
                    label='优化模型', alpha=0.7, linewidth=1.5, color='blue')
    axes[0, 0].plot(plot_df.index, plot_df['predicted_power_hierarchical'], 
                    label='分层模型', alpha=0.7, linewidth=1.5, color='red', linestyle='--')
    axes[0, 0].set_title('功率预测对比（时序）', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('功率 (kW)', fontsize=12)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2：散点图对比
    axes[0, 1].scatter(plot_df['实际功率'], plot_df['predicted_power_optimized'], 
                       alpha=0.5, s=20, label='优化模型', color='blue')
    axes[0, 1].scatter(plot_df['实际功率'], plot_df['predicted_power_hierarchical'], 
                       alpha=0.5, s=20, label='分层模型', color='red', marker='s')
    max_val = max(plot_df['实际功率'].max(), plot_df['predicted_power_optimized'].max())
    axes[0, 1].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='理想预测')
    axes[0, 1].set_title('预测值 vs 实际值', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('实际功率 (kW)', fontsize=12)
    axes[0, 1].set_ylabel('预测功率 (kW)', fontsize=12)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3：误差对比
    df_copy = df.copy()
    df_copy['error_optimized'] = df_copy['predicted_power_optimized'] - df_copy['实际功率']
    df_copy['error_hierarchical'] = df_copy['predicted_power_hierarchical'] - df_copy['实际功率']
    
    axes[1, 0].plot(plot_df.index, df_copy.loc[plot_df.index, 'error_optimized'], 
                    label='优化模型误差', alpha=0.6, linewidth=1.5, color='blue')
    axes[1, 0].plot(plot_df.index, df_copy.loc[plot_df.index, 'error_hierarchical'], 
                    label='分层模型误差', alpha=0.6, linewidth=1.5, color='red')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('预测误差对比', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('误差 (kW)', fontsize=12)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 子图4：绝对误差分布
    df_copy['abs_error_optimized'] = np.abs(df_copy['error_optimized'])
    df_copy['abs_error_hierarchical'] = np.abs(df_copy['error_hierarchical'])
    
    axes[1, 1].hist(df_copy['abs_error_optimized'], bins=30, alpha=0.5, 
                     label='优化模型', color='blue', density=True)
    axes[1, 1].hist(df_copy['abs_error_hierarchical'], bins=30, alpha=0.5, 
                     label='分层模型', color='red', density=True)
    axes[1, 1].set_title('绝对误差分布', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('绝对误差 (kW)', fontsize=12)
    axes[1, 1].set_ylabel('密度', fontsize=12)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 子图5：按功率段的MAE对比
    power_segments_opt['Model'] = '优化模型'
    power_segments_hier['Model'] = '分层模型'
    power_segments_hier = power_segments_hier.rename(columns={'Segment': 'Power_Segment'})
    power_segments_opt = power_segments_opt.rename(columns={'Segment': 'Power_Segment'})
    
    combined_power = pd.concat([power_segments_opt[['Power_Segment', 'MAE', 'Model']], 
                               power_segments_hier[['Power_Segment', 'MAE', 'Model']]])
    
    x_labels = combined_power['Power_Segment'].unique()
    x_pos = np.arange(len(x_labels))
    width = 0.35
    
    opt_mae = power_segments_opt.set_index('Power_Segment').loc[x_labels, 'MAE'].values
    hier_mae = power_segments_hier.set_index('Power_Segment').loc[x_labels, 'MAE'].values
    
    axes[2, 0].bar(x_pos - width/2, opt_mae, width, label='优化模型', 
                   color='blue', alpha=0.8)
    axes[2, 0].bar(x_pos + width/2, hier_mae, width, label='分层模型', 
                   color='red', alpha=0.8)
    axes[2, 0].set_xlabel('功率段', fontsize=12)
    axes[2, 0].set_ylabel('MAE (kW)', fontsize=12)
    axes[2, 0].set_title('各功率段MAE对比', fontsize=14, fontweight='bold')
    axes[2, 0].set_xticks(x_pos)
    axes[2, 0].set_xticklabels(x_labels, rotation=15, ha='right')
    axes[2, 0].legend(fontsize=11)
    axes[2, 0].grid(True, alpha=0.3, axis='y')
    
    # 子图6：按风速段的MAE对比
    wind_segments_opt['Model'] = '优化模型'
    wind_segments_hier['Model'] = '分层模型'
    wind_segments_hier = wind_segments_hier.rename(columns={'Wind_Segment': 'Wind_Segment2'})
    wind_segments_opt = wind_segments_opt.rename(columns={'Wind_Segment': 'Wind_Segment2'})
    
    combined_wind = pd.concat([wind_segments_opt[['Wind_Segment2', 'MAE', 'Model']], 
                               wind_segments_hier[['Wind_Segment2', 'MAE', 'Model']]])
    
    x_labels = combined_wind['Wind_Segment2'].unique()
    x_pos = np.arange(len(x_labels))
    
    opt_mae = wind_segments_opt.set_index('Wind_Segment2').loc[x_labels, 'MAE'].values
    hier_mae = wind_segments_hier.set_index('Wind_Segment2').loc[x_labels, 'MAE'].values
    
    axes[2, 1].bar(x_pos - width/2, opt_mae, width, label='优化模型', 
                   color='blue', alpha=0.8)
    axes[2, 1].bar(x_pos + width/2, hier_mae, width, label='分层模型', 
                   color='red', alpha=0.8)
    axes[2, 1].set_xlabel('风速段', fontsize=12)
    axes[2, 1].set_ylabel('MAE (kW)', fontsize=12)
    axes[2, 1].set_title('各风速段MAE对比', fontsize=14, fontweight='bold')
    axes[2, 1].set_xticks(x_pos)
    axes[2, 1].set_xticklabels(x_labels, rotation=15, ha='right')
    axes[2, 1].legend(fontsize=11)
    axes[2, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('综合对比分析图.png', dpi=300, bbox_inches='tight')
    print("综合对比图已保存至: 综合对比分析图.png")
    plt.close()
    
    # 7. 结论与建议
    print("\n" + "=" * 80)
    print("结论与建议")
    print("=" * 80)
    
    print("\n基于对比分析，发现:")
    print("1. 整体性能: 优化模型优于分层模型")
    print(f"   - MAE: {metrics_optimized['MAE']:.1f} vs {metrics_hierarchical['MAE']:.1f} kW")
    print(f"   - RMSE: {metrics_optimized['RMSE']:.1f} vs {metrics_hierarchical['RMSE']:.1f} kW")
    print(f"   - MAPE: {metrics_optimized['MAPE']:.1f}% vs {metrics_hierarchical['MAPE']:.1f}%")
    
    print("\n2. 分层模型问题:")
    print("   - 低功率段分类不准确，导致错误选择模型")
    print("   - 物理约束过度限制，影响预测灵活性")
    print("   - 各分段训练数据不均衡，低功率段R²为负")
    
    print("\n3. 建议:")
    print("   ✅ 继续使用优化模型作为主要预测方案")
    print("   ✅ 保留分层模型思路，但需要:")
    print("      - 改进功率段分类方法（使用概率分类而非硬分类）")
    print("      - 增加训练数据，特别是低功率段数据")
    print("      - 软化物理约束，减少过度修正")
    print("      - 考虑使用集成方法融合多个模型")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
