"""
功率段误差分析脚本
为分层模型训练做准备
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """加载数据"""
    print("正在加载数据...")
    
    df = pd.read_csv('对比数据/0116数据优化版本数据分析/predictions_optimized-数据分析6-1---8.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 计算误差
    df['error'] = df['predicted_power'] - df['实际功率']
    df['abs_error'] = np.abs(df['error'])
    df['mape'] = np.abs(df['error'] / (df['实际功率'] + 1e-10)) * 100
    
    print(f"加载数据: {len(df)} 条记录")
    print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    
    return df


def define_power_segments(df):
    """定义功率段"""
    df = df.copy()
    
    # 定义功率段
    conditions = [
        (df['实际功率'] < 5000),
        (df['实际功率'] >= 5000) & (df['实际功率'] < 15000),
        (df['实际功率'] >= 15000) & (df['实际功率'] < 30000),
        (df['实际功率'] >= 30000)
    ]
    
    labels = ['低功率段(<5k)', '中低功率段(5-15k)', '中高功率段(15-30k)', '高功率段(>30k)']
    
    df['power_segment'] = np.select(conditions, labels, default='未知')
    
    return df


def analyze_by_segment(df):
    """按功率段分析"""
    print("\n" + "="*60)
    print("按功率段误差分析")
    print("="*60)
    
    segments = ['低功率段(<5k)', '中低功率段(5-15k)', '中高功率段(15-30k)', '高功率段(>30k)']
    
    results = []
    
    for segment in segments:
        segment_df = df[df['power_segment'] == segment]
        
        if len(segment_df) == 0:
            continue
        
        # 计算指标
        mae = mean_absolute_error(segment_df['实际功率'], segment_df['predicted_power'])
        rmse = np.sqrt(mean_squared_error(segment_df['实际功率'], segment_df['predicted_power']))
        r2 = r2_score(segment_df['实际功率'], segment_df['predicted_power'])
        
        # 过滤零功率计算MAPE
        nonzero_mask = segment_df['实际功率'] > 100
        if nonzero_mask.sum() > 0:
            mape = np.mean(np.abs((segment_df.loc[nonzero_mask, '实际功率'] - 
                                   segment_df.loc[nonzero_mask, 'predicted_power']) / 
                                   segment_df.loc[nonzero_mask, '实际功率'])) * 100
        else:
            mape = np.nan
        
        # 统计信息
        actual_mean = segment_df['实际功率'].mean()
        predicted_mean = segment_df['predicted_power'].mean()
        bias = predicted_mean - actual_mean
        
        results.append({
            'segment': segment,
            'count': len(segment_df),
            'actual_mean': actual_mean,
            'predicted_mean': predicted_mean,
            'bias': bias,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        })
        
        print(f"\n{segment}:")
        print(f"  样本数: {len(segment_df)}")
        print(f"  实际均值: {actual_mean:.2f} kW")
        print(f"  预测均值: {predicted_mean:.2f} kW")
        print(f"  偏差: {bias:.2f} kW ({bias/actual_mean*100:.2f}%)")
        print(f"  MAE:  {mae:.2f} kW")
        print(f"  RMSE: {rmse:.2f} kW")
        print(f"  MAPE: {mape:.2f}%" if not np.isnan(mape) else f"  MAPE: N/A")
        print(f"  R²:   {r2:.4f}")
    
    return pd.DataFrame(results)


def analyze_by_time(df):
    """按时段分析"""
    print("\n" + "="*60)
    print("按时段误差分析")
    print("="*60)
    
    df['hour'] = df['timestamp'].dt.hour
    
    hourly_stats = df.groupby('hour').agg({
        '实际功率': ['mean', 'std'],
        'predicted_power': ['mean', 'std'],
        'abs_error': ['mean', 'std']
    }).round(2)
    
    print(hourly_stats)
    
    return hourly_stats


def analyze_zero_power_errors(df):
    """分析零功率预测误差"""
    print("\n" + "="*60)
    print("零功率预测误差分析")
    print("="*60)
    
    # 实际功率为0的情况
    actual_zero = df[df['实际功率'] == 0]
    
    print(f"\n实际功率为0的时间点数: {len(actual_zero)}")
    if len(actual_zero) > 0:
        print(f"  预测值范围: {actual_zero['predicted_power'].min():.2f} - {actual_zero['predicted_power'].max():.2f} kW")
        print(f"  预测均值: {actual_zero['predicted_power'].mean():.2f} kW")
        print(f"  预测中位数: {actual_zero['predicted_power'].median():.2f} kW")
        print(f"  预测标准差: {actual_zero['predicted_power'].std():.2f} kW")
        
        # 统计预测值<1000的数量
        pred_low = (actual_zero['predicted_power'] < 1000).sum()
        print(f"  预测值<1000 kW的时间点: {pred_low} ({pred_low/len(actual_zero)*100:.2f}%)")
    
    # 预测值<1000但实际功率>0的情况
    pred_low_actual_high = df[(df['predicted_power'] < 1000) & (df['实际功率'] > 10000)]
    print(f"\n预测值<1000但实际功率>10000 kW的时间点数: {len(pred_low_actual_high)}")
    if len(pred_low_actual_high) > 0:
        print(f"  实际功率均值: {pred_low_actual_high['实际功率'].mean():.2f} kW")
        print(f"  预测值均值: {pred_low_actual_high['predicted_power'].mean():.2f} kW")


def analyze_power_transitions(df):
    """分析功率转换点"""
    print("\n" + "="*60)
    print("功率转换点分析")
    print("="*60)
    
    # 计算功率变化率
    df = df.sort_values('timestamp').copy()
    df['actual_change'] = df['实际功率'].diff()
    df['predicted_change'] = df['predicted_power'].diff()
    df['change_error'] = df['predicted_change'] - df['actual_change']
    
    # 大幅变化的情况（变化>10000 kW）
    large_changes = df[np.abs(df['actual_change']) > 10000]
    
    print(f"\n实际功率变化>10000 kW的时间点数: {len(large_changes)}")
    if len(large_changes) > 0:
        print(f"  平均实际变化: {large_changes['actual_change'].mean():.2f} kW")
        print(f"  平均预测变化: {large_changes['predicted_change'].mean():.2f} kW")
        print(f"  平均变化误差: {large_changes['change_error'].mean():.2f} kW")


def plot_segment_analysis(df):
    """绘制功率段分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    segments = ['低功率段(<5k)', '中低功率段(5-15k)', '中高功率段(15-30k)', '高功率段(>30k)']
    colors = ['blue', 'green', 'orange', 'red']
    
    # 子图1：误差箱线图
    error_data = []
    error_labels = []
    for i, segment in enumerate(segments):
        segment_df = df[df['power_segment'] == segment]
        if len(segment_df) > 0:
            error_data.append(segment_df['error'].values)
            error_labels.append(segment)
    
    bp = axes[0, 0].boxplot(error_data, labels=error_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axes[0, 0].set_title('各功率段误差分布', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('误差 (kW)', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # 子图2：预测vs实际散点图
    for i, segment in enumerate(segments):
        segment_df = df[df['power_segment'] == segment]
        if len(segment_df) > 0:
            axes[0, 1].scatter(segment_df['实际功率'], segment_df['predicted_power'], 
                           alpha=0.6, s=20, label=segment, color=colors[i])
    
    # 添加对角线
    max_val = max(df['实际功率'].max(), df['predicted_power'].max())
    axes[0, 1].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='理想预测')
    axes[0, 1].set_title('预测值 vs 实际值', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('实际功率 (kW)', fontsize=12)
    axes[0, 1].set_ylabel('预测功率 (kW)', fontsize=12)
    axes[0, 1].legend(fontsize=10, loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3：绝对误差分布
    for i, segment in enumerate(segments):
        segment_df = df[df['power_segment'] == segment]
        if len(segment_df) > 0:
            axes[1, 0].hist(segment_df['abs_error'], bins=30, alpha=0.6, 
                           label=segment, color=colors[i], density=True)
    
    axes[1, 0].set_title('绝对误差分布', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('绝对误差 (kW)', fontsize=12)
    axes[1, 0].set_ylabel('密度', fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 子图4：MAPE分布
    for i, segment in enumerate(segments):
        segment_df = df[df['power_segment'] == segment]
        if len(segment_df) > 0:
            # 过滤零功率
            nonzero_df = segment_df[segment_df['实际功率'] > 100]
            if len(nonzero_df) > 0:
                axes[1, 1].hist(nonzero_df['mape'], bins=30, alpha=0.6, 
                               label=segment, color=colors[i], density=True)
    
    axes[1, 1].set_title('MAPE分布（排除零功率）', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('MAPE (%)', fontsize=12)
    axes[1, 1].set_ylabel('密度', fontsize=12)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_xlim(0, 200)
    
    plt.tight_layout()
    plt.savefig('功率段分析图.png', dpi=300, bbox_inches='tight')
    print("\n功率段分析图已保存至: 功率段分析图.png")
    plt.close()


def plot_time_series_analysis(df):
    """绘制时序分析图"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # 选择前500个点
    n_points = min(500, len(df))
    plot_df = df.iloc[:n_points].copy()
    
    # 子图1：功率时序
    axes[0].plot(plot_df.index, plot_df['实际功率'], 
                label='实际功率', alpha=0.8, linewidth=1.5)
    axes[0].plot(plot_df.index, plot_df['predicted_power'], 
                label='预测功率', alpha=0.6, linewidth=1.2, linestyle='--')
    axes[0].set_title('功率时序对比', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('功率 (kW)', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 子图2：误差时序
    axes[1].plot(plot_df.index, plot_df['error'], 
                color='red', alpha=0.7, linewidth=1)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_title('预测误差时序', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('误差 (kW)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # 子图3：MAPE时序
    nonzero_mask = plot_df['实际功率'] > 100
    axes[2].plot(plot_df[nonzero_mask].index, plot_df[nonzero_mask]['mape'], 
                color='green', alpha=0.7, linewidth=1)
    axes[2].axhline(y=20, color='r', linestyle='--', alpha=0.5, label='20%阈值')
    axes[2].set_title('MAPE时序（排除零功率）', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('时间点', fontsize=12)
    axes[2].set_ylabel('MAPE (%)', fontsize=12)
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 200)
    
    plt.tight_layout()
    plt.savefig('时序分析图.png', dpi=300, bbox_inches='tight')
    print("时序分析图已保存至: 时序分析图.png")
    plt.close()


def main():
    """主函数"""
    print("=" * 60)
    print("功率段误差分析")
    print("=" * 60)
    
    # 1. 加载数据
    df = load_data()
    
    # 2. 定义功率段
    df = define_power_segments(df)
    
    # 3. 按功率段分析
    segment_results = analyze_by_segment(df)
    segment_results.to_csv('功率段误差统计.csv', index=False)
    print("\n功率段误差统计已保存至: 功率段误差统计.csv")
    
    # 4. 按时段分析
    hourly_stats = analyze_by_time(df)
    hourly_stats.to_csv('时段误差统计.csv')
    print("\n时段误差统计已保存至: 时段误差统计.csv")
    
    # 5. 零功率误差分析
    analyze_zero_power_errors(df)
    
    # 6. 功率转换点分析
    analyze_power_transitions(df)
    
    # 7. 可视化
    plot_segment_analysis(df)
    plot_time_series_analysis(df)
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)


if __name__ == "__main__":
    main()
