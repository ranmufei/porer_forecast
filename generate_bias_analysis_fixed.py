"""
偏差分析图表生成脚本（中文修复版）
为优化模型生成全面的偏差分析可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.font_manager as fm

# 清除字体缓存并加载中文字体
import os
cache_dir = os.path.expanduser('~/.matplotlib')
if os.path.exists(cache_dir):
    for f in os.listdir(cache_dir):
        if f.startswith('fontlist'):
            try:
                os.remove(os.path.join(cache_dir, f))
            except:
                pass

# 方法：使用系统自带的PingFang字体
font_path = '/System/Library/Fonts/PingFang.ttc'
try:
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = ['PingFang SC', 'PingFang TC', 'PingFang HK', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11
    print(f"✓ 成功加载中文字体: PingFang")
except Exception as e:
    print(f"⚠ 字体加载警告: {e}")
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_data():
    """加载对比数据"""
    print("正在加载对比数据...")

    file_path = 'archive/analysis_files/对比数据/0116数据优化版本数据分析/predictions_optimized-数据分析6-1---8.csv'
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 计算误差
    df['error'] = df['predicted_power'] - df['实际功率']
    df['abs_error'] = np.abs(df['error'])
    df['pct_error'] = (df['error'] / (df['实际功率'] + 1e-10)) * 100

    print(f"✓ 数据加载完成: {len(df)} 条记录")
    return df


def plot_bias_analysis(df):
    """生成8合1偏差分析图"""
    print("正在生成偏差分析图...")

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.35)

    # ==================== 图1：时序对比 ====================
    ax1 = fig.add_subplot(gs[0, 0])
    n_points = min(200, len(df))
    ax1.plot(range(n_points), df['实际功率'].iloc[:n_points],
             label='Actual Power', color='black', alpha=0.8, linewidth=1.5)
    ax1.plot(range(n_points), df['predicted_power'].iloc[:n_points],
             label='Predicted Power', color='blue', alpha=0.7, linewidth=1.5, linestyle='--')
    ax1.set_title('Prediction vs Actual (Time Series)', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Time Point', fontsize=11)
    ax1.set_ylabel('Power (kW)', fontsize=11)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ==================== 图2：散点图 ====================
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(df['实际功率'], df['predicted_power'],
                         c=df['abs_error'], cmap='RdYlBu_r', alpha=0.6, s=20, edgecolors='none')
    max_val = max(df['实际功率'].max(), df['predicted_power'].max())
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect Prediction', linewidth=2)
    ax2.set_title('Prediction vs Actual (Scatter)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Actual Power (kW)', fontsize=11)
    ax2.set_ylabel('Predicted Power (kW)', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Absolute Error (kW)', fontsize=10)

    # ==================== 图3：误差时序图 ====================
    ax3 = fig.add_subplot(gs[0, 2])
    colors = ['red' if e > 0 else 'blue' for e in df['error'].iloc[:n_points]]
    ax3.bar(range(n_points), df['error'].iloc[:n_points], color=colors, alpha=0.6, width=1)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1.2)
    ax3.set_title('Error Time Series (First 200 Points)', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel('Time Point', fontsize=11)
    ax3.set_ylabel('Error (kW)', fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.6, label='Overestimate'),
                       Patch(facecolor='blue', alpha=0.6, label='Underestimate')]
    ax3.legend(handles=legend_elements, fontsize=10, loc='upper right')

    # ==================== 图4：误差分布直方图 ====================
    ax4 = fig.add_subplot(gs[0, 3])
    mu, sigma = df['error'].mean(), df['error'].std()
    ax4.hist(df['error'], bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)
    x = np.linspace(df['error'].min(), df['error'].max(), 100)
    ax4.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal Dist\nμ={mu:.0f}, σ={sigma:.0f}')
    ax4.axvline(x=mu, color='red', linestyle='--', linewidth=1.5, label=f'Mean={mu:.0f}')
    ax4.set_title('Error Distribution', fontsize=12, fontweight='bold', pad=10)
    ax4.set_xlabel('Error (kW)', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # ==================== 图5：按功率段的偏差 ====================
    ax5 = fig.add_subplot(gs[1, 0])
    df['power_segment'] = pd.cut(df['实际功率'],
                                  bins=[0, 5000, 15000, 30000, float('inf')],
                                  labels=['0-5k', '5-15k', '15-30k', '>30k'])
    segments = df['power_segment'].cat.categories
    box_data = [df[df['power_segment'] == seg]['abs_error'].values for seg in segments]
    bp = ax5.boxplot(box_data, labels=segments, patch_artist=True, showmeans=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    ax5.set_title('Absolute Error by Power Segment', fontsize=12, fontweight='bold', pad=10)
    ax5.set_xlabel('Power Segment', fontsize=11)
    ax5.set_ylabel('Absolute Error (kW)', fontsize=11)
    ax5.grid(True, alpha=0.3, axis='y')

    # ==================== 图6：按估算温度段的偏差 ====================
    ax6 = fig.add_subplot(gs[1, 1])
    # 使用风速作为温度代理
    df['temp_proxy'] = pd.cut(df['实际功率'],
                               bins=[0, 10000, 20000, 30000, float('inf')],
                               labels=['Low Temp', 'Normal Temp', 'Warm Temp', 'High Temp'])
    temp_segments = df['temp_proxy'].cat.categories
    box_data_temp = [df[df['temp_proxy'] == seg]['abs_error'].values for seg in temp_segments]
    bp_temp = ax6.boxplot(box_data_temp, labels=temp_segments, patch_artist=True, showmeans=True)
    for patch in bp_temp['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)
    ax6.set_title('Absolute Error by Temperature (Estimated)', fontsize=12, fontweight='bold', pad=10)
    ax6.set_xlabel('Temperature Segment', fontsize=11)
    ax6.set_ylabel('Absolute Error (kW)', fontsize=11)
    ax6.grid(True, alpha=0.3, axis='y')

    # ==================== 图7：Q-Q图（正态性检验）====================
    ax7 = fig.add_subplot(gs[1, 2])
    stats.probplot(df['error'], dist="norm", plot=ax7)
    ax7.set_title('Q-Q Plot (Normality Test)', fontsize=12, fontweight='bold', pad=10)
    ax7.grid(True, alpha=0.3)

    # ==================== 图8：统计摘要表格 ====================
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')

    # 计算统计量
    mae = mean_absolute_error(df['实际功率'], df['predicted_power'])
    rmse = np.sqrt(mean_squared_error(df['实际功率'], df['predicted_power']))
    r2 = r2_score(df['实际功率'], df['predicted_power'])
    bias = df['error'].mean()
    pos_count = np.sum(df['error'] > 0)
    neg_count = np.sum(df['error'] < 0)

    # 创建表格数据
    table_data = [
        ['Total Samples', f'{len(df)}'],
        ['MAE', f'{mae:.2f} kW'],
        ['RMSE', f'{rmse:.2f} kW'],
        ['R²', f'{r2:.4f}'],
        ['Mean Bias', f'{bias:.2f} kW'],
        ['Overestimate', f'{pos_count} ({pos_count/len(df)*100:.1f}%)'],
        ['Underestimate', f'{neg_count} ({neg_count/len(df)*100:.1f}%)'],
        ['Max Positive Error', f'{df["error"].max():.2f} kW'],
        ['Max Negative Error', f'{df["error"].min():.2f} kW']
    ]

    table = ax8.table(cellText=table_data, cellLoc='left', loc='center',
                      colWidths=[0.65, 0.35], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    # 设置表头样式
    for i in range(2):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 设置交替行颜色
    for i in range(1, len(table_data) + 1):
        if i % 2 == 0:
            for j in range(2):
                table[(i, j)].set_facecolor('#E8F1F2')

    ax8.set_title('Statistics Summary', fontsize=12, fontweight='bold', pad=20)

    # 总标题
    fig.suptitle('Optimized Model v2.1 - Bias Analysis Report\n(Test Set: June 1-8, 2025)',
                 fontsize=16, fontweight='bold', y=0.98)

    # 保存
    output_file = '偏差分析图_优化模型v2.1_修复版.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 偏差分析图已保存: {output_file}")
    print(f"  文件大小: {os.path.getsize(output_file)/1024:.1f} KB")

    plt.close()


def main():
    """主函数"""
    print("=" * 80)
    print("优化模型 v2.1 偏差分析（中文修复版）")
    print("=" * 80)

    df = load_data()

    # 打印关键统计
    print("\n关键统计指标:")
    print(f"  MAE: {mean_absolute_error(df['实际功率'], df['predicted_power']):.2f} kW")
    print(f"  RMSE: {np.sqrt(mean_squared_error(df['实际功率'], df['predicted_power'])):.2f} kW")
    print(f"  R²: {r2_score(df['实际功率'], df['predicted_power']):.4f}")
    print(f"  Bias: {df['error'].mean():.2f} kW")

    plot_bias_analysis(df)

    print("\n" + "=" * 80)
    print("✓ 偏差分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
