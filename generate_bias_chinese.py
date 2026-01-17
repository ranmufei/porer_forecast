"""
纯中文版偏差分析图
使用中文标签和图例
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.font_manager as fm
import os

# 清除字体缓存
cache_dir = os.path.expanduser('~/.matplotlib')
if os.path.exists(cache_dir):
    for f in os.listdir(cache_dir):
        if f.startswith('fontlist'):
            try:
                os.remove(os.path.join(cache_dir, f))
            except:
                pass

# 加载中文字体
font_path = '/System/Library/Fonts/PingFang.ttc'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = ['PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

def load_data():
    print("正在加载数据...")
    file_path = 'archive/analysis_files/对比数据/0116数据优化版本数据分析/predictions_optimized-数据分析6-1---8.csv'
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['error'] = df['predicted_power'] - df['实际功率']
    df['abs_error'] = np.abs(df['error'])
    print(f"✓ 加载完成: {len(df)} 条记录")
    return df

def plot_chinese_version(df):
    print("正在生成中文版图表...")
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 图1：预测vs实际时序
    ax1 = fig.add_subplot(gs[0, 0])
    n = min(150, len(df))
    ax1.plot(df['实际功率'].iloc[:n], label='实际功率', color='black', alpha=0.8, linewidth=1.5)
    ax1.plot(df['predicted_power'].iloc[:n], label='预测功率', color='#E74C3C', alpha=0.7, linewidth=1.5)
    ax1.set_title('预测值 vs 实际值', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('时间点', fontsize=11)
    ax1.set_ylabel('功率 (kW)', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 图2：散点图
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(df['实际功率'], df['predicted_power'],
                         c=df['abs_error'], cmap='YlOrRd', alpha=0.5, s=15)
    max_v = max(df['实际功率'].max(), df['predicted_power'].max())
    ax2.plot([0, max_v], [0, max_v], 'k--', alpha=0.5, linewidth=2)
    ax2.text(max_v*0.05, max_v*0.9, f'R²={r2_score(df["实际功率"], df["predicted_power"]):.3f}',
             fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.set_title('预测值 vs 实际值散点图', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('实际功率 (kW)', fontsize=11)
    ax2.set_ylabel('预测功率 (kW)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='误差(kW)')

    # 图3：误差分布
    ax3 = fig.add_subplot(gs[0, 2])
    mu, sigma = df['error'].mean(), df['error'].std()
    ax3.hist(df['error'], bins=40, alpha=0.7, color='#3498DB', edgecolor='black')
    ax3.axvline(mu, color='red', linestyle='--', linewidth=2, label=f'均值={mu:.0f}')
    ax3.set_title(f'误差分布 (μ={mu:.0f}, σ={sigma:.0f})', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel('误差 (kW)', fontsize=11)
    ax3.set_ylabel('频数', fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

    # 图4：按功率段误差
    ax4 = fig.add_subplot(gs[1, 0])
    df['segment'] = pd.cut(df['实际功率'], bins=[0, 5000, 15000, 30000, float('inf')],
                          labels=['0-5k', '5-15k', '15-30k', '>30k'])
    segs = df['segment'].cat.categories
    data = [df[df['segment'] == s]['abs_error'] for s in segs]
    bp = ax4.boxplot(data, labels=segs, patch_artist=True, showmeans=True)
    colors = ['#AED6F1', '#A9DFBF', '#F9E79F', '#F5B7B1']
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
    ax4.set_title('各功率段误差分布', fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlabel('功率段', fontsize=11)
    ax4.set_ylabel('绝对误差 (kW)', fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')

    # 图5：误差时序（正负）
    ax5 = fig.add_subplot(gs[1, 1])
    pos = df[df['error'] > 0]['error']
    neg = df[df['error'] < 0]['error']
    ax5.hist([pos, neg], bins=30, stacked=True, color=['#E74C3C', '#3498DB'],
             alpha=0.7, label=['高估', '低估'], edgecolor='black')
    ax5.set_title('误差方向分布', fontsize=13, fontweight='bold', pad=10)
    ax5.set_xlabel('误差 (kW)', fontsize=11)
    ax5.set_ylabel('频数', fontsize=11)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.axvline(0, color='black', linestyle='-', linewidth=1)

    # 图6：统计摘要
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    mae = mean_absolute_error(df['实际功率'], df['predicted_power'])
    rmse = np.sqrt(mean_squared_error(df['实际功率'], df['predicted_power']))
    r2 = r2_score(df['实际功率'], df['predicted_power'])

    stats_text = f"""
    统计摘要
    {'='*40}
    样本数: {len(df)}

    MAE: {mae:.2f} kW
    RMSE: {rmse:.2f} kW
    R²: {r2:.4f}
    偏差: {df['error'].mean():.2f} kW

    高估: {(df["error"]>0).sum()} ({(df["error"]>0).sum()/len(df)*100:.1f}%)
    低估: {(df["error"]<0).sum()} ({(df["error"]<0).sum()/len(df)*100:.1f}%)

    最大正误差: {df["error"].max():.2f} kW
    最大负误差: {df["error"].min():.2f} kW
    """

    ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax6.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    fig.suptitle('优化模型 v2.1 偏差分析报告（中文版）', fontsize=15, fontweight='bold', y=0.98)

    plt.savefig('偏差分析图_中文版.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ 中文版图表已保存: 偏差分析图_中文版.png")
    plt.close()

def main():
    print("=" * 60)
    print("纯中文版偏差分析图")
    print("=" * 60)

    df = load_data()
    plot_chinese_version(df)

    print("\n✓ 完成!")

if __name__ == "__main__":
    main()
