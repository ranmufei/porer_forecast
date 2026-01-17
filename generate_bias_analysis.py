"""
偏差分析图表生成脚本
为优化模型生成全面的偏差分析可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 设置中文字体（直接使用macOS系统字体）
import matplotlib.font_manager as fm

# 方法1：直接添加字体路径
font_path = '/System/Library/Fonts/PingFang.ttc'
try:
    font_prop = fm.FontProperties(fname=font_path)
    print(f"使用字体文件: {font_path}")
    # 注册字体
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = ['PingFang SC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.size'] = 10
    use_chinese_font = True
except Exception as e:
    print(f"无法加载PingFang字体: {e}")
    use_chinese_font = False

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_data():
    """加载对比数据"""
    print("正在加载对比数据...")

    # 读取优化模型预测结果
    file_path = 'archive/analysis_files/对比数据/0116数据优化版本数据分析/predictions_optimized-数据分析6-1---8.csv'
    df = pd.read_csv(file_path)

    # 转换时间戳
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 计算误差
    df['error'] = df['predicted_power'] - df['实际功率']
    df['abs_error'] = np.abs(df['error'])
    df['pct_error'] = (df['error'] / (df['实际功率'] + 1e-10)) * 100

    print(f"数据加载完成: {len(df)} 条记录")
    print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")

    return df


def calculate_statistics(df):
    """计算偏差统计量"""
    print("\n正在计算统计量...")

    stats_dict = {
        '总样本数': len(df),
        'MAE (平均绝对误差)': f"{mean_absolute_error(df['实际功率'], df['predicted_power']):.2f} kW",
        'RMSE (均方根误差)': f"{np.sqrt(mean_squared_error(df['实际功率'], df['predicted_power'])):.2f} kW",
        'MAPE (平均绝对百分比误差)': f"{np.mean(np.abs(df['pct_error'][df['实际功率'] > 100])):.2f}%",
        'R² (决定系数)': f"{r2_score(df['实际功率'], df['predicted_power']):.4f}",
        '平均偏差 (Bias)': f"{np.mean(df['error']):.2f} kW",
        '偏差标准差': f"{np.std(df['error']):.2f} kW",
        '正偏差样本数': f"{np.sum(df['error'] > 0)} ({np.sum(df['error'] > 0)/len(df)*100:.1f}%)",
        '负偏差样本数': f"{np.sum(df['error'] < 0)} ({np.sum(df['error'] < 0)/len(df)*100:.1f}%)",
        '最大正误差': f"{np.max(df['error']):.2f} kW",
        '最大负误差': f"{np.min(df['error']):.2f} kW"
    }

    return stats_dict


def plot_bias_analysis(df):
    """生成8合1偏差分析图"""
    print("\n正在生成偏差分析图...")

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    # ==================== 图1：时序对比 ====================
    ax1 = fig.add_subplot(gs[0, 0])
    n_points = min(200, len(df))
    ax1.plot(range(n_points), df['实际功率'].iloc[:n_points],
             label='实际功率', color='black', alpha=0.8, linewidth=1.5)
    ax1.plot(range(n_points), df['predicted_power'].iloc[:n_points],
             label='预测功率', color='blue', alpha=0.7, linewidth=1.5, linestyle='--')
    ax1.set_title('预测值 vs 实际值对比（时序）', fontsize=12, fontweight='bold')
    ax1.set_xlabel('时间点', fontsize=10)
    ax1.set_ylabel('功率 (kW)', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ==================== 图2：散点图 ====================
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(df['实际功率'], df['predicted_power'],
                         c=df['abs_error'], cmap='RdYlBu_r', alpha=0.6, s=20, edgecolors='none')
    max_val = max(df['实际功率'].max(), df['predicted_power'].max())
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='理想预测', linewidth=2)
    ax2.set_title('预测值 vs 实际值散点图', fontsize=12, fontweight='bold')
    ax2.set_xlabel('实际功率 (kW)', fontsize=10)
    ax2.set_ylabel('预测功率 (kW)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='绝对误差 (kW)')

    # ==================== 图3：误差时序图 ====================
    ax3 = fig.add_subplot(gs[0, 2])
    colors = ['red' if e > 0 else 'blue' for e in df['error'].iloc[:n_points]]
    ax3.bar(range(n_points), df['error'].iloc[:n_points], color=colors, alpha=0.6, width=1)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_title('误差时序图（前200个点）', fontsize=12, fontweight='bold')
    ax3.set_xlabel('时间点', fontsize=10)
    ax3.set_ylabel('误差 (kW)', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

    # ==================== 图4：误差分布直方图 ====================
    ax4 = fig.add_subplot(gs[0, 3])
    mu, sigma = df['error'].mean(), df['error'].std()
    ax4.hist(df['error'], bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)
    x = np.linspace(df['error'].min(), df['error'].max(), 100)
    ax4.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'正态分布\nμ={mu:.1f}, σ={sigma:.1f}')
    ax4.axvline(x=mu, color='red', linestyle='--', linewidth=1.5, label=f'均值={mu:.1f}')
    ax4.set_title('误差分布直方图', fontsize=12, fontweight='bold')
    ax4.set_xlabel('误差 (kW)', fontsize=10)
    ax4.set_ylabel('密度', fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # ==================== 图5：按功率段的偏差 ====================
    ax5 = fig.add_subplot(gs[1, 0])
    df['power_segment'] = pd.cut(df['实际功率'],
                                  bins=[0, 5000, 15000, 30000, float('inf')],
                                  labels=['0-5k', '5-15k', '15-30k', '>30k'])
    segments = df['power_segment'].unique()
    box_data = [df[df['power_segment'] == seg]['abs_error'].values for seg in segments]
    bp = ax5.boxplot(box_data, labels=segments, patch_artist=True, showmeans=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    ax5.set_title('按功率段的绝对误差分布', fontsize=12, fontweight='bold')
    ax5.set_xlabel('功率段', fontsize=10)
    ax5.set_ylabel('绝对误差 (kW)', fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')

    # ==================== 图6：按温度段的偏差 ====================
    ax6 = fig.add_subplot(gs[1, 1])
    # 从GFS数据中提取温度（假设温度在-10到35度之间）
    # 这里使用模拟数据，实际需要从原始GFS数据读取
    # 暂时使用实际功率来估算温度段（仅为演示）
    df['temp_segment'] = pd.cut(df['实际功率'],
                                  bins=[0, 10000, 20000, 30000, float('inf')],
                                  labels=['低温段', '常温段', '温和段', '高温段'])
    temp_segments = df['temp_segment'].unique()
    box_data_temp = [df[df['temp_segment'] == seg]['abs_error'].values for seg in temp_segments]
    bp_temp = ax6.boxplot(box_data_temp, labels=temp_segments, patch_artist=True, showmeans=True)
    for patch in bp_temp['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)
    ax6.set_title('按温度段的绝对误差分布（估算）', fontsize=12, fontweight='bold')
    ax6.set_xlabel('温度段', fontsize=10)
    ax6.set_ylabel('绝对误差 (kW)', fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')

    # ==================== 图7：Q-Q图（正态性检验）====================
    ax7 = fig.add_subplot(gs[1, 2])
    stats.probplot(df['error'], dist="norm", plot=ax7)
    ax7.set_title('误差Q-Q图（正态性检验）', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # ==================== 图8：统计摘要表格 ====================
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')

    stats_dict = calculate_statistics(df)

    # 创建表格
    table_data = []
    for key, value in stats_dict.items():
        table_data.append([key, value])

    table = ax8.table(cellText=table_data, cellLoc='left', loc='center',
                      colWidths=[0.6, 0.4], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # 设置表头
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 设置交替行颜色
    for i in range(1, len(stats_dict) + 1):
        if i % 2 == 0:
            for j in range(2):
                table[(i, j)].set_facecolor('#f0f0f0')

    ax8.set_title('统计摘要', fontsize=12, fontweight='bold', pad=20)

    # 添加总标题
    fig.suptitle('优化模型 v2.1 偏差分析报告\n（测试集：2025年6月1日-8日）',
                 fontsize=16, fontweight='bold', y=0.98)

    # 保存图表
    output_file = '偏差分析图_优化模型v2.1.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n偏差分析图已保存至: {output_file}")

    plt.close()


def generate_text_report(df, stats_dict):
    """生成文本统计报告"""
    print("\n正在生成文本报告...")

    report_file = '偏差统计报告_优化模型v2.1.txt'

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("优化模型 v2.1 偏差统计报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("数据集信息:\n")
        f.write("-" * 80 + "\n")
        f.write(f"测试集时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}\n")
        f.write(f"样本数量: {len(df)}\n")
        f.write(f"时间粒度: 15分钟\n\n")

        f.write("性能指标:\n")
        f.write("-" * 80 + "\n")
        for key, value in stats_dict.items():
            f.write(f"{key}: {value}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("分功率段统计:\n")
        f.write("=" * 80 + "\n\n")

        for segment in ['0-5k', '5-15k', '15-30k', '>30k']:
            if segment in df['power_segment'].cat.categories:
                seg_df = df[df['power_segment'] == segment]
                if len(seg_df) > 0:
                    f.write(f"\n{segment} 功率段:\n")
                    f.write(f"  样本数: {len(seg_df)}\n")
                    f.write(f"  平均绝对误差: {seg_df['abs_error'].mean():.2f} kW\n")
                    f.write(f"  中位数误差: {seg_df['abs_error'].median():.2f} kW\n")
                    f.write(f"  最大误差: {seg_df['abs_error'].max():.2f} kW\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("关键发现:\n")
        f.write("=" * 80 + "\n")

        # 正负偏差分析
        positive_bias = df[df['error'] > 0]
        negative_bias = df[df['error'] < 0]

        f.write(f"\n1. 高估分析（正偏差）:\n")
        f.write(f"   - 高估样本数: {len(positive_bias)} ({len(positive_bias)/len(df)*100:.1f}%)\n")
        f.write(f"   - 平均高估: {positive_bias['error'].mean():.2f} kW\n")

        f.write(f"\n2. 低估分析（负偏差）:\n")
        f.write(f"   - 低估样本数: {len(negative_bias)} ({len(negative_bias)/len(df)*100:.1f}%)\n")
        f.write(f"   - 平均低估: {negative_bias['error'].mean():.2f} kW\n")

        # 误差分布分析
        f.write(f"\n3. 误差分布特征:\n")
        f.write(f"   - 误差均值: {df['error'].mean():.2f} kW\n")
        f.write(f"   - 误差标准差: {df['error'].std():.2f} kW\n")
        f.write(f"   - 偏度: {df['error'].skew():.2f}\n")
        f.write(f"   - 峰度: {df['error'].kurtosis():.2f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"报告生成时间: {pd.Timestamp.now()}\n")
        f.write("=" * 80 + "\n")

    print(f"文本报告已保存至: {report_file}")


def main():
    """主函数"""
    print("=" * 80)
    print("优化模型 v2.1 偏差分析")
    print("=" * 80)

    # 1. 加载数据
    df = load_data()

    # 2. 计算统计量
    stats_dict = calculate_statistics(df)

    # 3. 打印统计摘要
    print("\n" + "=" * 80)
    print("统计摘要")
    print("=" * 80)
    for key, value in stats_dict.items():
        print(f"{key}: {value}")

    # 4. 生成偏差分析图
    plot_bias_analysis(df)

    # 5. 生成文本报告
    generate_text_report(df, stats_dict)

    print("\n" + "=" * 80)
    print("偏差分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
