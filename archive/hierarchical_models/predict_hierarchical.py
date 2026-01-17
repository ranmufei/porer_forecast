"""
分层模型+物理约束融合预测脚本
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_models():
    """加载所有分层模型"""
    print("正在加载分层模型...")
    
    # 加载模型信息
    with open('hierarchical_model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)
    
    # 加载各功率段模型
    models = {}
    for segment_key in model_info['segments']:
        with open(f'hierarchical_model_{segment_key}.pkl', 'rb') as f:
            models[segment_key] = pickle.load(f)
        print(f"  已加载: {models[segment_key]['info']['name']}")
    
    print(f"共加载 {len(models)} 个模型")
    
    return models, model_info


def classify_power_segment(features, model_info):
    """
    基于特征分类功率段
    使用风速和估算功率进行分类
    """
    # 从特征中提取风速和估算功率
    wind_speed = features['gfs_wind_speed']
    estimated_max_power = features['estimated_max_power']
    
    # 物理规则分类
    if wind_speed < 3:
        return 'zero'  # 切入风速以下，大概率零功率
    elif wind_speed > 25:
        return 'zero'  # 切出风速以上，停机
    elif estimated_max_power < 100:
        return 'zero'
    elif estimated_max_power < 5000:
        return 'low'
    elif estimated_max_power < 15000:
        return 'medium_low'
    elif estimated_max_power < 30000:
        return 'medium_high'
    else:
        return 'high'


def wind_power_curve(wind_speed, v_in=3, v_rated=12, v_out=25, P_rated=50000):
    """
    标准风电功率曲线
    Args:
        wind_speed: 风速 (m/s)
        v_in: 切入风速 (m/s)
        v_rated: 额定风速 (m/s)
        v_out: 切出风速 (m/s)
        P_rated: 额定功率 (kW)
    """
    if wind_speed < v_in or wind_speed > v_out:
        return 0.0
    elif v_in <= wind_speed < v_rated:
        # 三次方关系
        return P_rated * (wind_speed**3 - v_in**3) / (v_rated**3 - v_in**3)
    elif v_rated <= wind_speed <= v_out:
        return P_rated
    else:
        return 0.0


def apply_physical_constraints(predictions, df):
    """
    应用物理约束后处理
    """
    print("\n正在应用物理约束...")
    
    df = df.copy()
    df['predicted_raw'] = predictions.copy()
    df['predicted_constrained'] = predictions.copy()
    
    # 1. 负值截断
    df['predicted_constrained'] = df['predicted_constrained'].clip(lower=0)
    
    # 2. 风速-功率曲线约束
    print("  应用功率曲线约束...")
    max_power_curve = df['gfs_wind_speed'].apply(wind_power_curve)
    df['predicted_constrained'] = np.minimum(
        df['predicted_constrained'], 
        max_power_curve * 1.2  # 允许20%的超出容差
    )
    
    # 3. 切入风速强制零功率
    print("  应用切入风速约束...")
    cutin_mask = df['gfs_wind_speed'] < 3
    df.loc[cutin_mask, 'predicted_constrained'] = 0
    
    # 4. 切出风速强制零功率
    print("  应用切出风速约束...")
    cutoff_mask = df['gfs_wind_speed'] > 25
    df.loc[cutoff_mask, 'predicted_constrained'] = 0
    
    # 5. 功率变化率约束（限制相邻时段变化）
    print("  应用功率变化率约束...")
    df['predicted_constrained'] = df['predicted_constrained'].astype(float)
    
    # 计算变化率
    df['predicted_change'] = df['predicted_constrained'].diff()
    
    # 限制单时段变化不超过20%或5000kW（取较小值）
    max_change = 5000
    
    # 向前修正
    for i in range(1, len(df)):
        change = df.iloc[i]['predicted_constrained'] - df.iloc[i-1]['predicted_constrained']
        if abs(change) > max_change:
            # 限制变化幅度
            df.iloc[i, df.columns.get_loc('predicted_constrained')] = \
                df.iloc[i-1]['predicted_constrained'] + np.sign(change) * max_change
    
    # 向后修正（反向遍历）
    for i in range(len(df)-2, -1, -1):
        change = df.iloc[i]['predicted_constrained'] - df.iloc[i+1]['predicted_constrained']
        if abs(change) > max_change:
            df.iloc[i, df.columns.get_loc('predicted_constrained')] = \
                df.iloc[i+1]['predicted_constrained'] + np.sign(change) * max_change
    
    # 6. 移动平均平滑
    print("  应用移动平均平滑...")
    df['predicted_constrained'] = pd.Series(
        df['predicted_constrained']
    ).rolling(
        window=3, 
        min_periods=1, 
        center=True
    ).mean().values
    
    # 7. 再次负值截断
    df['predicted_constrained'] = df['predicted_constrained'].clip(lower=0)
    
    print(f"  物理约束应用完成")
    
    return df['predicted_constrained'].values


def predict_with_hierarchical_models(gfs_file):
    """
    使用分层模型进行预测
    """
    print("=" * 60)
    print("分层模型+物理约束融合预测")
    print("=" * 60)
    
    # 1. 加载模型
    models, model_info = load_models()
    
    # 2. 读取GFS数据
    print(f"\n正在读取GFS数据: {gfs_file}")
    df = pd.read_csv(gfs_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 3. 特征工程
    print("正在进行特征工程...")
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df['wind_dir_sin'] = np.sin(2 * np.pi * df['gfs_wind_direction'] / 360)
    df['wind_dir_cos'] = np.cos(2 * np.pi * df['gfs_wind_direction'] / 360)
    
    df['wind_speed_cube'] = df['gfs_wind_speed'] ** 3
    df['wind_below_cutin'] = (df['gfs_wind_speed'] < 3).astype(int)
    df['wind_above_cutoff'] = (df['gfs_wind_speed'] > 25).astype(int)
    df['wind_in_ideal_range'] = ((df['gfs_wind_speed'] >= 10) & 
                                 (df['gfs_wind_speed'] <= 15)).astype(int)
    
    df['estimated_max_power'] = 50000 * (df['gfs_wind_speed'] / 12) ** 3
    df['estimated_max_power'] = df['estimated_max_power'].clip(lower=0, upper=50000)
    
    df['wind_temp_interaction'] = df['gfs_wind_speed'] * df['gfs_temp']
    
    print(f"特征工程完成，共 {len(df.columns)} 个特征")
    
    # 4. 逐点预测
    print("\n正在进行分层预测...")
    predictions = np.zeros(len(df))
    segment_counts = {}
    
    for i in range(len(df)):
        # 提取特征
        features = df.iloc[i].to_dict()
        
        # 分类功率段
        segment = classify_power_segment(features, model_info)
        segment_counts[segment] = segment_counts.get(segment, 0) + 1
        
        # 获取对应模型
        model_data = models[segment]
        model = model_data['model']
        feature_list = model_data['info']['features']
        
        # 准备输入
        X = np.array([features[f] for f in feature_list]).reshape(1, -1)
        
        # 预测
        pred = model.predict(X)[0]
        predictions[i] = max(pred, 0)  # 确保非负
    
    print("\n功率段分布:")
    for segment, count in segment_counts.items():
        print(f"  {models[segment]['info']['name']}: {count} ({count/len(df)*100:.1f}%)")
    
    # 5. 应用物理约束
    predictions_constrained = apply_physical_constraints(predictions, df)
    
    # 6. 保存结果
    result_df = df[['timestamp']].copy()
    result_df['predicted_power'] = predictions_constrained
    
    output_file = 'predictions_hierarchical.csv'
    result_df.to_csv(output_file, index=False)
    print(f"\n预测结果已保存至: {output_file}")
    print(f"共预测 {len(result_df)} 个时间点")
    
    # 7. 显示前10条结果
    print("\n前10条预测结果:")
    print(result_df.head(10).to_string(index=False))
    
    return result_df, df


def evaluate_hierarchical_model():
    """
    评估分层模型性能（使用对比数据）
    """
    print("\n" + "=" * 60)
    print("评估分层模型性能")
    print("=" * 60)
    
    # 加载对比数据
    compare_file = '对比数据/0116数据优化版本数据分析/predictions_optimized-数据分析6-1---8.csv'
    print(f"正在加载对比数据: {compare_file}")
    
    compare_df = pd.read_csv(compare_file)
    compare_df['timestamp'] = pd.to_datetime(compare_df['timestamp'])
    
    # 准备GFS特征
    gfs_df = pd.read_csv('data_gfs_forecast.csv')
    gfs_df['timestamp'] = pd.to_datetime(gfs_df['timestamp'])
    
    # 合并
    merged_df = pd.merge(compare_df, gfs_df, on='timestamp', how='inner')
    
    print(f"评估数据量: {len(merged_df)} 条")
    
    # 特征工程
    df = merged_df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df['wind_dir_sin'] = np.sin(2 * np.pi * df['gfs_wind_direction'] / 360)
    df['wind_dir_cos'] = np.cos(2 * np.pi * df['gfs_wind_direction'] / 360)
    
    df['wind_speed_cube'] = df['gfs_wind_speed'] ** 3
    df['wind_below_cutin'] = (df['gfs_wind_speed'] < 3).astype(int)
    df['wind_above_cutoff'] = (df['gfs_wind_speed'] > 25).astype(int)
    df['wind_in_ideal_range'] = ((df['gfs_wind_speed'] >= 10) & 
                                 (df['gfs_wind_speed'] <= 15)).astype(int)
    
    df['estimated_max_power'] = 50000 * (df['gfs_wind_speed'] / 12) ** 3
    df['estimated_max_power'] = df['estimated_max_power'].clip(lower=0, upper=50000)
    
    df['wind_temp_interaction'] = df['gfs_wind_speed'] * df['gfs_temp']
    
    # 加载模型
    models, model_info = load_models()
    
    # 预测
    print("\n正在进行预测...")
    predictions = np.zeros(len(df))
    segment_counts = {}
    
    for i in range(len(df)):
        features = df.iloc[i].to_dict()
        segment = classify_power_segment(features, model_info)
        segment_counts[segment] = segment_counts.get(segment, 0) + 1
        
        model_data = models[segment]
        model = model_data['model']
        feature_list = model_data['info']['features']
        
        X = np.array([features[f] for f in feature_list]).reshape(1, -1)
        pred = model.predict(X)[0]
        predictions[i] = max(pred, 0)
    
    # 应用物理约束
    predictions_constrained = apply_physical_constraints(predictions, df)
    
    # 评估
    actual = df['实际功率'].values
    
    mae = mean_absolute_error(actual, predictions_constrained)
    rmse = np.sqrt(mean_squared_error(actual, predictions_constrained))
    r2 = r2_score(actual, predictions_constrained)
    
    # MAPE（排除零功率）
    nonzero_mask = actual > 100
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((actual[nonzero_mask] - predictions_constrained[nonzero_mask]) / 
                               actual[nonzero_mask])) * 100
    else:
        mape = np.nan
    
    print("\n" + "=" * 60)
    print("分层模型性能指标")
    print("=" * 60)
    print(f"MAE:  {mae:.2f} kW")
    print(f"RMSE: {rmse:.2f} kW")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²:   {r2:.4f}")
    
    # 零功率分析
    actual_zero = actual == 0
    pred_zero = predictions_constrained < 500
    print(f"\n实际功率为0的时间点: {actual_zero.sum()}")
    print(f"预测功率<500 kW的时间点: {pred_zero.sum()}")
    print(f"零功率预测准确率: {np.sum(actual_zero & pred_zero) / actual_zero.sum() * 100:.2f}%")
    
    # 保存对比结果
    result_df = df[['timestamp']].copy()
    result_df['predicted_power'] = predictions_constrained
    result_df['actual_power'] = actual
    result_df['original_predicted'] = df['predicted_power'].values
    
    output_file = '对比数据/分层模型预测对比.csv'
    result_df.to_csv(output_file, index=False)
    print(f"\n对比结果已保存至: {output_file}")
    
    return result_df


def plot_comparison_hierarchical():
    """
    绘制分层模型与原始模型对比
    """
    print("\n正在生成对比图...")
    
    compare_file = '对比数据/分层模型预测对比.csv'
    if not pd.io.common.file_exists(compare_file):
        print("请先运行评估函数生成对比数据")
        return
    
    df = pd.read_csv(compare_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 计算误差
    df['error_hierarchical'] = df['predicted_power'] - df['actual_power']
    df['error_original'] = df['original_predicted'] - df['actual_power']
    df['abs_error_hierarchical'] = np.abs(df['error_hierarchical'])
    df['abs_error_original'] = np.abs(df['error_original'])
    
    # 选择前500个点
    n_points = min(500, len(df))
    plot_df = df.iloc[:n_points].copy()
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    
    # 子图1：功率时序对比
    axes[0, 0].plot(plot_df.index, plot_df['actual_power'], 
                    label='实际功率', alpha=0.8, linewidth=2, color='black')
    axes[0, 0].plot(plot_df.index, plot_df['original_predicted'], 
                    label='原始预测', alpha=0.6, linewidth=1.5, color='blue', linestyle='--')
    axes[0, 0].plot(plot_df.index, plot_df['predicted_power'], 
                    label='分层+物理约束', alpha=0.8, linewidth=2, color='red')
    axes[0, 0].set_title('功率预测对比（时序）', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('功率 (kW)', fontsize=12)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2：预测vs实际散点图
    axes[0, 1].scatter(plot_df['actual_power'], plot_df['original_predicted'], 
                       alpha=0.5, s=20, label='原始预测', color='blue')
    axes[0, 1].scatter(plot_df['actual_power'], plot_df['predicted_power'], 
                       alpha=0.7, s=20, label='分层+物理约束', color='red')
    max_val = max(plot_df['actual_power'].max(), plot_df['predicted_power'].max())
    axes[0, 1].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='理想预测')
    axes[0, 1].set_title('预测值 vs 实际值', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('实际功率 (kW)', fontsize=12)
    axes[0, 1].set_ylabel('预测功率 (kW)', fontsize=12)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3：误差对比
    axes[1, 0].plot(plot_df.index, plot_df['error_original'], 
                    label='原始预测误差', alpha=0.6, linewidth=1.5, color='blue')
    axes[1, 0].plot(plot_df.index, plot_df['error_hierarchical'], 
                    label='分层模型误差', alpha=0.8, linewidth=1.5, color='red')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('预测误差对比', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('误差 (kW)', fontsize=12)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 子图4：绝对误差对比
    axes[1, 1].plot(plot_df.index, plot_df['abs_error_original'], 
                    label='原始绝对误差', alpha=0.6, linewidth=1.5, color='blue')
    axes[1, 1].plot(plot_df.index, plot_df['abs_error_hierarchical'], 
                    label='分层绝对误差', alpha=0.8, linewidth=1.5, color='red')
    axes[1, 1].set_title('绝对误差对比', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('绝对误差 (kW)', fontsize=12)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 子图5：误差分布直方图
    axes[2, 0].hist(plot_df['abs_error_original'], bins=30, alpha=0.5, 
                     label='原始预测', color='blue', density=True)
    axes[2, 0].hist(plot_df['abs_error_hierarchical'], bins=30, alpha=0.7, 
                     label='分层+物理约束', color='red', density=True)
    axes[2, 0].set_title('绝对误差分布', fontsize=14, fontweight='bold')
    axes[2, 0].set_xlabel('绝对误差 (kW)', fontsize=12)
    axes[2, 0].set_ylabel('密度', fontsize=12)
    axes[2, 0].legend(fontsize=11)
    axes[2, 0].grid(True, alpha=0.3, axis='y')
    
    # 子图6：累计误差分布
    axes[2, 1].hist(plot_df['abs_error_original'], bins=30, alpha=0.5, 
                     label='原始预测', color='blue', density=True, cumulative=True)
    axes[2, 1].hist(plot_df['abs_error_hierarchical'], bins=30, alpha=0.7, 
                     label='分层+物理约束', color='red', density=True, cumulative=True)
    axes[2, 1].set_title('累计绝对误差分布', fontsize=14, fontweight='bold')
    axes[2, 1].set_xlabel('绝对误差 (kW)', fontsize=12)
    axes[2, 1].set_ylabel('累计密度', fontsize=12)
    axes[2, 1].legend(fontsize=11)
    axes[2, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('分层模型对比图.png', dpi=300, bbox_inches='tight')
    print("对比图已保存至: 分层模型对比图.png")
    plt.close()


def main():
    """主函数"""
    print("=" * 60)
    print("分层模型+物理约束融合预测系统")
    print("=" * 60)
    
    # 1. 对GFS数据进行预测
    result_df, _ = predict_with_hierarchical_models('data_gfs_forecast.csv')
    
    # 2. 评估模型性能
    evaluate_hierarchical_model()
    
    # 3. 生成对比图
    plot_comparison_hierarchical()
    
    print("\n" + "=" * 60)
    print("融合预测完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
