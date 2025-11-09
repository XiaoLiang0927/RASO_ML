import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def order_model(x, alpha_prime, beta):
    return alpha_prime * x * np.exp(beta * x)
def run_customer_estimation_for_all_merchants(file_path, start_time=None, end_time=None):
    print("开始读取CSV文件...")
    try:
        df = pd.read_csv(file_path)
        
        # 如果有时间范围参数，筛选数据
        if start_time is not None and end_time is not None:
            df = df[(df['timestamp'] >= start_time) & (df['timestamp'] < end_time)]
            print(f"筛选时间范围 {start_time}-{end_time} 的数据，共 {len(df)} 条记录")
        else:
            print(f"成功读取文件，共 {len(df)} 条数据记录")
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None
    
    # 确保列名符合预期
    expected_columns = ['iteration', 'customer_id', 'customer_x', 'customer_y', 'merchant_id', 'merchant_x', 'merchant_y', 'distance', 'timestamp']
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        print(f"错误：CSV文件缺少以下列：{missing_columns}")
        return None

    merchant_ids = sorted(df['merchant_id'].unique())
    print(f"识别到 {len(merchant_ids)} 个不同的商家ID: {', '.join(str(m) for m in merchant_ids[:5])}{'...' if len(merchant_ids) > 5 else ''}")
    results = {}

    total_merchants = len(merchant_ids)
    print(f"开始处理 {total_merchants} 个商家的数据...")

     # 获取总迭代次数
    total_iterations = df['iteration'].nunique()
    print(f"总迭代次数: {total_iterations}")
    
    for idx, merchant_id in enumerate(merchant_ids, 1):
        print(f"\n--- 正在为商家 ID: {merchant_id} 进行建模 ({idx}/{total_merchants}) ---")
        df_merchant = df[df['merchant_id'] == merchant_id].copy()

        if len(df_merchant) < 10:  # 如果某个商家订单太少，则跳过
            print(f"商家 {merchant_id} 订单数据过少 ({len(df_merchant)} 条)，跳过此商家。")
            continue

        print(f"  处理商家 {merchant_id} 的 {len(df_merchant)} 条订单数据...")
        
        '''
        # 1. 聚合数据 (Binning)
        bin_width = 1.0
        max_dist = df_merchant['distance'].max()
        bins = np.arange(0, max_dist + bin_width, bin_width)
        print(f"  按距离分组，最大距离: {max_dist:.2f} km, 分组宽度: {bin_width} km")
        df_merchant['distance_bin'] = pd.cut(df_merchant['distance'], bins=bins, right=False)
        aggregated_data = df_merchant.groupby('distance_bin', observed=True).size().reset_index(name='order_count')
        # 将category类型的distance_bin转换为数值类型的distance_mid
        aggregated_data['distance_mid'] = aggregated_data['distance_bin'].apply(lambda b: b.mid).astype(float)
        print(f"  聚合后得到 {len(aggregated_data)} 个距离分组")

        # 保存聚合数据用于检查
        agg_data_file = f'aggregated_data_merchant_{merchant_id}.csv'
        aggregated_data.to_csv(agg_data_file, index=False)
        print(f"  已将聚合数据保存到 {agg_data_file} 用于检查")
        
        # 2. 拟合模型
        print("  开始拟合订单-距离模型...")

        x_data = aggregated_data['distance_mid'].values
        y_data = aggregated_data['order_count'].values

        # 检查数据点数量是否足够进行拟合
        if len(x_data) < 2:
            print("  聚合后的数据点不足以进行拟合，跳过此商家。")
            continue

        try:
            print("  使用非线性最小二乘法拟合模型...")
            initial_guess = [max(y_data), -0.5]
            popt, _ = curve_fit(order_model, x_data, y_data, p0=initial_guess, maxfev=5000)

            estimated_alpha_prime, estimated_beta = popt

            # 计算R-squared
            residuals = y_data - order_model(x_data, *popt)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            # 避免ss_tot为0的情况
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            print(f"  拟合完成: alpha' = {estimated_alpha_prime:.4f}, beta = {estimated_beta:.4f}, R^2 = {r_squared:.4f}")

            # 3. 保存结果
            results[merchant_id] = {
                'alpha_prime': estimated_alpha_prime,
                'beta': estimated_beta,
                'r_squared': r_squared,
                'aggregated_data': aggregated_data
            }
            print(f"  商家 {merchant_id} 的模型参数已保存")
        except RuntimeError:
            print(f"  商家 {merchant_id} 的数据无法收敛，拟合失败。")
        except Exception as e:
            print(f"  为商家 {merchant_id} 建模时发生错误: {e}")

    return results
    '''
    # 1. 计算每个迭代中该商家的订单数量
        # 新的计算方法通过将商家的总订单数除以全局迭代次数，
        # 考虑了该商家没有订单的所有迭代。
        avg_orders_per_iteration = len(df_merchant) / total_iterations if total_iterations > 0 else 0
        print(f"  平均每次迭代订单数: {avg_orders_per_iteration:.2f}")
        
        # 2. 聚合数据 (Binning) - 但这次我们计算的是订单密度
        bin_width = 1.0  # 使用更小的分组宽度以提高精度
        max_dist = df_merchant['distance'].max()
        bins = np.arange(0, max_dist + bin_width, bin_width)
        print(f"  按距离分组，最大距离: {max_dist:.2f} km, 分组宽度: {bin_width} km")
        
        df_merchant['distance_bin'] = pd.cut(df_merchant['distance'], bins=bins, right=False)
        aggregated_data = df_merchant.groupby('distance_bin', observed=True).size().reset_index(name='order_count')
        
        # 将订单数量归一化到单次迭代
        aggregated_data['order_count'] = aggregated_data['order_count'] / total_iterations
        
        # 将category类型的distance_bin转换为数值类型的distance_mid
        aggregated_data['distance_mid'] = aggregated_data['distance_bin'].apply(lambda b: b.mid).astype(float)
        print(f"  聚合后得到 {len(aggregated_data)} 个距离分组")
        
        # 3. 计算每个距离区间的面积以得到订单密度
        # 每个距离区间的环面积 = π*(r2^2 - r1^2)，其中r1和r2是区间的上下限
        # 直接提取区间的左右边界值，避免使用apply方法
        right_values = np.array([float(interval.right) for interval in aggregated_data['distance_bin']])
        left_values = np.array([float(interval.left) for interval in aggregated_data['distance_bin']])
        aggregated_data['area'] = np.pi * (right_values**2 - left_values**2)
        
        # 计算订单密度 (每平方公里的订单数)
        aggregated_data['order_density'] = aggregated_data['order_count'] / aggregated_data['area']
        
        # 保存聚合数据用于检查
        agg_data_file = f'aggregated_data_merchant_{merchant_id}.csv'
        aggregated_data.to_csv(agg_data_file, index=False)
        print(f"  已将聚合数据保存到 {agg_data_file} 用于检查")
        
        # 4. 拟合订单密度模型
        print("  开始拟合订单密度-距离模型...")

        x_data = aggregated_data['distance_mid'].values
        y_data = aggregated_data['order_count'].values

        # 检查数据点数量是否足够进行拟合
        if len(x_data) < 2:
            print("  聚合后的数据点不足以进行拟合，跳过此商家。")
            continue

        try:
            print("  使用非线性最小二乘法拟合模型...")
            # 使用更合理的初始猜测
            valid_mask = (y_data > 0) & (x_data > 0)
            if np.sum(valid_mask) < 2:
                print("  有效数据点不足，跳过此商家。")
                continue
                
            x_valid = x_data[valid_mask]
            y_valid = y_data[valid_mask]
            
            # 使用对数线性回归获取更好的初始参数
            # 模型: y = α' * x * exp(β*x)
            # 取对数: ln(y/x) = ln(α') + β*x
            log_ratio = np.log(y_valid / x_valid)
            X = np.vstack([np.ones(len(x_valid)), x_valid]).T
            coeffs = np.linalg.lstsq(X, log_ratio, rcond=None)[0]
            alpha_prime_guess = np.exp(coeffs[0])
            beta_guess = coeffs[1]
            
            # 使用非线性最小二乘拟合
            popt, pcov = curve_fit(
                order_model,
                x_valid, 
                y_valid, 
                p0=[alpha_prime_guess, beta_guess],
                maxfev=5000
            )

            estimated_alpha_prime, estimated_beta = popt

            # 计算R-squared
            y_pred = order_model(x_valid, *popt)
            ss_res = np.sum((y_valid - y_pred) ** 2)
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            print(f"  拟合完成: alpha' = {estimated_alpha_prime:.4f}, beta = {estimated_beta:.4f}, R^2 = {r_squared:.4f}")

            # 5. 保存结果
            results[merchant_id] = {
                'alpha_prime': estimated_alpha_prime,
                'beta': estimated_beta,
                'r_squared': r_squared,
                'avg_orders_per_iteration': avg_orders_per_iteration,
                'aggregated_data': aggregated_data
            }
            print(f"  商家 {merchant_id} 的模型参数已保存")
        except RuntimeError as e:
            print(f"  商家 {merchant_id} 的数据无法收敛，拟合失败: {e}")
        except Exception as e:
            print(f"  为商家 {merchant_id} 建模时发生错误: {e}")

    return results


# --- 运行主程序 ---
try:
    # 直接使用默认路径
    # csv_file_path = "/Users/xiao/Documents/博二上/博二上课题/deliveryRadius_2Areas_2.14_6eventDriven/all_iterations_orders.csv"
    # csv_file_path = "/Users/xiao/Documents/博二上/博二上课题/deliveryRadius_2Areas_2.14_6eventDriven_2hrs/all_iterations_orders_sameRadii_2hrs2.csv" 
    csv_file_path = "/Users/xiao/Documents/博二上/博二上课题/deliveryRadius_2Areas_2.14_6eventDriven_2hrs/all_iterations_orders_Fixed2hrs.csv" #12:00-14:00
    # csv_file_path = "/Users/xiao/Documents/博二上/博二上课题/deliveryRadius_2Areas_2.14_6eventDriven_2hrs/all_iterations_orders_Fixed4hrs.csv" #10:00-14:00
    

    print(f"正在处理文件: {csv_file_path}")
    
    # 筛选12:00-14:00的数据 (已知startTime = 1665964800对应08:00)
    # start_time = 1665972000  # 10:00
    # start_time = 1665972000 + 1800  # 10:30
    # start_time = 1665972000 + 1800*2  # 11:00
    # start_time = 1665972000 + 1800*3  # 11:30
    start_time = 1665979200  # 12:00
    # start_time = 1665979200 + 1800  # 12:30
    # start_time = 1665979200 + 1800*2  # 13:00
    # start_time = 1665979200 + 1800*3  # 13:30


    # start_time = 1665979200  # 12:00
    # start_time= 1665979200 + 900 #12:15
    # start_time= 1665979200 + 900*2 #12:30
    # start_time= 1665979200 + 900*3 #12:45
    # start_time= 1665979200 + 900*4 #13:00
    # start_time= 1665979200 + 900*5 #13:15
    # start_time= 1665979200 + 900*6 #13:30
    # start_time= 1665979200 + 900*7 #13:45
    # end_time = start_time + 3600*2      # 14:00
    end_time = start_time + 1800      # 30min
    # end_time = start_time + 900 #15min
    
    # 修改run_customer_estimation_for_all_merchants函数以支持时间筛选
    merchant_results = run_customer_estimation_for_all_merchants(csv_file_path, start_time, end_time)
    
    if merchant_results is None:
        print("程序无法处理指定的文件，请检查文件路径和格式是否正确。")
        exit(1)
    elif len(merchant_results) == 0:
        print("未能从文件中提取任何有效的商家数据，请检查数据格式。")
        exit(1)
except Exception as e:
    print(f"程序运行时发生错误: {e}")
    exit(1)

# --- 可视化所有商家的结果 ---
if merchant_results:
    print("\n开始生成结果汇总和可视化...")
    # 创建结果汇总表格
    print("正在汇总所有商家的参数结果...")
    summary_data = []
    for merchant_id, res in merchant_results.items():
        summary_data.append({
            'merchant_id': merchant_id,
            'alpha_prime': res['alpha_prime'],
            'beta': res['beta'],
            'r_squared': res['r_squared'],
            'avg_orders': res['avg_orders_per_iteration']
        })
    
    summary_df = pd.DataFrame(summary_data)
    # 按R²值排序，方便查看拟合效果最好的商家
    summary_df_sorted = summary_df.sort_values('r_squared', ascending=False)
    print("\n--- 所有商家拟合参数汇总 (按R²值降序排列) ---")
    print(summary_df_sorted.to_string(index=False))
    
    # 将结果保存到CSV文件
    print("正在将参数结果保存到CSV文件...")
    summary_df.to_csv('merchant_model_parameters.csv', index=False)
    print("参数结果已保存到 merchant_model_parameters.csv")
    
    # 为每个商家创建可视化图表
    print("\n开始为每个商家生成订单-距离关系图...")
    # 统一横纵坐标范围和刻度
    # 计算全局最大距离和全局订单数量/曲线最大值
    global_x_max = max(res['aggregated_data']['distance_mid'].max() for res in merchant_results.values())
    global_y_data_max = max(res['aggregated_data']['order_count'].max() for res in merchant_results.values())
    x_smooth_global = np.linspace(0, global_x_max, 200)
    global_y_model_max = max(
        order_model(x_smooth_global, res['alpha_prime'], res['beta']).max()
        for res in merchant_results.values()
    )
    global_y_max = max(global_y_data_max, global_y_model_max)
    # 统一刻度间隔
    x_tick = 1.0  # 与分箱宽度一致
    def nice_step(max_val, target_ticks=6):
        if max_val <= 0:
            return 1.0
        raw = max_val / target_ticks
        exp = np.floor(np.log10(raw))
        frac = raw / (10**exp)
        if frac <= 1:
            nice = 1
        elif frac <= 2:
            nice = 2
        elif frac <= 5:
            nice = 5
        else:
            nice = 10
        return nice * (10**exp)
    y_tick = nice_step(global_y_max)
    # y_tick = 1.0
    # 为每个商家生成统一坐标的图
    total_merchants = len(merchant_results)
    for idx, (merchant_id, res) in enumerate(merchant_results.items(), 1):
        print(f"正在生成商家 {merchant_id} 的可视化图表 ({idx}/{total_merchants})...")
        agg_data = res['aggregated_data']
        
        plt.figure(figsize=(10, 6))
        # 绘制实际数据点
        plt.scatter(agg_data['distance_mid'], agg_data['order_count'], color='blue', s=50, alpha=0.7,
                    label='Actual Order Count')
        
        # 绘制拟合曲线（使用统一的横坐标范围）
        x_smooth = x_smooth_global
        y_smooth = order_model(x_smooth, res['alpha_prime'], res['beta'])
        plt.plot(x_smooth, y_smooth, color='red', linewidth=2,
                 label=f'Fitted Curve: α={res["alpha_prime"]:.2f}, β={res["beta"]:.4f}')
        
        # 统一坐标范围与刻度
        plt.xlim(0, global_x_max)
        plt.ylim(0, global_y_max * 1.05)
        plt.xticks(np.arange(0, global_x_max + 1e-9, x_tick))
        plt.yticks(np.arange(0, global_y_max * 1.05 + 1e-9, y_tick))
        # plt.xlim(0, global_x_max)
        # plt.ylim(0, 5.533333 * 1.05)
        # plt.xticks(np.arange(0, global_x_max + 1e-9, x_tick))
        # plt.yticks(np.arange(0, 5.533333 * 1.05 + 1e-9, y_tick))
        
        plt.title(f'Merchant {merchant_id} Order Density (R²={res["r_squared"]:.3f})', fontsize=22)
        plt.xlabel('Distance (km)', fontsize=22)
        plt.ylabel('Average # of Orders', fontsize=22)
        plt.legend(loc='upper right', fontsize=18, markerscale=1.5, frameon=True)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        file_name = f'merchant_{merchant_id}_order_distribution.png'
        plt.savefig(file_name)
        plt.close()
        print(f"  已保存图表到文件: {file_name}")
    
    print(f"所有商家的可视化结果已保存为PNG文件。")
    
    # 绘制所有商家的beta值比较图
    print("\n正在生成商家距离衰减系数(β)比较图...")
    plt.figure(figsize=(12, 8))
    summary_df_sorted_beta = summary_df.sort_values('beta')
    plt.bar([str(x) for x in summary_df_sorted_beta['merchant_id']], 
            summary_df_sorted_beta['beta'], color='orange')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Merchant β Comparison', fontsize=16)
    plt.xlabel('Merchant ID', fontsize=12)
    plt.ylabel('β (distance attenuation coefficients)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    beta_file = 'merchant_beta_comparison.png'
    plt.savefig(beta_file)
    plt.close()
    print(f"已保存商家距离衰减系数比较图到文件: {beta_file}")
    
    # 绘制alpha_prime和beta的散点图，分析两个参数之间的关系
    print("\n正在生成商家参数分布散点图...")
    plt.figure(figsize=(10, 8))
    plt.scatter(summary_df['alpha_prime'], summary_df['beta'], s=80, c=summary_df['r_squared'], 
               cmap='viridis', alpha=0.8)
    
    # 添加商家ID标签
    for i, row in summary_df.iterrows():
        plt.annotate(row['merchant_id'], 
                    (row['alpha_prime'], row['beta']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.colorbar(label='R²')
    plt.title('Merchant Parameter Distribution', fontsize=16)
    plt.xlabel('α\'', fontsize=12)
    plt.ylabel('β (distance attenuation coefficients)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    param_file = 'merchant_parameter_distribution.png'
    plt.savefig(param_file)
    plt.close()
    print(f"已保存商家参数分布图到文件: {param_file}")
    
    print("\n所有分析和可视化任务已完成！")
else:
    print("未能获取任何商家的拟合结果。")

'''
# --- 可视化所有商家的结果 ---
if merchant_results:
    print("\n开始生成结果汇总和可视化...")
    # 创建结果汇总表格
    print("正在汇总所有商家的参数结果...")
    summary_data = []
    for merchant_id, res in merchant_results.items():
        summary_data.append({
            'merchant_id': merchant_id,
            'alpha_prime': res['alpha_prime'],
            'beta': res['beta'],
            'r_squared': res['r_squared']
        })
    
    summary_df = pd.DataFrame(summary_data)
    # 按R²值排序，方便查看拟合效果最好的商家
    summary_df_sorted = summary_df.sort_values('r_squared', ascending=False)
    print("\n--- 所有商家拟合参数汇总 (按R²值降序排列) ---")
    print(summary_df_sorted.to_string(index=False))
    
    # 将结果保存到CSV文件
    print("正在将参数结果保存到CSV文件...")
    summary_df.to_csv('merchant_model_parameters.csv', index=False)
    print("参数结果已保存到 merchant_model_parameters.csv")
    
    # 为每个商家创建可视化图表
    print("\n开始为每个商家生成订单-距离关系图...")
    total_merchants = len(merchant_results)
    for idx, (merchant_id, res) in enumerate(merchant_results.items(), 1):
        print(f"正在生成商家 {merchant_id} 的可视化图表 ({idx}/{total_merchants})...")
        agg_data = res['aggregated_data']
        
        plt.figure(figsize=(10, 6))
        # 绘制实际数据点
        plt.scatter(agg_data['distance_mid'], agg_data['order_count'], color='blue', s=50, alpha=0.7,
                  label=f'Actual Data')
        
        # 绘制拟合曲线
        x_smooth = np.linspace(0, agg_data['distance_mid'].max(), 200)
        plt.plot(x_smooth, order_model(x_smooth, res['alpha_prime'], res['beta']), color='red', linewidth=2,
                 label=f'Fitted Curve: α\'={res["alpha_prime"]:.2f}, β={res["beta"]:.4f}')
        
        plt.title(f'Merchant {merchant_id} order distribution (R²={res["r_squared"]:.3f})', fontsize=16)
        plt.xlabel('Distance (km)', fontsize=12)
        plt.ylabel('Order Count', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        file_name = f'merchant_{merchant_id}_order_distribution.png'
        plt.savefig(file_name)
        plt.close()
        print(f"  已保存图表到文件: {file_name}")
    
    print(f"所有商家的可视化结果已保存为PNG文件。")
    
    # 绘制所有商家的beta值比较图
    print("\n正在生成商家距离衰减系数(β)比较图...")
    plt.figure(figsize=(12, 8))
    summary_df_sorted_beta = summary_df.sort_values('beta')
    plt.bar(summary_df_sorted_beta['merchant_id'], summary_df_sorted_beta['beta'], color='orange')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Merchant β Comparison', fontsize=16)
    plt.xlabel('Merchant ID', fontsize=12)
    plt.ylabel('β (distance attenuation coefficients)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    beta_file = 'merchant_beta_comparison.png'
    plt.savefig(beta_file)
    plt.close()
    print(f"已保存商家距离衰减系数比较图到文件: {beta_file}")
    
    # 绘制alpha_prime和beta的散点图，分析两个参数之间的关系
    print("\n正在生成商家参数分布散点图...")
    plt.figure(figsize=(10, 8))
    plt.scatter(summary_df['alpha_prime'], summary_df['beta'], s=80, c=summary_df['r_squared'], 
               cmap='viridis', alpha=0.8)
    
    # 添加商家ID标签
    for i, row in summary_df.iterrows():
        plt.annotate(row['merchant_id'], 
                    (row['alpha_prime'], row['beta']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.colorbar(label='R²')
    plt.title('Merchant parameter distribution: the relationship between α\' and β', fontsize=16)
    plt.xlabel('α\' ', fontsize=12)
    plt.ylabel('β (distance attenuation coefficients)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    param_file = 'merchant_parameter_distribution.png'
    plt.savefig(param_file)
    plt.close()
    print(f"已保存商家参数分布图到文件: {param_file}")
    
    print("\n所有分析和可视化任务已完成！")
else:
    print("未能获取任何商家的拟合结果。")
    '''