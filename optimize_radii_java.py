#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配送半径动态规划优化器 - 

目标：为R个商家分别设置配送半径ρ_r，最大化总潜在订单数
约束：预计配送时间L - 承诺配送时间L^o ≤ ε
"""

import json
import time
import os
import sys
import argparse

# 尝试导入可选依赖
try:
    import joblib
    import pandas as pd
    import numpy as np
    from scipy.integrate import quad
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"警告: 部分依赖未安装: {e}")
    print("将使用降级模式，返回默认半径配置")

# 尝试导入Gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("警告: Gurobi未安装，将使用动态规划求解器。")
    print("如需更快的求解速度，请安装Gurobi: pip install gurobipy")
    print("并获取有效的Gurobi许可证。")

# --- 1. 全局配置 ---
class Config:
    # 文件路径
    MODEL_PATH = '/Users/xiao/Documents/博二上/博二上课题/pythonProject/model_gradient_boosting_fixed.pkl'
    SCALER_PATH = '/Users/xiao/Documents/博二上/博二上课题/pythonProject/scaler_fixed.pkl'
    MERCHANT_PARAMS_PATH = '/Users/xiao/Documents/博二上/博二上课题/pythonProject/merchant_model_parameters.csv'
    BEST_FEATURES_PATH = '/Users/xiao/Documents/博二上/博二上课题/pythonProject/best_features_fixed.json'  
    # MODEL_PATH = '/Users/xiao/Documents/博二上/博二上课题/pythonProject/12:30-13:00/ML_Model/model_ridge_regression_fixed.pkl'
    # SCALER_PATH = '/Users/xiao/Documents/博二上/博二上课题/pythonProject/12:30-13:00/ML_Model/scaler_fixed.pkl'
    # MERCHANT_PARAMS_PATH = '/Users/xiao/Documents/博二上/博二上课题/pythonProject/12:30-13:00/merchant_model_parameters.csv'
    # BEST_FEATURES_PATH = '/Users/xiao/Documents/博二上/博二上课题/pythonProject/12:30-13:00/ML_Model/best_features_fixed.json'  
    NUM_MERCHANTS = 5  # R = 5
    L_PROMISED = 50.0  # L^o = 50分钟 (承诺配送时间)
    EPSILON = 20.0     # ε = 20分钟 (容忍度)
    MAX_L = L_PROMISED + EPSILON  # 最大允许配送时间 = 60分钟
    
    # 半径搜索参数
    RADIUS_MIN = 2.0
    RADIUS_MAX = 10.0
    RADIUS_STEP = 1.0  # 搜索步长，值越小越精确，但计算量越大

# 新增：根据决策时间选择模型/特征文件路径
# BASE_DIR = '/Users/xiao/Documents/博二上/博二上课题/pythonProject'         #2hrs30min
# BASE_DIR = '/Users/xiao/Documents/博二上/博二上课题/pythonProject/15min'         #2hrs15min
BASE_DIR = '/Users/xiao/Documents/博二上/博二上课题/pythonProject/4hrs30min'

# 15分钟时间窗口（注释掉，以便将来可以恢复）
# TIME_SLOTS = [
#     '12:00-12:15',
#     '12:15-12:30',
#     '12:30-12:45',
#     '12:45-13:00',
#     '13:00-13:15',
#     '13:15-13:30',
#     '13:30-13:45',
#     '13:45-14:00',
# ]

# 30分钟时间窗口，从10:00到14:00
TIME_SLOTS = [
    '10:00-10:30',
    '10:30-11:00',
    '11:00-11:30',
    '11:30-12:00',
    '12:00-12:30',
    '12:30-13:00',
    '13:00-13:30',
    '13:30-14:00',
]

def _slot_label_from_index(idx):
    if idx < 0 or idx >= len(TIME_SLOTS):
        return None
    return TIME_SLOTS[idx]

def _override_config_paths_for_slot(slot_label):
    """根据时间窗口标签覆盖 Config 中的文件路径"""
    if not slot_label:
        return
    slot_dir = os.path.join(BASE_DIR, slot_label)
    model_dir = os.path.join(slot_dir, 'ML_Model')

    Config.MODEL_PATH = os.path.join(model_dir, 'model_gradient_boosting_fixed.pkl')
    Config.SCALER_PATH = os.path.join(model_dir, 'scaler_fixed.pkl')
    Config.MERCHANT_PARAMS_PATH = os.path.join(slot_dir, 'merchant_model_parameters.csv')
    Config.BEST_FEATURES_PATH = os.path.join(model_dir, 'best_features_fixed.json')

def _select_slot_by_epoch(base_epoch, decision_time):
    """根据 Unix 时间戳选择对应的时间窗口标签"""
    try:
        diff = int(decision_time) - int(base_epoch)
    except Exception:
        return None
    if diff < 0:
        return _slot_label_from_index(0)
    # idx = diff // 900  # 每 900 秒（15分钟）一个窗口
    idx = diff // 1800  # 每 1800 秒（30分钟）一个窗口
    if idx >= len(TIME_SLOTS):
        idx = len(TIME_SLOTS) - 1
    return _slot_label_from_index(int(idx))

def load_dependencies():
    """加载模型、缩放器和商家参数"""
    if not DEPENDENCIES_AVAILABLE:
        print("依赖不可用，无法加载模型文件")
        return None
        
    try:
        print("正在加载依赖文件...")
        
        dependencies = {}
        
        # 加载模型
        if os.path.exists(Config.MODEL_PATH):
            dependencies['model'] = joblib.load(Config.MODEL_PATH)
            print(f"✓ 模型已加载: {Config.MODEL_PATH}")
        else:
            print(f"模型文件未找到: {Config.MODEL_PATH}")
            return None
        
        # 加载缩放器
        if os.path.exists(Config.SCALER_PATH):
            dependencies['scaler'] = joblib.load(Config.SCALER_PATH)
            print(f"✓ 缩放器已加载: {Config.SCALER_PATH}")
        else:
            print(f"缩放器文件未找到: {Config.SCALER_PATH}")
            return None
        
        # 加载商家参数
        if os.path.exists(Config.MERCHANT_PARAMS_PATH):
            dependencies['merchant_params'] = pd.read_csv(Config.MERCHANT_PARAMS_PATH)
            print(f"✓ 商家参数已加载: {Config.MERCHANT_PARAMS_PATH}")
        else:
            print(f"商家参数文件未找到: {Config.MERCHANT_PARAMS_PATH}")
            return None
    
        # 加载最佳特征
        if os.path.exists(Config.BEST_FEATURES_PATH):
            with open(Config.BEST_FEATURES_PATH, 'r') as f:
                dependencies['best_features'] = json.load(f)
            print(f"✓ 最佳特征已加载: {Config.BEST_FEATURES_PATH}")
        else:
            print(f"最佳特征文件未找到: {Config.BEST_FEATURES_PATH}")
            return None
        
        print("所有依赖文件加载完成。\n")
        return dependencies
    except Exception as e:
        print(f"加载依赖文件时出错: {e}")
        return None

# --- 2. 订单计算和特征准备 ---
def order_model(x, alpha_prime, beta):
    """订单选择模型"""
    return alpha_prime * x * np.exp(beta * x)

def calculate_orders(radius, alpha_prime, beta):
    """计算给定半径下的潜在订单数"""
    result, _ = quad(order_model, 0, radius, args=(alpha_prime, beta))
    return result

def prepare_features_fixed(radii, orders, capacity_info, merchant_ids, best_features, merchant_params):
    """
    根据给定的半径、订单数和容量准备特征（已更新为使用 capacity_info）
    
    参数:
    - radii: 各商家的半径列表
    - orders: 各商家的订单数列表  
    - capacity_info: 包含司机容量统计信息的字典
        { 'total': T, 'mean': M, 'std': S, 'max': X, 'min': N, 'count': C }
    - merchant_ids: 商家ID列表
    - best_features: 最佳特征列表
    - merchant_params: 商家参数DataFrame
    
    返回:
    - DataFrame: 包含所有特征的单行数据框
    """
    feature_dict = {}
    
    # 1. 基础半径特征
    for i, merchant_id in enumerate(merchant_ids):
        feature_name = f'radius_{merchant_id}'
        if feature_name in best_features:
            feature_dict[feature_name] = radii[i]
    
    # 2. 容量相关特征
    # total_capacity = capacity_info['total']
    num_drivers = capacity_info['count']

    if 'total_capacity' in best_features:
        feature_dict['total_capacity'] = capacity_info['total']
    
    # 估算司机数量（假设每个司机平均容量为30）
    # num_drivers = max(1, int(total_capacity / 10))
    
    if 'avg_capacity_per_driver' in best_features:
        # feature_dict['avg_capacity_per_driver'] = total_capacity / num_drivers
        feature_dict['avg_capacity_per_driver'] = capacity_info['mean']
    
    # 假设司机容量的标准差（简化处理）
    if 'capacity_std' in best_features:
        # feature_dict['capacity_std'] = total_capacity * 0.1  # 假设10%的变异系数
        feature_dict['capacity_std'] = capacity_info['std']
    
    if 'max_driver_capacity' in best_features:
        # feature_dict['max_driver_capacity'] = total_capacity / num_drivers * 1.2  # 假设最大容量是平均的1.2倍
        feature_dict['max_driver_capacity'] = capacity_info['max']
    
    if 'min_driver_capacity' in best_features:
        # feature_dict['min_driver_capacity'] = total_capacity / num_drivers * 0.8  # 假设最小容量是平均的0.8倍
        feature_dict['min_driver_capacity'] = capacity_info['min']
    
    # 3. 订单相关特征
    for i, merchant_id in enumerate(merchant_ids):
        # 基础订单数
        feature_name = f'orders_{merchant_id}'
        if feature_name in best_features:
            feature_dict[feature_name] = orders[i]
        
        # 订单密度 = 订单数 / 覆盖面积
        density_feature = f'order_density_{merchant_id}'
        if density_feature in best_features:
            coverage_area = np.pi * radii[i] ** 2
            feature_dict[density_feature] = orders[i] / coverage_area if coverage_area > 0 else 0
        
        # 半径效率 = 积分值 / 半径
        efficiency_feature = f'radius_efficiency_{merchant_id}'
        if efficiency_feature in best_features:
            # 从merchant_params获取参数
            merchant_row = merchant_params[merchant_params['merchant_id'].str.strip() == merchant_id]
            if not merchant_row.empty:
                alpha_prime = merchant_row['alpha_prime'].iloc[0]
                beta = merchant_row['beta'].iloc[0]
                # 计算积分
                integral, _ = quad(lambda x: order_model(x, alpha_prime, beta), 0, radii[i])
                feature_dict[efficiency_feature] = integral / radii[i] if radii[i] > 0 else 0
            else:
                feature_dict[efficiency_feature] = 0
    
    # 4. 聚合特征
    if 'total_potential_orders' in best_features:
        feature_dict['total_potential_orders'] = sum(orders)
    
    # 加权订单总数（使用半径作为权重）
    if 'total_weighted_orders' in best_features:
        weighted_orders = sum(orders[i] * radii[i] for i in range(len(orders)))
        feature_dict['total_weighted_orders'] = weighted_orders
    
    # 5. 半径统计特征
    if 'avg_radius' in best_features:
        feature_dict['avg_radius'] = np.mean(radii)
    
    if 'std_radius' in best_features:
        feature_dict['std_radius'] = np.std(radii)
    
    if 'max_radius' in best_features:
        feature_dict['max_radius'] = np.max(radii)
    
    if 'min_radius' in best_features:
        feature_dict['min_radius'] = np.min(radii)
    
    if 'radius_range' in best_features:
        feature_dict['radius_range'] = np.max(radii) - np.min(radii)
    
    if 'radius_cv' in best_features:
        mean_radius = np.mean(radii)
        feature_dict['radius_cv'] = np.std(radii) / mean_radius if mean_radius > 0 else 0
    
    # 6. 订单密度统计特征
    order_densities = []
    for i in range(len(orders)):
        coverage_area = np.pi * radii[i] ** 2
        density = orders[i] / coverage_area if coverage_area > 0 else 0
        order_densities.append(density)
    
    if 'avg_order_density' in best_features:
        feature_dict['avg_order_density'] = np.mean(order_densities)
    
    if 'std_order_density' in best_features:
        feature_dict['std_order_density'] = np.std(order_densities)
    
    if 'max_order_density' in best_features:
        feature_dict['max_order_density'] = np.max(order_densities)
    
    if 'total_coverage_area' in best_features:
        feature_dict['total_coverage_area'] = np.pi * sum(r**2 for r in radii)
    
    # 7. 交互特征
    total_orders = sum(orders)
    total_capacity = capacity_info['total']
    total_coverage = feature_dict.get('total_coverage_area', np.pi * sum(r**2 for r in radii))
    
    if 'orders_capacity_ratio' in best_features:
        feature_dict['orders_capacity_ratio'] = total_orders / total_capacity if total_capacity > 0 else 0
    
    if 'orders_per_driver' in best_features:
        feature_dict['orders_per_driver'] = total_orders / num_drivers if num_drivers > 0 else 0
    
    if 'demand_supply_imbalance' in best_features:
        feature_dict['demand_supply_imbalance'] = (total_orders - total_capacity) / total_capacity if total_capacity > 0 else 0
    
    if 'coverage_efficiency' in best_features:
        feature_dict['coverage_efficiency'] = total_orders / total_coverage if total_coverage > 0 else 0
    
    if 'weighted_orders_capacity_ratio' in best_features:
        weighted_orders = feature_dict.get('total_weighted_orders', sum(orders[i] * radii[i] for i in range(len(orders))))
        feature_dict['weighted_orders_capacity_ratio'] = weighted_orders / total_capacity if total_capacity > 0 else 0
    
    if 'capacity_per_coverage' in best_features:
        feature_dict['capacity_per_coverage'] = total_capacity / total_coverage if total_coverage > 0 else 0
    
    if 'driver_efficiency' in best_features:
        avg_capacity = capacity_info['mean']
        feature_dict['driver_efficiency'] = total_orders / (num_drivers * avg_capacity) if (num_drivers * avg_capacity) > 0 else 0
    
    # 8. 对数变换特征
    if 'log_capacity_per_coverage' in best_features:
        capacity_per_coverage = feature_dict.get('capacity_per_coverage', 0)
        feature_dict['log_capacity_per_coverage'] = np.log1p(max(0, capacity_per_coverage))
    
    # 创建DataFrame，确保特征顺序与best_features一致
    final_features = {}
    for feature in best_features:
        if feature in feature_dict:
            final_features[feature] = feature_dict[feature]
        else:
            final_features[feature] = 0  # 缺失特征用0填充
    
    return pd.DataFrame([final_features])

def predict_delivery_time(features_df, model, scaler=None):
    """使用加载的模型预测配送时间（不缩放）"""
    # 不进行任何特征缩放，直接使用原始特征进行预测
    prediction = model.predict(features_df)
    return prediction[0]

# --- 3. 求解器 ---
class DPSolverFixed:
    def __init__(self, dependencies, capacity_info):
        self.model = dependencies['model']
        self.scaler = dependencies['scaler']
        self.merchant_params = dependencies['merchant_params'].head(Config.NUM_MERCHANTS)
        self.best_features = dependencies['best_features']
        self.capacity_info = capacity_info
        
        # 清理商家ID
        self.merchant_ids = self.merchant_params['merchant_id'].str.strip().tolist()
        
        # 记忆化缓存
        self.memo = {}
        
        # 定义半径的离散化搜索空间
        self.radius_options = np.arange(
            Config.RADIUS_MIN, 
            Config.RADIUS_MAX + Config.RADIUS_STEP, 
            Config.RADIUS_STEP
        )
        self.total_combinations = len(self.radius_options) ** Config.NUM_MERCHANTS
        print(f"半径搜索步长: {Config.RADIUS_STEP}, 每个商家有 {len(self.radius_options)} 个选择。")
        print(f"将要处理的商家ID: {self.merchant_ids}")
        print(f"总计需要探索 {self.total_combinations:,.0f} 种组合。")
        print(f"司机容量信息 (总数: {self.capacity_info['count']}, 总容量: {self.capacity_info['total']:.2f})")
        print("-" * 30)
    
    def solve(self):
        """启动求解过程"""
        start_time = time.time()
        
        print("开始动态规划求解...")
        max_orders, optimal_radii, optimal_orders, predicted_L = self._solve_recursive(0, [])
        
        end_time = time.time()
        print(f"求解完成，耗时: {end_time - start_time:.2f} 秒")
        print(f"探索的状态数: {len(self.memo)}")
        
        return max_orders, optimal_radii, optimal_orders, predicted_L
    
    def _solve_recursive(self, merchant_index, current_radii):
        """递归求解动态规划"""
        # 基础情况：所有商家都已分配半径
        if merchant_index == Config.NUM_MERCHANTS:
            # 计算每个商家的订单数
            orders = []
            for i, radius in enumerate(current_radii):
                merchant_row = self.merchant_params.iloc[i]
                alpha_prime = merchant_row['alpha_prime']
                beta = merchant_row['beta']
                order_count = calculate_orders(radius, alpha_prime, beta)
                orders.append(order_count)
            
            # 准备特征并预测配送时间
            try:
                features_df = prepare_features_fixed(
                    current_radii, orders, self.capacity_info, self.merchant_ids, 
                    self.best_features, self.merchant_params
                )
                predicted_L = predict_delivery_time(features_df, self.model, self.scaler)
                
                # 检查约束条件
                if predicted_L <= Config.MAX_L:
                    total_orders = sum(orders)
                    return total_orders, current_radii.copy(), orders.copy(), predicted_L
                else:
                    return -1, None, None, predicted_L  # 不满足约束
            except Exception as e:
                print(f"预测错误: {e}")
                return -1, None, None, float('inf')
        
        # 记忆化检查
        state_key = tuple(current_radii)
        if state_key in self.memo:
            return self.memo[state_key]
        
        # 尝试当前商家的所有可能半径
        best_orders = -1
        best_radii = None
        best_orders_list = None
        best_predicted_L = float('inf')
        
        for radius in self.radius_options:
            new_radii = current_radii + [radius]
            orders, radii_result, orders_result, predicted_L = self._solve_recursive(
                merchant_index + 1, new_radii
            )
            
            if orders > best_orders and radii_result is not None:
                best_orders = orders
                best_radii = radii_result
                best_orders_list = orders_result
                best_predicted_L = predicted_L
        
        # 存储结果到记忆化缓存
        self.memo[state_key] = (best_orders, best_radii, best_orders_list, best_predicted_L)
        return best_orders, best_radii, best_orders_list, best_predicted_L


class GurobiSolver:
    """使用Gurobi求解器优化商家配送半径"""
    
    def __init__(self, dependencies, capacity_info):
        self.model = dependencies['model']
        self.scaler = dependencies['scaler']
        self.merchant_params = dependencies['merchant_params'].head(Config.NUM_MERCHANTS)
        self.best_features = dependencies['best_features']
        self.capacity_info = capacity_info
        
        # 清理商家ID
        self.merchant_ids = self.merchant_params['merchant_id'].str.strip().tolist()
        
        # 定义半径的离散化搜索空间
        self.radius_options = np.arange(
            Config.RADIUS_MIN, 
            Config.RADIUS_MAX + Config.RADIUS_STEP, 
            Config.RADIUS_STEP
        )
        self.num_radius_options = len(self.radius_options)
        
        print(f"半径搜索步长: {Config.RADIUS_STEP}, 每个商家有 {self.num_radius_options} 个选择。")
        print(f"将要处理的商家ID: {self.merchant_ids}")
        print(f"司机容量信息 (总数: {self.capacity_info['count']}, 总容量: {self.capacity_info['total']:.2f})")
        print("-" * 30)
    
    def solve(self):
        """使用Gurobi求解优化问题"""
        if not GUROBI_AVAILABLE:
            raise RuntimeError("Gurobi不可用，无法进行优化。请安装 gurobipy 并配置有效许可证。")
        
        start_time = time.time()
        print("开始Gurobi求解...")
        
        try:
            # 创建Gurobi模型
            m = gp.Model("radius_optimization")
            m.setParam('OutputFlag', 1)  # 显示求解过程
            m.setParam('TimeLimit', 300)  # 设置时间限制（秒）
            
            # 创建变量：x[i,j] = 1 表示商家i选择半径选项j
            x = {}
            for i in range(Config.NUM_MERCHANTS):
                for j in range(self.num_radius_options):
                    x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            
            # 约束：每个商家只能选择一个半径
            for i in range(Config.NUM_MERCHANTS):
                m.addConstr(gp.quicksum(x[i, j] for j in range(self.num_radius_options)) == 1)
            
            # 创建辅助变量：每个商家的半径和订单数
            radii = {}
            orders = {}
            for i in range(Config.NUM_MERCHANTS):
                # 商家i的半径 = 所选半径选项的值
                radii[i] = m.addVar(lb=Config.RADIUS_MIN, ub=Config.RADIUS_MAX, name=f"radius_{i}")
                m.addConstr(radii[i] == gp.quicksum(self.radius_options[j] * x[i, j] for j in range(self.num_radius_options)))
                
                # 计算每个商家的订单数
                merchant_row = self.merchant_params.iloc[i]
                alpha_prime = merchant_row['alpha_prime']
                beta = merchant_row['beta']
                
                # 由于订单数是半径的非线性函数，我们需要使用分段线性近似
                # 为每个可能的半径预计算订单数
                order_values = [calculate_orders(self.radius_options[j], alpha_prime, beta) 
                               for j in range(self.num_radius_options)]
                
                # 商家i的订单数 = 所选半径对应的订单数
                orders[i] = m.addVar(lb=0, name=f"orders_{i}")
                m.addConstr(orders[i] == gp.quicksum(order_values[j] * x[i, j] for j in range(self.num_radius_options)))
            
            # 计算总订单数
            total_orders = m.addVar(name="total_orders")
            m.addConstr(total_orders == gp.quicksum(orders[i] for i in range(Config.NUM_MERCHANTS)))
            
            # 添加配送时间约束
            # 由于配送时间预测是非线性的，我们需要使用回调函数或分段线性近似
            # 这里使用一个简化的方法：采样多个半径组合，检查约束
            
            # 设置目标函数：最大化总订单数
            m.setObjective(total_orders, GRB.MAXIMIZE)
            
            # 添加延迟约束回调
            def delivery_time_callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    # 获取当前解
                    current_radii = []
                    current_orders = []
                    for i in range(Config.NUM_MERCHANTS):
                        radius_val = model.cbGetSolution(radii[i])
                        order_val = model.cbGetSolution(orders[i])
                        current_radii.append(radius_val)
                        current_orders.append(order_val)
                    
                    # 预测配送时间
                    try:
                        features_df = prepare_features_fixed(
                            current_radii, current_orders, self.capacity_info, self.merchant_ids, 
                            self.best_features, self.merchant_params
                        )
                        predicted_L = predict_delivery_time(features_df, self.model, self.scaler)
                        
                        # 如果违反配送时间约束，添加拒绝当前解的约束
                        if predicted_L > Config.MAX_L:
                            # 添加一个约束，排除当前解
                            expr = gp.LinExpr()
                            for i in range(Config.NUM_MERCHANTS):
                                for j in range(self.num_radius_options):
                                    if abs(current_radii[i] - self.radius_options[j]) < 1e-6:
                                        expr.add(x[i, j], -1)
                                    else:
                                        expr.add(x[i, j])
                            model.cbLazy(expr >= 1)
                    except Exception as e:
                        print(f"回调中预测错误: {e}")
            
            # 启用延迟约束
            m.setParam('LazyConstraints', 1)
            
            # 求解模型
            m.optimize(delivery_time_callback)
            
            # 获取最优解
            if m.status == GRB.OPTIMAL:
                optimal_radii = []
                optimal_orders = []
                for i in range(Config.NUM_MERCHANTS):
                    radius_val = radii[i].X
                    order_val = orders[i].X
                    optimal_radii.append(radius_val)
                    optimal_orders.append(order_val)
                
                # 计算最终的配送时间
                features_df = prepare_features_fixed(
                    optimal_radii, optimal_orders, self.capacity_info, self.merchant_ids, 
                    self.best_features, self.merchant_params
                )
                predicted_L = predict_delivery_time(features_df, self.model, self.scaler)
                
                max_orders = sum(optimal_orders)
                
                end_time = time.time()
                print(f"Gurobi求解完成，耗时: {end_time - start_time:.2f} 秒")
                print(f"最优目标值: {max_orders:.2f}")
                print(f"预测配送时间: {predicted_L:.2f}")
                
                return max_orders, optimal_radii, optimal_orders, predicted_L
            else:
                raise RuntimeError(f"Gurobi求解失败，状态码: {m.status}")
                
        except Exception as e:
            raise RuntimeError(f"Gurobi求解错误: {e}")

def manual_test_mode(dependencies, capacity_info=None):
    """手动测试模式，支持手动输入商家半径和司机容量"""
    print("\n--- 手动测试模式 ---")
    
    # 如果没有提供司机容量信息，则手动输入
    if capacity_info is None:
        capacity_info = manual_input_capacity()
    
    print(f"当前司机容量信息 (总数: {capacity_info['count']}, 总容量: {capacity_info['total']:.2f})")
    print("请为每个商家输入配送半径 (1-10):")
    
    merchant_params = dependencies['merchant_params'].head(Config.NUM_MERCHANTS)
    merchant_ids = merchant_params['merchant_id'].str.strip().tolist()
    
    radii = []
    for i, merchant_id in enumerate(merchant_ids):
        while True:
            try:
                radius_input = input(f"商家 {merchant_id} 的半径: ")
                radius = float(radius_input)
                if Config.RADIUS_MIN <= radius <= Config.RADIUS_MAX:
                    radii.append(radius)
                    break
                else:
                    print(f"半径必须在 {Config.RADIUS_MIN} 到 {Config.RADIUS_MAX} 之间。")
            except ValueError:
                print("输入无效，请输入一个数字。")
    
    # 计算订单数
    orders = []
    for i, radius in enumerate(radii):
        merchant_row = merchant_params.iloc[i]
        alpha_prime = merchant_row['alpha_prime']
        beta = merchant_row['beta']
        order_count = calculate_orders(radius, alpha_prime, beta)
        orders.append(order_count)
    
    # 预测配送时间
    try:
        features_df = prepare_features_fixed(radii, orders, capacity_info, merchant_ids, 
                                           dependencies['best_features'], dependencies['merchant_params'])
        predicted_L = predict_delivery_time(features_df, dependencies['model'], dependencies['scaler'])
        
        # 显示结果
        print("\n" + "="*30)
        print("      手动测试结果")
        print("="*30)
        
        result_df = pd.DataFrame({
            '商家ID': merchant_ids,
            '半径 (ρ_r)': [round(r, 2) for r in radii],
            '潜在订单数 (O_r)': [round(o, 2) for o in orders]
        })
        print(result_df.to_string(index=False))
        
        print(f"\n总潜在订单数: {sum(orders):.2f}")
        print(f"预测的平均配送时间: {predicted_L:.2f} 分钟")
        print(f"约束检查: L - L^o = {predicted_L:.2f} - {Config.L_PROMISED} = {predicted_L - Config.L_PROMISED:.2f}")
        
        if predicted_L <= Config.MAX_L:
            print(f"✓ 满足约束条件 (≤ {Config.EPSILON} 分钟)")
        else:
            print(f"✗ 不满足约束条件 (> {Config.EPSILON} 分钟)")
            
    except Exception as e:
        print(f"预测失败: {e}")

def manual_input_capacity():
    """手动输入司机容量信息"""
    print("\n--- 司机容量输入 ---")
    
    while True:
        try:
            num_drivers = int(input("请输入司机数量: "))
            if num_drivers > 0:
                break
            else:
                print("司机数量必须是正整数。")
        except ValueError:
            print("输入无效，请输入一个整数。")
    
    driver_capacities = []
    for i in range(num_drivers):
        while True:
            try:
                capacity = float(input(f"请输入司机 {i+1} 的剩余容量: "))
                if capacity >= 0:
                    driver_capacities.append(capacity)
                    break
                else:
                    print("容量必须是非负数。")
            except ValueError:
                print("输入无效，请输入一个数字。")
    
    cap_array = np.array(driver_capacities)
    
    # 确保至少有一个司机的容量为正
    if np.sum(cap_array) <= 0:
        print("警告: 所有司机的总剩余容量为0，将使用默认值1")
        cap_array = np.array([1.0])
    
    capacity_info = {
        'total': np.sum(cap_array),
        'mean': np.mean(cap_array),
        'std': np.std(cap_array) if len(cap_array) > 1 else 0,
        'max': np.max(cap_array),
        'min': np.min(cap_array),
        'count': len(cap_array)
    }
    
    print(f"司机容量信息已设置: 总数={capacity_info['count']}, 总容量={capacity_info['total']:.2f}")
    return capacity_info

def get_default_radii():
    """返回默认的半径配置"""
    return {
        "1": 5.0,
        "2": 5.0, 
        "3": 5.0,
        "4": 5.0,
        "5": 5.0
    }

def main():
    """主函数，支持命令行参数和交互模式"""
    parser = argparse.ArgumentParser(description='配送半径动态规划优化器')
    parser.add_argument('--capacity', type=float, help='（旧）所有司机的剩余容量之和')
    parser.add_argument('--driver', type=str, action='append', help='每个司机的剩余容量，格式为 ID:Capacity (例如: 1:10)')
    parser.add_argument('--json-output', action='store_true', help='以JSON格式输出结果')
    parser.add_argument('--interactive', action='store_true', help='交互模式')
    parser.add_argument('--manual-test', action='store_true', help='手动测试模式，允许手动输入商家半径和司机容量')
    # 新增：时间参数
    parser.add_argument('--decision-time', type=float, help='Unix 时间戳（秒），用于选择时间窗口（支持科学计数法，如 1.665981E9）')
    # parser.add_argument('--base-epoch', type=float, default=1665979200.0, help='参考起始 Unix 时间戳（默认: 12:00 起点，支持科学计数法）')
    # parser.add_argument('--time-slot', type=str, choices=['12:00-12:30','12:30-13:00','13:00-13:30','13:30-14:00'], help='直接指定时间窗口标签')
    # parser.add_argument('--time-slot', type=str, choices=['12:00-12:15','12:15-12:30','12:30-12:45','12:45-13:00','13:00-13:15','13:15-13:30','13:30-13:45','13:45-14:00'], help='直接指定时间窗口标签')
    parser.add_argument('--base-epoch', type=float, default=1665972000.0, help='参考起始 Unix 时间戳（默认: 10:00 起点，支持科学计数法）')
    parser.add_argument('--time-slot', type=str, choices=['10:00-10:30','10:30-11:00','11:00-11:30','11:30-12:00','12:00-12:30','12:30-13:00','13:00-13:30','13:30-14:00'], help='直接指定时间窗口标签')
    
    args = parser.parse_args()

    # 根据时间参数切换模型/特征文件路径
    try:
        slot_label = None
        if args.time_slot:
            slot_label = args.time_slot
        elif args.decision_time is not None:
            slot_label = _select_slot_by_epoch(int(args.base_epoch), int(args.decision_time))
        if slot_label:
            _override_config_paths_for_slot(slot_label)
            print(f"已根据时间窗口切换路径到: {slot_label}")
    except Exception as e:
        print(f"警告: 时间窗口解析失败，使用默认路径。原因: {e}")
    
    # 加载依赖
    try:
        dependencies = load_dependencies()
        if dependencies is None:
            # 如果加载依赖失败，返回默认半径值
            if args.json_output:
                result = {
                    "success": True,
                    "message": "依赖加载失败，使用默认半径",
                    "radii": get_default_radii(),
                    "total_orders": 0,
                    "predicted_delivery_time": 50.0
                }
                print(json.dumps(result, ensure_ascii=False))
            else:
                print("警告: 加载依赖失败")
                print("使用默认半径配置...")
                default_radii = get_default_radii()
                for merchant_id, radius in default_radii.items():
                    print(f"商家 {merchant_id}: {radius} km")
            return
    except Exception as e:
        # 如果加载依赖失败，返回默认半径值
        if args.json_output:
            result = {
                "success": True,
                "message": f"依赖加载失败，使用默认半径: {str(e)}",
                "radii": get_default_radii(),
                "total_orders": 0,
                "predicted_delivery_time": 50.0
            }
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(f"警告: 加载依赖失败: {e}")
            print("使用默认半径配置...")
            default_radii = get_default_radii()
            for merchant_id, radius in default_radii.items():
                print(f"商家 {merchant_id}: {radius} km")
        return
        
    # 如果是手动测试模式，直接进入手动测试
    if args.manual_test:
        # 不需要预先提供capacity_info，manual_test_mode会自己处理
        manual_test_mode(dependencies)
        return
    
    # 获取司机容量
    capacity_info = None
    
    if args.driver:
        # 优先使用 --driver 参数
        try:
            driver_capacities_list = []
            for driver_arg in args.driver:
                parts = driver_arg.split(':')
                if len(parts) != 2:
                    raise ValueError(f"无效的 --driver 参数格式: {driver_arg}")
                # 司机ID (parts[0]) 在这里不需要，我们只关心容量
                capacity = int(parts[1])
                driver_capacities_list.append(capacity)
            
            if not driver_capacities_list:
                raise ValueError("提供了 --driver 参数，但列表为空")
            
            cap_array = np.array(driver_capacities_list)
            
            # 确保至少有一个司机的容量为正
            if np.sum(cap_array) <= 0:
                raise ValueError("所有司机的总剩余容量为0或负数")
            
            capacity_info = {
                'total': np.sum(cap_array),
                'mean': np.mean(cap_array),
                'std': np.std(cap_array),
                'max': np.max(cap_array),
                'min': np.min(cap_array),
                'count': len(cap_array)
            }
            
        except Exception as e:
            if args.json_output:
                result = {"success": False, "error": f"解析司机容量时出错: {str(e)}", "radii": {}}
                print(json.dumps(result, ensure_ascii=False))
            else:
                print(f"错误: 解析司机容量时出错: {e}")
            sys.exit(1)
            
    elif args.capacity is not None:
        # 回退到 --capacity (通常用于交互模式或旧的测试)
        total_capacity = args.capacity
        if total_capacity <= 0:
            if args.json_output:
                result = {"success": False, "error": "容量必须是正数", "radii": {}}
                print(json.dumps(result, ensure_ascii=False))
            else:
                print("错误: 容量必须是正数")
            sys.exit(1)
        
        # 估算 capacity_info (基于旧逻辑)
        print("警告: 正在使用总容量估算司机统计信息。为获得更准确结果，请使用 --driver 参数。")
        num_drivers = max(1, int(total_capacity / 10)) # 假设平均容量为10
        avg_cap = total_capacity / num_drivers
        capacity_info = {
            'total': total_capacity,
            'mean': avg_cap,
            'std': total_capacity * 0.1,  # 假设10%的变异系数
            'max': avg_cap * 1.2,  # 假设最大容量是平均的1.2倍
            'min': avg_cap * 0.8,  # 假设最小容量是平均的0.8倍
            'count': num_drivers
        }

    elif args.interactive:
        # 交互模式
        print("--- 配送半径动态规划优化器 (修复版本) ---")
        print(f"约束条件: L - L^o ≤ {Config.EPSILON} 分钟 (L^o = {Config.L_PROMISED} 分钟)")
        print(f"半径范围: {Config.RADIUS_MIN} ≤ ρ_r ≤ {Config.RADIUS_MAX}")
        
        while True:
            try:
                driver_capacity_input = input("请输入所有司机的剩余容量之和 (例如: 150): ")
                total_capacity = float(driver_capacity_input)
                if total_capacity > 0:
                    break
                else:
                    print("容量必须是正数。")
            except ValueError:
                print("输入无效，请输入一个数字。")
        
        # 估算 capacity_info
        print("警告: 正在使用总容量估算司机统计信息...")
        num_drivers = max(1, int(total_capacity / 10))
        avg_cap = total_capacity / num_drivers
        capacity_info = {
            'total': total_capacity,
            'mean': avg_cap,
            'std': total_capacity * 0.1,
            'max': avg_cap * 1.2,
            'min': avg_cap * 0.8,
            'count': num_drivers
        }
    
    else:
        # 没有提供容量
        if args.json_output:
            # Java 调用必须提供 --driver。如果没提供，说明调用有误
            result = {
                "success": False,
                "error": "未提供司机容量。请使用 --driver (格式: ID:Capacity) 参数。",
                "radii": {}
            }
            print(json.dumps(result, ensure_ascii=False))
        else:
            print("错误: 未提供司机容量。请使用 --driver (格式: ID:Capacity) 参数或 --capacity (总容量)。")
        sys.exit(1)
    
    # 获取司机容量
    capacity_info = None
    
    if args.driver:
        # 优先使用 --driver 参数
        try:
            driver_capacities_list = []
            for driver_arg in args.driver:
                parts = driver_arg.split(':')
                if len(parts) != 2:
                    raise ValueError(f"无效的 --driver 参数格式: {driver_arg}")
                # 司机ID (parts[0]) 在这里不需要，我们只关心容量
                capacity = int(parts[1])
                driver_capacities_list.append(capacity)
            
            if not driver_capacities_list:
                raise ValueError("提供了 --driver 参数，但列表为空")
            
            cap_array = np.array(driver_capacities_list)
            
            # 确保至少有一个司机的容量为正
            if np.sum(cap_array) <= 0:
                raise ValueError("所有司机的总剩余容量为0或负数")
            
            capacity_info = {
                'total': np.sum(cap_array),
                'mean': np.mean(cap_array),
                'std': np.std(cap_array),
                'max': np.max(cap_array),
                'min': np.min(cap_array),
                'count': len(cap_array)
            }
            
        except Exception as e:
            if args.json_output:
                result = {"success": False, "error": f"解析司机容量时出错: {str(e)}", "radii": {}}
                print(json.dumps(result, ensure_ascii=False))
            else:
                print(f"错误: 解析司机容量时出错: {e}")
            sys.exit(1)
            
    elif args.capacity is not None:
        # 回退到 --capacity (通常用于交互模式或旧的测试)
        total_capacity = args.capacity
        if total_capacity <= 0:
            if args.json_output:
                result = {"success": False, "error": "容量必须是正数", "radii": {}}
                print(json.dumps(result, ensure_ascii=False))
            else:
                print("错误: 容量必须是正数")
            sys.exit(1)
        
        # 估算 capacity_info (基于旧逻辑)
        print("警告: 正在使用总容量估算司机统计信息。为获得更准确结果，请使用 --driver 参数。")
        num_drivers = max(1, int(total_capacity / 10)) # 假设平均容量为10
        avg_cap = total_capacity / num_drivers
        capacity_info = {
            'total': total_capacity,
            'mean': avg_cap,
            'std': total_capacity * 0.1,  # 假设10%的变异系数
            'max': avg_cap * 1.2,  # 假设最大容量是平均的1.2倍
            'min': avg_cap * 0.8,  # 假设最小容量是平均的0.8倍
            'count': num_drivers
        }

    elif args.interactive:
        # 交互模式
        print("--- 配送半径动态规划优化器 (修复版本) ---")
        print(f"约束条件: L - L^o ≤ {Config.EPSILON} 分钟 (L^o = {Config.L_PROMISED} 分钟)")
        print(f"半径范围: {Config.RADIUS_MIN} ≤ ρ_r ≤ {Config.RADIUS_MAX}")
        
        while True:
            try:
                driver_capacity_input = input("请输入所有司机的剩余容量之和 (例如: 150): ")
                total_capacity = float(driver_capacity_input)
                if total_capacity > 0:
                    break
                else:
                    print("容量必须是正数。")
            except ValueError:
                print("输入无效，请输入一个数字。")
        
        # 估算 capacity_info
        print("警告: 正在使用总容量估算司机统计信息...")
        num_drivers = max(1, int(total_capacity / 10))
        avg_cap = total_capacity / num_drivers
        capacity_info = {
            'total': total_capacity,
            'mean': avg_cap,
            'std': total_capacity * 0.1,
            'max': avg_cap * 1.2,
            'min': avg_cap * 0.8,
            'count': num_drivers
        }
    
    else:
        # 没有提供容量
        if args.json_output:
            # Java 调用必须提供 --driver。如果没提供，说明调用有误
            result = {
                "success": False,
                "error": "未提供司机容量。请使用 --driver (格式: ID:Capacity) 参数。",
                "radii": {}
            }
            print(json.dumps(result, ensure_ascii=False))
        else:
            print("错误: 未提供司机容量。请使用 --driver (格式: ID:Capacity) 参数或 --capacity (总容量)。")
        sys.exit(1)

    # 如果是手动测试模式，直接进入手动测试
    if args.manual_test:
        # 不需要预先提供capacity_info，manual_test_mode会自己处理
        manual_test_mode(dependencies)
        return
        
    # 执行优化
    try:
        if GUROBI_AVAILABLE:
            solver = GurobiSolver(dependencies, capacity_info)
        else:
            solver = DPSolverFixed(dependencies, capacity_info)
            
        max_orders, optimal_radii, optimal_orders, predicted_L = solver.solve()
        
        if max_orders != -1 and optimal_radii:
            # 构建结果字典
            radii_dict = {}
            for i, merchant_id in enumerate(solver.merchant_ids):
                radii_dict[merchant_id] = round(optimal_radii[i], 2)
            orders_dict = {}
            for i, merchant_id in enumerate(solver.merchant_ids):
                orders_dict[str(merchant_id)] = round(optimal_orders[i], 2)
            
            result = {
                "success": True,
                "total_orders": round(max_orders, 2),
                "predicted_delivery_time": round(predicted_L, 2),
                "max_allowed_time": Config.MAX_L,
                "radii": radii_dict,
                "orders_per_merchant": orders_dict
            }
            
            if args.json_output:
                print(json.dumps(result, ensure_ascii=False))
            else:
                print("\n" + "="*30)
                print("      最 优 解 决 方 案")
                print("="*30)
                print(f"找到最优解！")
                print(f"\n最大总潜在订单数: {max_orders:.2f}")
                print(f"预测的平均配送时间: {predicted_L:.2f} 分钟 (约束: ≤ {Config.MAX_L} 分钟)")
                
                print("\n各商家的最优半径和对应订单数:")
                result_df = pd.DataFrame({
                    '商家ID': solver.merchant_ids,
                    '最优半径 (ρ_r)': [round(r, 2) for r in optimal_radii],
                    '潜在订单数 (O_r)': [round(o, 2) for o in optimal_orders]
                })
                print(result_df.to_string(index=False))
        else:
            result = {
                "success": False,
                "error": "在给定的约束和参数下，未找到任何可行的解决方案",
                "radii": {}
            }
            
            if args.json_output:
                print(json.dumps(result, ensure_ascii=False))
            else:
                print("在给定的约束和参数下，未找到任何可行的解决方案。")
                print("您可以尝试：")
                print("  - 增加司机容量")
                print("  - 放宽约束 (增大 epsilon)")
                print("  - 减小半径搜索步长 (RADIUS_STEP)")
                
    except Exception as e:
        result = {
            "success": False,
            "error": f"优化过程中发生错误: {str(e)}",
            "radii": {}
        }
        
        if args.json_output:
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(f"错误: 优化过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()