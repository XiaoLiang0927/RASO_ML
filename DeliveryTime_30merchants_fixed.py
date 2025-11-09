import pandas as pd
import numpy as np
from scipy.integrate import quad
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.base import clone
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, VotingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, RFE, RFECV, SelectKBest, f_regression, mutual_info_regression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import json
import os
import warnings
warnings.filterwarnings('ignore')
from scipy import stats

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DeliveryTimePredictorFixed:
    def __init__(self):
        self.merchant_params = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.models = {}
        self.results = {}
        self.best_features = None
        self.feature_importance = None
        
    def load_merchant_params(self, file_path):
        """Load merchant model parameters from CSV file"""
        print("Loading merchant model parameters...")
        self.merchant_params = pd.read_csv(file_path)
        print(f"Successfully loaded parameters for {len(self.merchant_params)} merchants")
        
    def order_model(self, x, alpha_prime, beta):
        """Order count model as a function of distance"""
        return alpha_prime * x * np.exp(beta * x)
    
    def prepare_features_with_individual_radii_fixed(self, merchant_radius_path, driver_capacity_path, orders_path):
        """
        修复版本的特征准备函数，只保留以下特征：
        1. 每个商家的半径
        2. 每个商家的潜在订单数量
        3. 每个司机的剩余容量
        4. 潜在订单数量之和
        5. 司机剩余容量之和
        6. 潜在订单数量之和与司机剩余容量之和的比值
        """
        print("Preparing features with fixed logic...")
        
        # 1. 读取商户半径数据
        merchant_radius_df = pd.read_csv(merchant_radius_path)
        merchant_radius_df['merchant_id'] = merchant_radius_df['merchant_id'].str.strip()
        
        # 数据验证
        print(f"Merchant radius data shape: {merchant_radius_df.shape}")
        print(f"Unique merchants: {merchant_radius_df['merchant_id'].nunique()}")
        print(f"Unique iterations: {merchant_radius_df['iteration'].nunique()}")
        
        # 透视为宽格式
        radius_wide = merchant_radius_df.pivot(
            index='iteration', 
            columns='merchant_id', 
            values='radius'
        ).reset_index()
        
        # 重命名列
        radius_wide.columns = ['iteration'] + [f'radius_{col}' for col in radius_wide.columns if col != 'iteration']
        
        # 2. 读取司机容量数据
        driver_capacity_df = pd.read_csv(driver_capacity_path)
        
        # 数据验证
        print(f"Driver capacity data shape: {driver_capacity_df.shape}")
        print(f"Unique timestamps: {driver_capacity_df['timestamp'].nunique()}")
        
        # 过滤特定时间点的数据 (12:00)
        # target_time = 1665980100
        # target_time = 1665972000 #10:00
        target_time = 1665972000 + 1800 #10:30
        # target_time = 1665972000 + 1800*2 #11:00
        # target_time = 1665972000 + 1800*3 #11:30
        # target_time = 1665979200 #12:00
        # target_time = 1665979200 + 1800 #12:30
        # target_time = 1665979200 + 1800*2 #13:00
        # target_time = 1665979200 + 1800*3 #13:30

        #15min
        # target_time = 1665979200  # 12:00
        # target_time= 1665979200 + 900 #12:15
        # target_time= 1665979200 + 900*2 #12:30
        # target_time= 1665979200 + 900*3 #12:45
        # target_time= 1665979200 + 900*4 #13:00
        # target_time= 1665979200 + 900*5 #13:15
        # target_time= 1665979200 + 900*6 #13:30
        # target_time= 1665979200 + 900*7 #13:45

        driver_capacity_at_12 = driver_capacity_df[driver_capacity_df['timestamp'] == target_time]
        
        if driver_capacity_at_12.empty:
            print(f"Warning: No driver capacity data found for timestamp {target_time}")
            # 使用 target_time 之前的最近时间戳；若不存在，则使用最早可用时间戳
            available_timestamps = np.sort(driver_capacity_df['timestamp'].unique())
            previous_candidates = available_timestamps[available_timestamps < target_time]
            if len(previous_candidates) > 0:
                target_time = previous_candidates[-1]
                print(f"Using previous closest timestamp {target_time}")
            else:
                target_time = available_timestamps[0]
                print(f"No earlier timestamp found; using earliest available {target_time}")
            driver_capacity_at_12 = driver_capacity_df[driver_capacity_df['timestamp'] == target_time]
        
        # 计算总容量和容量统计 - 只保留总容量
        total_capacity = driver_capacity_at_12.groupby('iteration')['capacity'].agg([
            ('total_capacity', 'sum'),
            # ('avg_capacity_per_driver', 'mean'),  # 注释掉不需要的特征
            # ('capacity_std', 'std'),
            #('num_drivers', 'count'),
            # ('max_driver_capacity', 'max'),
            # ('min_driver_capacity', 'min')
        ]).reset_index()
        
        # 填充缺失值
        # total_capacity['capacity_std'].fillna(0, inplace=True)  # 注释掉不需要的特征处理
        
        # 3. 计算潜在订单数 - 只保留每个商家的订单数和总订单数
        merchant_params = self.merchant_params.set_index('merchant_id')
        potential_orders_list = []
        
        for iteration, group in merchant_radius_df.groupby('iteration'):
            orders_dict = {'iteration': iteration}
            total_orders = 0
            # total_weighted_orders = 0  # 注释掉不需要的特征
            # radius_stats = []
            # order_density_stats = []
            
            for _, row in group.iterrows():
                merchant_id = row['merchant_id']
                radius = row['radius']
                # radius_stats.append(radius)  # 注释掉不需要的特征
                
                if merchant_id in merchant_params.index:
                    alpha_prime = merchant_params.loc[merchant_id, 'alpha_prime']
                    beta = merchant_params.loc[merchant_id, 'beta']
                    
                    # 计算订单积分
                    integral, _ = quad(lambda x: self.order_model(x, alpha_prime, beta), 0, radius)
                    orders_dict[f'orders_{merchant_id}'] = integral
                    total_orders += integral
                    
                    # 以下特征都注释掉
                    # # 计算加权订单数（考虑商户参数）
                    # weight = alpha_prime * abs(beta)  # 使用参数作为权重
                    # total_weighted_orders += integral * weight
                    
                    # # 计算订单密度
                    # area = np.pi * radius**2 if radius > 0 else 1e-6
                    # density = integral / area
                    # order_density_stats.append(density)
                    
                    # # 添加商户特定特征
                    # orders_dict[f'order_density_{merchant_id}'] = density
                    # orders_dict[f'radius_efficiency_{merchant_id}'] = integral / radius if radius > 0 else 0
            
            # 添加聚合特征 - 只保留总潜在订单数
            orders_dict['total_potential_orders'] = total_orders
            # orders_dict['total_weighted_orders'] = total_weighted_orders  # 注释掉不需要的特征
            
            # 以下特征都注释掉
            # # 半径统计特征
            # if radius_stats:
            #     orders_dict['avg_radius'] = np.mean(radius_stats)
            #     orders_dict['std_radius'] = np.std(radius_stats)
            #     orders_dict['max_radius'] = np.max(radius_stats)
            #     orders_dict['min_radius'] = np.min(radius_stats)
            #     orders_dict['radius_range'] = np.max(radius_stats) - np.min(radius_stats)
            #     orders_dict['radius_cv'] = np.std(radius_stats) / np.mean(radius_stats) if np.mean(radius_stats) > 0 else 0
            
            # # 订单密度统计特征
            # if order_density_stats:
            #     orders_dict['avg_order_density'] = np.mean(order_density_stats)
            #     orders_dict['std_order_density'] = np.std(order_density_stats)
            #     orders_dict['max_order_density'] = np.max(order_density_stats)
            #     orders_dict['total_coverage_area'] = np.pi * sum([r**2 for r in radius_stats])
            
            potential_orders_list.append(orders_dict)
        
        potential_orders_df = pd.DataFrame(potential_orders_list)
        
        # 4. 合并特征
        features_df = radius_wide.merge(total_capacity, on='iteration')
        features_df = features_df.merge(potential_orders_df, on='iteration')
        
        # 5. 计算目标变量 - 修复时间过滤逻辑
        orders_df = pd.read_csv(orders_path)
        
        # 数据验证
        print(f"Orders data shape: {orders_df.shape}")
        print(f"Orders timestamp range: {orders_df['timestamp'].min()} to {orders_df['timestamp'].max()}")
        
        # 恢复时间过滤 - 过滤12:00-14:00的订单
        # start_time = 1665972000  # 10:00
        start_time = 1665972000 + 1800  # 10:30
        # start_time = 1665972000 + 1800*2  # 11:00
        # start_time = 1665972000 + 1800*3  # 11:30
        # start_time = 1665979200  # 12:00
        # start_time = 1665979200 + 1800  # 12:30
        # start_time = 1665979200 + 1800*2  # 13:00
        # start_time = 1665979200 + 1800*3  # 13:30
        # end_time = 1665986400    # 14:00
        end_time = start_time + 1800    # 30min 
        
        #15min
        # start_time = 1665979200  # 12:00
        # start_time= 1665979200 + 900 #12:15
        # start_time= 1665979200 + 900*2 #12:30
        # start_time= 1665979200 + 900*3 #12:45
        # start_time= 1665979200 + 900*4 #13:00
        # start_time= 1665979200 + 900*5 #13:15
        # start_time= 1665979200 + 900*6 #13:30
        # start_time= 1665979200 + 900*7 #13:45
        # end_time = start_time + 900 #15min
        
        time_filtered_orders = orders_df[
            (orders_df['timestamp'] >= start_time) & 
            (orders_df['timestamp'] <= end_time)
        ]
        
        print(f"Filtered orders shape: {time_filtered_orders.shape}")
        
        if time_filtered_orders.empty:
            print("Warning: No orders found in the specified time range. Using all orders.")
            time_filtered_orders = orders_df
        
        # 计算配送时长（分钟）
        time_filtered_orders['delivery_duration'] = (
            time_filtered_orders['delivery_time'] - time_filtered_orders['timestamp']
        ) / 60
        
        # 数据清洗 - 移除异常值
        # 移除负配送时间和过长配送时间
        time_filtered_orders = time_filtered_orders[
            (time_filtered_orders['delivery_duration'] > 0) & 
            (time_filtered_orders['delivery_duration'] < 300)  # 小于5小时
        ]
        
        # 计算配送时间统计
        delivery_stats = time_filtered_orders.groupby('iteration')['delivery_duration'].agg([
            ('avg_delivery_time', 'mean'),
            ('median_delivery_time', 'median'),
            ('std_delivery_time', 'std'),
            ('min_delivery_time', 'min'),
            ('max_delivery_time', 'max'),
            ('count_orders', 'count'),
            ('q25_delivery_time', lambda x: x.quantile(0.25)),
            ('q75_delivery_time', lambda x: x.quantile(0.75))
        ]).reset_index()
        
        # 填充缺失值
        delivery_stats['std_delivery_time'].fillna(0, inplace=True)
        
        # 6. 合并最终数据
        final_df = features_df.merge(delivery_stats, on='iteration', how='inner')
        
        # 7. 创建交互特征 - 只保留订单容量比
        print("Creating interaction features...")
        
        # 基本交互特征 - 只保留订单容量比
        final_df['orders_capacity_ratio'] = final_df['total_potential_orders'] / final_df['total_capacity'].where(final_df['total_capacity'] > 0, 1)
        # final_df['orders_per_driver'] = final_df['total_potential_orders'] / final_df['num_drivers'].where(final_df['num_drivers'] > 0, 1)  # 注释掉不需要的特征
        
        # 以下特征都注释掉
        # # 高级交互特征（移除了使用count_orders的特征以避免数据泄漏）
        # final_df['demand_supply_imbalance'] = (final_df['total_potential_orders'] - final_df['total_capacity']) / final_df['total_capacity'].where(final_df['total_capacity'] > 0, 1)
        # final_df['coverage_efficiency'] = final_df['total_potential_orders'] / final_df['total_coverage_area'].where(final_df['total_coverage_area'] > 0, 1)
        
        # # 新增的交互特征（不依赖count_orders）
        # final_df['weighted_orders_capacity_ratio'] = final_df['total_weighted_orders'] / final_df['total_capacity'].where(final_df['total_capacity'] > 0, 1)
        # final_df['capacity_per_coverage'] = final_df['total_capacity'] / final_df['total_coverage_area'].where(final_df['total_coverage_area'] > 0, 1)
        # # final_df['driver_efficiency'] = final_df['total_potential_orders'] / (final_df['num_drivers'] * final_df['avg_capacity_per_driver']).where((final_df['num_drivers'] * final_df['avg_capacity_per_driver']) > 0, 1)
        
        # 8. 数据质量检查
        print("Performing data quality checks...")
        
        # 移除缺失值
        initial_rows = len(final_df)
        final_df.dropna(inplace=True)
        final_rows = len(final_df)
        
        if initial_rows != final_rows:
            print(f"Removed {initial_rows - final_rows} rows with missing values")
        
        # 移除常数特征
        constant_features = []
        for col in final_df.columns:
            if col not in ['iteration', 'avg_delivery_time', 'median_delivery_time', 'std_delivery_time', 
                          'min_delivery_time', 'max_delivery_time', 'count_orders', 'q25_delivery_time', 'q75_delivery_time']:
                if final_df[col].nunique() <= 1 or final_df[col].std() == 0:
                    constant_features.append(col)
        
        if constant_features:
            print(f"Removing constant features: {constant_features}")
            final_df.drop(columns=constant_features, inplace=True)
        
        # 9. 特征缩放和变换 - 注释掉所有变换
        # print("Applying feature transformations...")
        
        # # 对高度偏斜的特征应用对数变换
        # numeric_cols = final_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        # exclude_cols = ['iteration', 'avg_delivery_time', 'median_delivery_time', 'std_delivery_time', 
        #                'min_delivery_time', 'max_delivery_time', 'count_orders', 'q25_delivery_time', 'q75_delivery_time']
        # feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # for col in feature_cols:
        #     skewness = final_df[col].skew()
        #     if abs(skewness) > 2:  # 高度偏斜
        #         # 应用对数变换
        #         final_df[f'log_{col}'] = np.log1p(final_df[col].clip(lower=0))
        
        print(f"Feature preparation complete. Dataset contains {len(final_df)} samples with {len([col for col in final_df.columns if col not in ['iteration', 'avg_delivery_time', 'median_delivery_time', 'std_delivery_time', 'min_delivery_time', 'max_delivery_time', 'count_orders', 'q25_delivery_time', 'q75_delivery_time']])} features")
        
        # 保存处理后的数据
        final_df.to_csv('prepared_features_fixed.csv', index=False)
        print("Fixed features saved to prepared_features_fixed.csv")
        
        return final_df
    
    def train_models_fixed(self, features_df, test_size=0.2, random_state=42):
        """修复版本的模型训练函数"""
        print("Training models with fixed approach...")
        
        # 准备特征和目标变量
        exclude_cols = ['iteration', 'avg_delivery_time', 'median_delivery_time', 'std_delivery_time', 
                       'min_delivery_time', 'max_delivery_time', 'count_orders', 'q25_delivery_time', 'q75_delivery_time']
        X = features_df.drop([col for col in exclude_cols if col in features_df.columns], axis=1)
        y = features_df['avg_delivery_time']
        
        # 存储特征名称
        feature_names = X.columns.tolist()
        print(f"Training with {len(feature_names)} features")
        
        # 数据验证
        print(f"Target variable statistics:")
        print(f"Mean: {y.mean():.2f}, Std: {y.std():.2f}")
        print(f"Min: {y.min():.2f}, Max: {y.max():.2f}")
        
        # 检查异常值
        q1 = y.quantile(0.25)
        q3 = y.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (y < lower_bound) | (y > upper_bound)
        
        if outliers.sum() > 0:
            print(f"Detected {outliers.sum()} outliers in target variable")
            # 可选择移除异常值
            # X = X[~outliers]
            # y = y[~outliers]
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        # 特征缩放
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 定义模型
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=random_state),
            'SVR': SVR(kernel='rbf', C=1.0),
        }
        
        # 训练和评估模型
        self.results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            start_time = time.time()
            
            # 训练模型
            if name in ['SVR']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # 交叉验证
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # 交叉验证
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            # 计算指标
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            training_time = time.time() - start_time
            
            # 存储结果
            self.results[name] = {
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae,
                'CV_R2_mean': cv_scores.mean(),
                'CV_R2_std': cv_scores.std(),
                'Training_time': training_time
            }
            
            # 存储模型
            self.models[name] = model
            
            print(f"{name} completed: R²={r2:.4f}, CV R²={cv_scores.mean():.4f}±{cv_scores.std():.4f}, Time={training_time:.2f}s")
        
        return X_test, y_test, feature_names
    
    def get_current_features(self, features_df):
        """获取当前使用的所有特征列表"""
        exclude_cols = [
            'iteration', 'avg_delivery_time', 'median_delivery_time', 
            'std_delivery_time', 'min_delivery_time', 'max_delivery_time',
            'count_orders', 'q25_delivery_time', 'q75_delivery_time'
        ]
        
        feature_columns = [col for col in features_df.columns if col not in exclude_cols]
        
        print("\n=== 当前使用的特征列表 ===")
        print(f"总共 {len(feature_columns)} 个特征:")
        
        # 按类别分组显示特征
        basic_features = []
        radius_features = []
        orders_features = []
        capacity_features = []
        interaction_features = []
        statistical_features = []
        
        for feature in feature_columns:
            if 'radius' in feature:
                radius_features.append(feature)
            elif 'orders' in feature and 'interaction' not in feature:
                orders_features.append(feature)
            elif 'capacity' in feature or 'driver' in feature:
                capacity_features.append(feature)
            elif any(x in feature for x in ['ratio', 'efficiency', 'imbalance', 'per']):
                interaction_features.append(feature)
            elif any(x in feature for x in ['log', 'sqrt', 'squared']):
                statistical_features.append(feature)
            else:
                basic_features.append(feature)
        
        if basic_features:
            print(f"\n基础特征 ({len(basic_features)}个):")
            for feature in basic_features:
                print(f"  - {feature}")
        
        if radius_features:
            print(f"\n半径相关特征 ({len(radius_features)}个):")
            for feature in radius_features:
                print(f"  - {feature}")
        
        if orders_features:
            print(f"\n订单相关特征 ({len(orders_features)}个):")
            for feature in orders_features:
                print(f"  - {feature}")
        
        if capacity_features:
            print(f"\n容量/司机相关特征 ({len(capacity_features)}个):")
            for feature in capacity_features:
                print(f"  - {feature}")
        
        if interaction_features:
            print(f"\n交互特征 ({len(interaction_features)}个):")
            for feature in interaction_features:
                print(f"  - {feature}")
        
        if statistical_features:
            print(f"\n统计变换特征 ({len(statistical_features)}个):")
            for feature in statistical_features:
                print(f"  - {feature}")
        
        print(f"\n注意: 已移除 count_orders 相关特征以避免数据泄漏")
        
        return feature_columns

    def analyze_feature_importance(self, feature_names):
        """分析特征重要性"""
        print("\n=== Feature Importance Analysis ===")
        
        # 从树模型获取特征重要性
        tree_models = ['Random Forest', 'Gradient Boosting', 'XGBoost']
        
        for model_name in tree_models:
            if model_name in self.models:
                model = self.models[model_name]
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    feature_importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importance
                    }).sort_values('importance', ascending=False)
                    
                    print(f"\nTop 10 features for {model_name}:")
                    print(feature_importance_df.head(10))
    
    def save_results_fixed(self, feature_names):
        """保存结果"""
        # 保存模型性能结果
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv('model_evaluation_results_fixed.csv')
        
        # 保存模型
        for name, model in self.models.items():
            filename = f"model_{name.lower().replace(' ', '_')}_fixed.pkl"
            joblib.dump(model, filename)
        
        # 保存缩放器
        joblib.dump(self.scaler, 'scaler_fixed.pkl')
        
        # 保存特征名称列表
        with open('best_features_fixed.json', 'w') as f:
            json.dump(feature_names, f, indent=2)
        
        print("Results, models, and feature names saved successfully!")

    def plot_r2_and_training_time_fixed(self, save_path=None, sort_by=None, descending=True, show=False):
        """可视化各模型的R²（上）和训练时间（下）对比"""
        if not self.results:
            print("No results to plot. Please run training first.")
            return
        
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 构造数据
        plots_dir = os.path.join(os.getcwd(), 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        metrics_df = pd.DataFrame(self.results).T
        
        # 排序逻辑：默认按 R2 降序，可选择按训练时间
        if sort_by in ('R2', 'Training_time'):
            sorted_models = metrics_df.sort_values(sort_by, ascending=not descending).index
        else:
            sorted_models = metrics_df.sort_values('R2', ascending=False).index
        
        # 画布
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))
        bar_width = 0.6
        
        # 上图：R² 分数，使用渐变色
        r2_series = metrics_df.loc[sorted_models, 'R2']
        bars = axes[0].bar(
            range(len(sorted_models)),
            r2_series,
            color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_models))),
            width=bar_width
        )
        axes[0].set_title('R² Comparison', fontsize=16)
        axes[0].set_ylabel('R²', fontsize=14)
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_yticks(np.arange(0.0, 1.0 + 1e-9, 0.2))
        axes[0].set_xticks(range(len(sorted_models)))
        axes[0].set_xticklabels(sorted_models, rotation=45, ha='right',fontsize=14)
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 下图：训练时间，紫色条形图
        time_series = metrics_df.loc[sorted_models, 'Training_time']
        time_series.plot(kind='bar', ax=axes[1], color='purple', width=bar_width)
        axes[1].set_title('Training Time Comparison', fontsize=16)
        axes[1].set_ylabel('Time(seconds)', fontsize=14)
        axes[1].tick_params(axis='x', labelsize=14, rotation=45)
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        # 数值标注
        max_time = float(time_series.max()) if len(time_series) else 0.0
        for p in axes[1].patches:
            h = p.get_height()
            axes[1].text(
                p.get_x() + p.get_width()/2., 
                h + (max_time * 0.01 if max_time > 0 else 0.1),
                f'{h:.2f}', ha='center', va='bottom', fontsize=9
            )
        
        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(plots_dir, 'model_r2_and_training_time_fixed.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        print(f"Saved R² and training time comparison plot to {save_path}")


if __name__ == "__main__":
    # 初始化预测器
    predictor = DeliveryTimePredictorFixed()
    
    # 加载商户参数
    merchant_params_path = "merchant_model_parameters.csv"
    predictor.load_merchant_params(merchant_params_path)
    
    # 数据路径 - 使用本地样本数据进行测试
    # merchant_radius_path = r"/Users/xiao/Documents/博二上/博二上课题/deliveryRadius_2Areas_2.14_6eventDriven_2hrs/merchant_radius copy.csv" #2hrs
    # driver_capacity_path = r"/Users/xiao/Documents/博二上/博二上课题/deliveryRadius_2Areas_2.14_6eventDriven_2hrs/driver_capacity copy.csv" #2hrs
    # orders_path = r"/Users/xiao/Documents/博二上/博二上课题/deliveryRadius_2Areas_2.14_6eventDriven_2hrs/all_iterations_orders copy.csv" #2hrs
    merchant_radius_path = r"/Users/xiao/Documents/博二上/博二上课题/deliveryRadius_2Areas_2.14_6eventDriven_2hrs/merchant_radius.csv" #4hrs
    driver_capacity_path = r"/Users/xiao/Documents/博二上/博二上课题/deliveryRadius_2Areas_2.14_6eventDriven_2hrs/driver_capacity.csv" #4hrs
    orders_path = r"/Users/xiao/Documents/博二上/博二上课题/deliveryRadius_2Areas_2.14_6eventDriven_2hrs/all_iterations_orders.csv" #4hrs
    
    print("\n=== Step 1: Preparing features with fixed logic ===")
    try:
        features_df = predictor.prepare_features_with_individual_radii_fixed(
            merchant_radius_path, driver_capacity_path, orders_path
        )
        
        print("\n=== Step 2: 显示当前使用的特征 ===")
        current_features = predictor.get_current_features(features_df)
        
        print("\n=== Step 3: Training models with fixed approach ===")
        X_test, y_test, feature_names = predictor.train_models_fixed(
            features_df,
            test_size=0.2,
            random_state=42
        )
        
        print("\n=== Step 4: Analyzing feature importance ===")
        predictor.analyze_feature_importance(feature_names)

        print("\n=== Step 4.1: Visualizing R² and Training Time ===")
        predictor.plot_r2_and_training_time_fixed(
            save_path=os.path.join('plots', 'model_r2_and_training_time_fixed.png'),
            sort_by='R2',
            descending=True,
            show=False
        )

        print("\n=== Step 5: Saving results ===")
        predictor.save_results_fixed(feature_names)
        
        print("\n=== Summary of Fixed Results ===")
        best_model_name = max(predictor.results, key=lambda x: predictor.results[x]['R2'])
        best_r2 = predictor.results[best_model_name]['R2']
        best_rmse = predictor.results[best_model_name]['RMSE']
        print(f"Best model: {best_model_name}")
        print(f"Best R² score: {best_r2:.4f}")
        print(f"Best RMSE: {best_rmse:.4f}")
        
        print("\nFixed version completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please check if the sample data files exist in the sample_data directory.")
        print("You may need to copy the data files to the local directory or adjust the paths.")