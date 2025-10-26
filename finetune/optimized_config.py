#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kronos 完整优化配置文件
整合了原始 config.py 和 main.py 中的所有配置逻辑
完全独立，无需额外文件即可运行
"""

import os
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from pathlib import Path

# 初始化日志记录器
def setup_logger():
    """设置日志记录器"""
    logger = logging.getLogger('KronosPipeline')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# 设置日志记录器
logger = setup_logger()


class OptimizedConfig:
    """
    Kronos 完整优化配置类
    整合了所有配置逻辑，支持动态配置和模型版本管理
    完全独立，包含所有必要的配置信息
    """
    
    def __init__(self, 
                 data_source: str = 'sina',
                 model_version: str = 'base',
                 use_gpu: bool = True,
                 config_file: Optional[str] = None,
                 **kwargs):
        """
        初始化配置
        
        Args:
            data_source: 数据源类型，'qlib' 或 'sina'
            model_version: 模型版本，'mini', 'small', 'base'
            use_gpu: 是否使用GPU训练
            config_file: 外部配置文件路径
            **kwargs: 其他配置参数
        """
        self.data_source = data_source
        self.model_version = model_version
        self.use_gpu = use_gpu
        
        # 初始化所有配置
        self._init_base_config()
        self._init_data_config()
        self._init_model_config()
        self._init_training_config()
        self._init_paths_config()
        self._init_logging_config()
        self._init_backtest_config()
        
        # 从外部文件加载配置（如果提供）
        if config_file:
            self.load_from_file(config_file)
        
        # 应用额外参数
        self._apply_kwargs(kwargs)
        
        # 最终配置验证和路径设置
        self._finalize_config()
    
    def _init_base_config(self):
        """初始化基础配置"""
        # 基础参数
        self.seed = 100
        self.clip = 5.0
        
        # 时间相关配置 - 动态计算时间范围
        current_date = datetime.now()
        two_years_ago = (current_date - timedelta(days=365*2)).strftime('%Y-%m-%d')
        six_months_ago = (current_date - timedelta(days=182)).strftime('%Y-%m-%d')  # 使用整数避免精度问题
        current_date_str = current_date.strftime('%Y-%m-%d')
        
        self.dataset_begin_time = two_years_ago
        self.dataset_end_time = current_date_str
        self.train_time_range = [two_years_ago, six_months_ago]
        self.val_time_range = [six_months_ago, current_date_str]
        self.test_time_range = [six_months_ago, current_date_str]
        self.backtest_time_range = [six_months_ago, current_date_str]
    
    def _init_data_config(self):
        """初始化数据相关配置"""
        # 数据源配置
        self.qlib_data_path = './qlib_bin'
        self.instrument = 'csi300'
        self.dataset_path = "./data/processed_datasets"
        self.force_download_data = False
        
        # 数据特征配置
        self.lookback_window = 90  # Number of past time steps for input
        self.predict_window = 10   # Number of future time steps for prediction
        self.max_context = 512     # Maximum context length for the model
        
        # Features to be used from the raw data
        self.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']
        # Time-based features to be generated
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']
        
        # 数据源特定配置
        if self.data_source == 'sina':
            self.max_sina_symbols = 10000
        else:  # qlib
            self.max_sina_symbols = None
    
    def _init_model_config(self):
        """初始化模型相关配置"""
        # 模型版本配置 - 包含所有预训练模型路径
        self.model_versions = {
            'mini': {
                'tokenizer': 'NeoQuasar/Kronos-Tokenizer-2k',
                'predictor': 'NeoQuasar/Kronos-mini'
            },
            'small': {
                'tokenizer': 'NeoQuasar/Kronos-Tokenizer-base',
                'predictor': 'NeoQuasar/Kronos-small'
            },
            'base': {
                'tokenizer': 'NeoQuasar/Kronos-Tokenizer-base',
                'predictor': 'NeoQuasar/Kronos-base'
            },
            'customer': {
                'tokenizer': None,  # 从配置文件读取
                'predictor': None   # 从配置文件读取
            }
        }
        
        # 自定义模型配置路径（先设置，customer版本需要用到）
        self.custom_tokenizer_config = 'custom_tokenizer_config.json'
        self.custom_predictor_config = 'custom_predictor_config.json'
        
        # 设置预训练模型路径
        if self.model_version in self.model_versions:
            if self.model_version == 'customer':
                # customer版本从配置文件读取模型参数
                self._load_customer_model_config()
            else:
                self.pretrained_tokenizer_path = self.model_versions[self.model_version]['tokenizer']
                self.pretrained_predictor_path = self.model_versions[self.model_version]['predictor']
        else:
            raise ValueError(f"不支持的模型版本: {self.model_version}")
    
    def _load_customer_model_config(self):
        """加载customer版本的模型配置"""
        try:
            # 检查自定义配置文件是否存在
            if not os.path.exists(self.custom_tokenizer_config):
                raise FileNotFoundError(f"Customer tokenizer配置文件不存在: {self.custom_tokenizer_config}")
            if not os.path.exists(self.custom_predictor_config):
                raise FileNotFoundError(f"Customer predictor配置文件不存在: {self.custom_predictor_config}")
            
            # 读取tokenizer配置
            with open(self.custom_tokenizer_config, 'r', encoding='utf-8') as f:
                tokenizer_config = json.load(f)
            
            # 读取predictor配置
            with open(self.custom_predictor_config, 'r', encoding='utf-8') as f:
                predictor_config = json.load(f)
            
            # 设置预训练模型路径为None，表示从配置文件创建
            self.pretrained_tokenizer_path = None
            self.pretrained_predictor_path = None
            
            # 保存配置用于模型创建
            self.customer_tokenizer_config = tokenizer_config
            self.customer_predictor_config = predictor_config
            
            logger.info(f"成功加载customer模型配置:")
            logger.info(f"  - Tokenizer配置: {self.custom_tokenizer_config}")
            logger.info(f"  - Predictor配置: {self.custom_predictor_config}")
            
        except Exception as e:
            logger.error(f"加载customer模型配置失败: {str(e)}")
            raise
    
    def _init_training_config(self):
        """初始化训练相关配置"""
        # 训练超参数
        self.epochs = 30
        self.batch_size = 50
        self.log_interval = 100  # Log training status every N batches
        self.accumulation_steps = 10
        
        # 学习率配置
        self.tokenizer_learning_rate = 2e-4
        self.predictor_learning_rate = 4e-5
        
        # 优化器配置
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1
        
        # 训练迭代配置 - 根据batch_size动态计算
        self.n_train_iter = 800000
        self.n_val_iter = 40
        
        # 提前终止配置
        self.early_stopping_patience = 6
    
    def _init_paths_config(self):
        """初始化路径相关配置"""
        # 基础保存路径
        self.save_path = f"./outputs/{self.data_source}/{self.model_version}"
        
        # 模型保存文件夹名称
        self.tokenizer_save_folder_name = 'finetune_tokenizer'
        self.predictor_save_folder_name = 'finetune_predictor'
        self.backtest_save_folder_name = 'finetune_backtest'
        
        # 模型路径 - 保存训练过程中在测试集上评估过的最好模型
        self.finetuned_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/best_model"
        self.finetuned_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/best_model"
        
        # 历史模型记录目录
        self.model_history_dir = "./model_history"
        history_subdir = f"{self.data_source}/{self.model_version}"
        model_history_subdir = os.path.join(self.model_history_dir, history_subdir)
        self.his_best_tokenizer_path = os.path.join(model_history_subdir, "best_tokenizer")
        self.his_best_predictor_path = os.path.join(model_history_subdir, "best_predictor")
        
        # 回测结果路径
        self.backtest_result_path = "./outputs/backtest_results"
    
    def _init_logging_config(self):
        """初始化日志相关配置"""
        # Comet ML 配置
        self.use_comet = False
        self.comet_config = {
            "api_key": "YOUR_COMET_API_KEY",
            "project_name": "Kronos-Finetune-Demo",
            "workspace": "your_comet_workspace"
        }
        self.comet_tag = 'finetune_demo'
        self.comet_name = 'finetune_demo'
    
    def _init_backtest_config(self):
        """初始化回测相关配置"""
        # 回测参数
        self.backtest_n_symbol_hold = 50  # Number of symbols to hold in the portfolio
        self.backtest_n_symbol_drop = 5   # Number of symbols to drop from the pool
        self.backtest_hold_thresh = 5     # Minimum holding period for a stock
        self.backtest_batch_size = 1000
        
        # 推理参数
        self.inference_T = 0.6
        self.inference_top_p = 0.9
        self.inference_top_k = 0
        self.inference_sample_count = 5
        
        # 基准配置
        self.backtest_benchmark = self._set_benchmark(self.instrument)
    
    def _set_benchmark(self, instrument: str) -> str:
        """设置基准"""
        dt_benchmark = {
            'csi800': "SH000906",
            'csi1000': "SH000852",
            'csi300': "SH000300",
        }
        if instrument in dt_benchmark:
            return dt_benchmark[instrument]
        else:
            raise ValueError(f"未定义的基准: {instrument}")
    
    def load_from_file(self, config_file: str):
        """从文件加载配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # 更新配置对象
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    logger.warning(f"未知配置项: {key}")
            
            logger.info(f"从配置文件加载配置: {config_file}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            raise
    
    def save_to_file(self, config_file: str):
        """保存配置到文件"""
        try:
            config_dict = {}
            for key, value in self.__dict__.items():
                if not key.startswith('_'):
                    # 处理不可序列化的对象
                    if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                        config_dict[key] = value
                    else:
                        config_dict[key] = str(value)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置已保存到: {config_file}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {str(e)}")
            raise
    
    def _apply_kwargs(self, kwargs: dict):
        """应用额外参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"未知配置参数: {key}")
        
        # 如果设置了自定义配置文件路径，更新配置
        if hasattr(self, 'custom_tokenizer_config') and hasattr(self, 'custom_predictor_config'):
            self.custom_tokenizer_config = kwargs.get('custom_tokenizer_config', self.custom_tokenizer_config)
            self.custom_predictor_config = kwargs.get('custom_predictor_config', self.custom_predictor_config)
    
    def _finalize_config(self):
        """最终配置验证和路径设置"""
        # 如果是customer版本，重新加载配置（因为可能通过kwargs更新了配置文件路径）
        if self.model_version == 'customer':
            self._load_customer_model_config()
        
        # 确保目录存在
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.model_history_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.finetuned_tokenizer_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.finetuned_predictor_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.his_best_tokenizer_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.his_best_predictor_path), exist_ok=True)
        
        # 验证配置
        self._validate_config()
    
    def _validate_config(self):
        """验证配置的有效性"""
        # 验证模型版本
        if self.model_version not in self.model_versions:
            raise ValueError(f"不支持的模型版本: {self.model_version}")
        
        # 验证数据源
        if self.data_source not in ['qlib', 'sina']:
            raise ValueError(f"不支持的数据源: {self.data_source}")
        
        # 验证时间范围
        if self.dataset_begin_time >= self.dataset_end_time:
            raise ValueError("数据集开始时间必须早于结束时间")
        
        # 验证学习率
        if self.tokenizer_learning_rate <= 0 or self.predictor_learning_rate <= 0:
            raise ValueError("学习率必须大于0")
        
        logger.info("配置验证通过")
    
    def get_model_config(self, model_type: str) -> Dict:
        """获取模型配置信息"""
        if model_type == 'tokenizer':
            config = {
                'pretrained_path': self.pretrained_tokenizer_path,
                'finetuned_path': self.finetuned_tokenizer_path,
                'custom_config': self.custom_tokenizer_config,
                'learning_rate': self.tokenizer_learning_rate
            }
            # 如果是customer版本，添加配置参数
            if self.model_version == 'customer' and hasattr(self, 'customer_tokenizer_config'):
                config['customer_config'] = self.customer_tokenizer_config
            return config
        elif model_type == 'predictor':
            config = {
                'pretrained_path': self.pretrained_predictor_path,
                'finetuned_path': self.finetuned_predictor_path,
                'custom_config': self.custom_predictor_config,
                'learning_rate': self.predictor_learning_rate
            }
            # 如果是customer版本，添加配置参数
            if self.model_version == 'customer' and hasattr(self, 'customer_predictor_config'):
                config['customer_config'] = self.customer_predictor_config
            return config
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def get_training_config(self) -> Dict:
        """获取训练配置信息"""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'log_interval': self.log_interval,
            'accumulation_steps': self.accumulation_steps,
            'early_stopping_patience': self.early_stopping_patience,
            'n_train_iter': self.n_train_iter,
            'n_val_iter': self.n_val_iter,
            'adam_beta1': self.adam_beta1,
            'adam_beta2': self.adam_beta2,
            'adam_weight_decay': self.adam_weight_decay
        }
    
    def get_data_config(self) -> Dict:
        """获取数据配置信息"""
        return {
            'data_source': self.data_source,
            'qlib_data_path': self.qlib_data_path,
            'dataset_path': self.dataset_path,
            'instrument': self.instrument,
            'lookback_window': self.lookback_window,
            'predict_window': self.predict_window,
            'max_context': self.max_context,
            'feature_list': self.feature_list,
            'time_feature_list': self.time_feature_list,
            'train_time_range': self.train_time_range,
            'val_time_range': self.val_time_range,
            'test_time_range': self.test_time_range,
            'backtest_time_range': self.backtest_time_range
        }
    
    def update_paths(self, data_source: str = None, model_version: str = None):
        """动态更新路径配置"""
        if data_source:
            self.data_source = data_source
        if model_version:
            self.model_version = model_version
        
        # 重新初始化路径配置
        self._init_paths_config()
        self._finalize_config()
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"OptimizedConfig(data_source={self.data_source}, model_version={self.model_version}, use_gpu={self.use_gpu})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return f"OptimizedConfig(\n" \
               f"  data_source={self.data_source},\n" \
               f"  model_version={self.model_version},\n" \
               f"  use_gpu={self.use_gpu},\n" \
               f"  epochs={self.epochs},\n" \
               f"  batch_size={self.batch_size},\n" \
               f"  save_path={self.save_path}\n" \
               f")"


def parse_args():
    """解析命令行参数 - 完全独立，不依赖外部"""
    parser = argparse.ArgumentParser(description='Kronos模型训练流水线')
    parser.add_argument('--cpu', action='store_true', default=False, help='使用CPU训练')
    parser.add_argument('--data-source', type=str, default='sina', choices=['qlib', 'sina'], help='数据源类型')
    parser.add_argument('--config-path', type=str, default=None, help='配置文件路径')
    parser.add_argument('--force-download', action='store_true', default=False, help='强制重新下载数据')
    parser.add_argument('--model-version', type=str, default='base', choices=['mini', 'small', 'base', 'customer'],
                        help='模型版本: mini(小), small(中), base(大), customer(自定义配置)')
    parser.add_argument('--custom-tokenizer-config', type=str, default='custom_tokenizer_config.json',
                        help='自定义tokenizer配置文件路径')
    parser.add_argument('--custom-predictor-config', type=str, default='custom_predictor_config.json',
                        help='自定义predictor配置文件路径')
    parser.add_argument('--early-stopping-patience', type=int, default=8,
                        help='提前终止的耐心值（连续多少个epoch测试损失没有提升就停止）')
    return parser.parse_args()


def create_config_from_args(args) -> OptimizedConfig:
    """
    从命令行参数创建配置对象
    
    Args:
        args: 命令行参数对象
        
    Returns:
        OptimizedConfig: 配置对象
    """
    config = OptimizedConfig(
        data_source=args.data_source,
        model_version=args.model_version,
        use_gpu=not args.cpu,
        config_file=args.config_path,
        force_download_data=args.force_download,
        custom_tokenizer_config=args.custom_tokenizer_config,
        custom_predictor_config=args.custom_predictor_config,
        early_stopping_patience=args.early_stopping_patience
    )
    
    return config


def create_default_config() -> OptimizedConfig:
    """创建默认配置"""
    return OptimizedConfig()


def create_config_interactive():
    """交互式创建配置"""
    logger.info("=== Kronos 配置创建器 ===")
    
    # 数据源选择
    logger.info("1. 选择数据源:")
    logger.info("   a) sina (新浪财经)")
    logger.info("   b) qlib (Qlib数据库)")
    data_source_choice = input("请选择 (a/b) [默认: a]: ").strip().lower()
    data_source = 'sina' if data_source_choice in ['a', ''] else 'qlib'
    logger.info(f"选择数据源: {data_source}")
    
    # 模型版本选择
    logger.info("\n2. 选择模型版本:")
    logger.info("   a) mini (小模型)")
    logger.info("   b) small (中等模型)")
    logger.info("   c) base (大模型)")
    logger.info("   d) customer (自定义配置)")
    model_choice = input("请选择 (a/b/c/d) [默认: c]: ").strip().lower()
    model_version_map = {'a': 'mini', 'b': 'small', 'c': 'base', 'd': 'customer'}
    model_version = model_version_map.get(model_choice, 'base')
    logger.info(f"选择模型版本: {model_version}")
    
    # 如果是customer版本，需要输入配置文件路径
    custom_tokenizer_config = 'custom_tokenizer_config.json'
    custom_predictor_config = 'custom_predictor_config.json'
    
    if model_version == 'customer':
        logger.info("\n3. 自定义模型配置:")
        custom_tokenizer_config = input("Tokenizer配置文件路径 [默认: custom_tokenizer_config.json]: ").strip()
        if not custom_tokenizer_config:
            custom_tokenizer_config = 'custom_tokenizer_config.json'
        
        custom_predictor_config = input("Predictor配置文件路径 [默认: custom_predictor_config.json]: ").strip()
        if not custom_predictor_config:
            custom_predictor_config = 'custom_predictor_config.json'
        
        logger.info(f"自定义配置路径 - Tokenizer: {custom_tokenizer_config}, Predictor: {custom_predictor_config}")
    
    # 训练参数
    logger.info(f"\n{3 if model_version != 'customer' else 4}. 训练参数:")
    try:
        epochs = int(input("训练轮数 [默认: 8]: ") or "8")
        batch_size = int(input("批次大小 [默认: 50]: ") or "50")
    except ValueError:
        logger.warning("输入无效，使用默认值")
        epochs, batch_size = 8, 50
    
    logger.info(f"训练参数 - epochs: {epochs}, batch_size: {batch_size}")
    
    # GPU使用
    logger.info(f"\n{4 if model_version != 'customer' else 5}. 硬件配置:")
    use_gpu_choice = input("使用GPU训练? (y/n) [默认: y]: ").strip().lower()
    use_gpu = use_gpu_choice in ['y', 'yes', ''] or use_gpu_choice == ''
    logger.info(f"GPU使用: {use_gpu}")
    
    # Comet ML
    logger.info(f"\n{5 if model_version != 'customer' else 6}. 实验跟踪:")
    use_comet_choice = input("使用Comet ML跟踪实验? (y/n) [默认: n]: ").strip().lower()
    use_comet = use_comet_choice in ['y', 'yes']
    logger.info(f"Comet ML使用: {use_comet}")
    
    # 创建配置
    config_kwargs = {
        'data_source': data_source,
        'model_version': model_version,
        'epochs': epochs,
        'batch_size': batch_size,
        'use_gpu': use_gpu,
        'use_comet': use_comet
    }
    
    # 如果是customer版本，添加自定义配置文件路径
    if model_version == 'customer':
        config_kwargs['custom_tokenizer_config'] = custom_tokenizer_config
        config_kwargs['custom_predictor_config'] = custom_predictor_config
    
    config = OptimizedConfig(**config_kwargs)
    
    logger.info(f"配置创建完成: {config}")
    return config


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        # 交互式模式
        config = create_config_interactive()
    else:
        # 测试配置类
        logger.info("=== Kronos 优化配置测试 ===")
        
        # 测试默认配置
        config = OptimizedConfig()
        logger.info("默认配置:")
        logger.info(f"{config}")
        
        logger.info("模型配置:")
        logger.info(f"{config.get_model_config('tokenizer')}")
        
        logger.info("训练配置:")
        logger.info(f"{config.get_training_config()}")
        
        logger.info("数据配置:")
        logger.info(f"{config.get_data_config()}")
        
        # 测试自定义配置
        logger.info("=== 自定义配置测试 ===")
        custom_config = OptimizedConfig(
            data_source='qlib',
            model_version='mini',
            epochs=5,
            batch_size=32
        )
        logger.info(f"自定义配置: {custom_config}")
        
        # 测试命令行参数解析
        logger.info("=== 命令行参数测试 ===")
        test_args = parse_args()
        logger.info(f"解析的命令行参数: {test_args}")
        
        # 从参数创建配置
        config_from_args = create_config_from_args(test_args)
        logger.info(f"从参数创建的配置: {config_from_args}")