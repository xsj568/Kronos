#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kronos 完整训练流水线脚本
包含数据处理、分词模型训练和预测模型训练
支持GPU和CPU训练模式
"""

import os
import sys
import json
import time
import logging
import argparse
import torch
from time import gmtime, strftime
from pathlib import Path

# 确保项目根目录在路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from model.kronos import KronosTokenizer, Kronos
from common_data_processor import DataProcessorFactory, QlibDataProcessor, SinaDataProcessor
from utils.training_pipeline_utils import (
    setup_logging,
    setup_ddp,
    cleanup_ddp,
    set_seed,
    get_model_size,
    format_time,
    create_dataloaders_ddp,
    create_dataloaders_cpu,
    setup_comet_logger,
    save_model_checkpoint,
    save_training_summary,
    save_pipeline_config
)

# 初始化日志
logger = setup_logging()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Kronos模型完整训练流水线')
    parser.add_argument('--cpu', action='store_true', help='使用CPU训练')
    parser.add_argument('--data-source', type=str, default='qlib', choices=['qlib', 'sina'], help='数据源类型')
    parser.add_argument('--skip-data-process', action='store_true', help='跳过数据处理步骤')
    parser.add_argument('--skip-tokenizer', action='store_true', help='跳过分词模型训练')
    parser.add_argument('--skip-predictor', action='store_true', help='跳过预测模型训练')
    parser.add_argument('--fast-mode', action='store_true', help='快速模式，减少训练轮数和批次')
    parser.add_argument('--config-path', type=str, default=None, help='配置文件路径')
    return parser.parse_args()

def process_data(config, data_source):
    """处理数据"""
    logger.info(f"开始处理{data_source}数据...")
    try:
        # 使用工厂创建数据处理器
        processor = DataProcessorFactory.create_processor(data_source, config)
        result = processor.run_pipeline()
        logger.info(f"数据处理完成: {result}")
        return True
    except Exception as e:
        logger.error(f"处理数据时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def train_tokenizer_gpu(config):
    """使用GPU训练分词模型"""
    logger.info("开始使用GPU训练分词模型...")
    
    # 检查WORLD_SIZE环境变量
    if "WORLD_SIZE" not in os.environ:
        logger.error("GPU训练必须使用torchrun启动，例如：torchrun --standalone --nproc_per_node=NUM_GPUS train_complete_pipeline.py")
        return False
    
    try:
        # 设置分布式环境
        rank, world_size, local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
        is_master = (rank == 0)
        set_seed(config.seed, rank)
        
        # 保存目录
        save_dir = os.path.join(config.save_path, config.tokenizer_save_folder_name)
        if is_master:
            os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
            
        # 设置Comet日志记录器
        comet_logger = None
        if is_master and config.use_comet:
            import comet_ml
            comet_logger = comet_ml.Experiment(
                api_key=config.comet_config['api_key'],
                project_name=config.comet_config['project_name'],
                workspace=config.comet_config['workspace'],
            )
            comet_logger.add_tag(config.comet_tag)
            comet_logger.set_name(config.comet_name)
            comet_logger.log_parameters(config.__dict__)
            logger.info("Comet日志记录器初始化成功")
        
        # 模型初始化
        model = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
        
        if is_master:
            logger.info(f"分词模型大小: {get_model_size(model.module)}")
        
        # 创建数据加载器
        train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders_ddp(
            config.__dict__, rank, world_size
        )
        
        # 训练
        from train_tokenizer import train_model
        _, dt_result = train_model(
            model, device, config.__dict__, save_dir, comet_logger, rank, world_size
        )
        
        # 清理
        cleanup_ddp()
        
        if is_master:
            logger.info("分词模型GPU训练完成")
            
        return True
    except Exception as e:
        logger.error(f"分词模型GPU训练出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 确保清理分布式环境
        try:
            cleanup_ddp()
        except:
            pass
            
        return False

def train_tokenizer_cpu(config):
    """使用CPU训练分词模型"""
    logger.info("开始使用CPU训练分词模型...")
    
    try:
        # 设置
        device = torch.device("cpu")
        set_seed(config.seed)
        
        # 保存目录
        save_dir = os.path.join(config.save_path, config.tokenizer_save_folder_name)
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        
        # 设置Comet日志记录器
        comet_logger = None
        if config.use_comet:
            from importlib import import_module
            comet_ml = import_module('comet_ml')
            comet_logger = comet_ml.Experiment(
                api_key=config.comet_config['api_key'],
                project_name=config.comet_config['project_name'],
                workspace=config.comet_config['workspace'],
            )
            comet_logger.add_tag(config.comet_tag)
            comet_logger.set_name(config.comet_name)
            comet_logger.log_parameters(config.__dict__)
            logger.info("Comet日志记录器初始化成功")
        
        # 模型初始化
        model = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
        model.to(device)
        logger.info(f"分词模型大小: {get_model_size(model)}")
        
        # 创建数据加载器
        train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders_cpu(config.__dict__)
        
        # 训练
        from train_tokenizer_cpu import train_model
        _, dt_result = train_model(model, device, config.__dict__, save_dir, comet_logger)
        
        logger.info("分词模型CPU训练完成")
        return True
    except Exception as e:
        logger.error(f"分词模型CPU训练出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def train_predictor_gpu(config):
    """使用GPU训练预测模型"""
    logger.info("开始使用GPU训练预测模型...")
    
    # 检查WORLD_SIZE环境变量
    if "WORLD_SIZE" not in os.environ:
        logger.error("GPU训练必须使用torchrun启动，例如：torchrun --standalone --nproc_per_node=NUM_GPUS train_complete_pipeline.py")
        return False
    
    try:
        # 设置分布式环境
        rank, world_size, local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
        is_master = (rank == 0)
        set_seed(config.seed, rank)
        
        # 保存目录
        save_dir = os.path.join(config.save_path, config.predictor_save_folder_name)
        if is_master:
            os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
            
        # 设置Comet日志记录器
        comet_logger = None
        if is_master and config.use_comet:
            import comet_ml
            comet_logger = comet_ml.Experiment(
                api_key=config.comet_config['api_key'],
                project_name=config.comet_config['project_name'],
                workspace=config.comet_config['workspace'],
            )
            comet_logger.add_tag(config.comet_tag)
            comet_logger.set_name(config.comet_name)
            comet_logger.log_parameters(config.__dict__)
            logger.info("Comet日志记录器初始化成功")
        
        # 初始化模型
        tokenizer = KronosTokenizer.from_pretrained(config.finetuned_tokenizer_path)
        tokenizer.eval().to(device)
        
        model = Kronos.from_pretrained(config.pretrained_predictor_path)
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
        
        if is_master:
            logger.info(f"预测模型大小: {get_model_size(model.module)}")
        
        # 训练
        from train_predictor import train_model
        dt_result = train_model(
            model, tokenizer, device, config.__dict__, save_dir, comet_logger, rank, world_size
        )
        
        # 清理
        cleanup_ddp()
        
        if is_master:
            logger.info("预测模型GPU训练完成")
            
        return True
    except Exception as e:
        logger.error(f"预测模型GPU训练出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 确保清理分布式环境
        try:
            cleanup_ddp()
        except:
            pass
            
        return False

def train_predictor_cpu(config):
    """使用CPU训练预测模型"""
    logger.info("开始使用CPU训练预测模型...")
    
    try:
        # 设置
        device = torch.device("cpu")
        set_seed(config.seed)
        
        # 保存目录
        save_dir = os.path.join(config.save_path, config.predictor_save_folder_name)
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        
        # 设置Comet日志记录器
        comet_logger = None
        if config.use_comet:
            from importlib import import_module
            comet_ml = import_module('comet_ml')
            comet_logger = comet_ml.Experiment(
                api_key=config.comet_config['api_key'],
                project_name=config.comet_config['project_name'],
                workspace=config.comet_config['workspace'],
            )
            comet_logger.add_tag(config.comet_tag)
            comet_logger.set_name(config.comet_name)
            comet_logger.log_parameters(config.__dict__)
            logger.info("Comet日志记录器初始化成功")
        
        # 初始化模型
        tokenizer = KronosTokenizer.from_pretrained(config.finetuned_tokenizer_path)
        tokenizer.eval().to(device)
        
        model = Kronos.from_pretrained(config.pretrained_predictor_path)
        model.to(device)
        logger.info(f"预测模型大小: {get_model_size(model)}")
        
        # 训练
        from train_predictor_cpu import train_model
        dt_result = train_model(model, tokenizer, device, config.__dict__, save_dir, comet_logger)
        
        logger.info("预测模型CPU训练完成")
        return True
    except Exception as e:
        logger.error(f"预测模型CPU训练出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_pipeline():
    """运行完整训练流水线"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置配置
    config = Config()
    
    # 如果指定了配置文件，从文件加载配置
    if args.config_path:
        try:
            with open(args.config_path, 'r') as f:
                config_dict = json.load(f)
            # 更新配置
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            logger.info(f"从 {args.config_path} 加载配置成功")
        except Exception as e:
            logger.error(f"加载配置文件时出错: {str(e)}")
            return 1
    
    # 快速模式设置
    if args.fast_mode:
        config.epochs = 1
        config.max_train_batches = 10
        config.max_val_batches = 5
        logger.info("启用快速模式，减少训练轮数和批次")
    
    # 禁用Comet（默认不使用）
    config.use_comet = False
    
    # 保存流水线配置
    save_pipeline_config(config, config.save_path)
    
    # 步骤1：数据处理
    if not args.skip_data_process:
        if not process_data(config, args.data_source):
            logger.error("数据处理失败，流水线终止")
            return 1
    else:
        logger.info("跳过数据处理步骤")
    
    # 步骤2：训练分词模型
    if not args.skip_tokenizer:
        if args.cpu:
            if not train_tokenizer_cpu(config):
                logger.error("分词模型CPU训练失败，流水线终止")
                return 1
        else:
            if not train_tokenizer_gpu(config):
                logger.error("分词模型GPU训练失败，流水线终止")
                return 1
    else:
        logger.info("跳过分词模型训练步骤")
    
    # 步骤3：训练预测模型
    if not args.skip_predictor:
        if args.cpu:
            if not train_predictor_cpu(config):
                logger.error("预测模型CPU训练失败，流水线终止")
                return 1
        else:
            if not train_predictor_gpu(config):
                logger.error("预测模型GPU训练失败，流水线终止")
                return 1
    else:
        logger.info("跳过预测模型训练步骤")
    
    logger.info("完整训练流水线执行成功")
    return 0

if __name__ == '__main__':
    sys.exit(run_pipeline())