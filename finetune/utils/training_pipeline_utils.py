import os
import sys
import json
import time
import logging
import torch
import random
import numpy as np
import pandas as pd
import torch.distributed as dist
from pathlib import Path
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Ensure project root is in path
sys.path.append("../")
from model.kronos import KronosTokenizer, Kronos
from common_data_processor import FinancialDataset

# Setup logger
logger = logging.getLogger('KronosPipeline')

def setup_logging(log_level=logging.INFO):
    """
    设置日志配置
    
    Args:
        log_level: 日志级别
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('KronosPipeline')

def setup_ddp():
    """
    初始化分布式数据并行环境
    
    Returns:
        tuple: (rank, world_size, local_rank)
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available.")

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    logger.info(
        f"[DDP Setup] Global Rank: {rank}/{world_size}, "
        f"Local Rank (GPU): {local_rank} on device {torch.cuda.current_device()}"
    )
    return rank, world_size, local_rank

def cleanup_ddp():
    """清理分布式进程组"""
    if dist.is_initialized():
        dist.destroy_process_group()

def set_seed(seed: int, rank: int = 0):
    """
    设置随机种子以确保可重复性
    
    Args:
        seed: 基础种子值
        rank: 进程排名，用于确保不同进程有不同的种子
    """
    actual_seed = seed + rank
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_model_size(model: torch.nn.Module) -> str:
    """
    计算PyTorch模型中可训练参数的数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        str: 表示模型大小的字符串（例如"175.0B"，"7.1M"，"50.5K"）
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if total_params >= 1e9:
        return f"{total_params / 1e9:.1f}B"  # Billions
    elif total_params >= 1e6:
        return f"{total_params / 1e6:.1f}M"  # Millions
    else:
        return f"{total_params / 1e3:.1f}K"  # Thousands

def reduce_tensor(tensor: torch.Tensor, world_size: int, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """
    在分布式设置中减少张量的值
    
    Args:
        tensor: 要减少的张量
        world_size: 总进程数
        op: 归约操作（SUM，AVG等）
        
    Returns:
        torch.Tensor: 归约后的张量
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=op)
    if op == dist.ReduceOp.SUM and hasattr(dist.ReduceOp, 'AVG'):
        rt /= world_size
    return rt

def format_time(seconds: float) -> str:
    """
    将秒数格式化为人类可读的H:M:S字符串
    
    Args:
        seconds: 总秒数
        
    Returns:
        str: 格式化的时间字符串（例如"0:15:32"）
    """
    return str(timedelta(seconds=int(seconds)))

def create_dataloaders_ddp(config: dict, rank: int, world_size: int):
    """
    为训练和验证创建分布式数据加载器
    
    Args:
        config: 配置参数字典
        rank: 当前进程的全局排名
        world_size: 总进程数
        
    Returns:
        tuple: (train_loader, val_loader, train_dataset, valid_dataset)
    """
    logger.info(f"[Rank {rank}] 创建分布式数据加载器...")
    train_dataset = FinancialDataset('train', config=config)
    valid_dataset = FinancialDataset('val', config=config)
    logger.info(f"[Rank {rank}] 训练数据集大小: {len(train_dataset)}, 验证数据集大小: {len(valid_dataset)}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        shuffle=False,  # 由采样器处理洗牌
        num_workers=config.get('num_workers', 2),
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        pin_memory=True,
        drop_last=False
    )
    logger.info(f"[Rank {rank}] 数据加载器创建完成。每轮训练步数: {len(train_loader)}, 验证步数: {len(val_loader)}")
    return train_loader, val_loader, train_dataset, valid_dataset

def create_dataloaders_cpu(config: dict):
    """
    为CPU训练创建标准（非分布式）数据加载器
    
    Args:
        config: 配置参数字典
        
    Returns:
        tuple: (train_loader, val_loader, train_dataset, valid_dataset)
    """
    logger.info("[CPU] 创建数据加载器...")
    train_dataset = FinancialDataset('train', config=config)
    valid_dataset = FinancialDataset('val', config=config)
    logger.info(f"[CPU] 训练数据集大小: {len(train_dataset)}, 验证数据集大小: {len(valid_dataset)}")

    num_workers = config.get('num_workers', 0) or 0
    pin_memory = False  # 在CPU上，pin_memory没有好处

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    logger.info(f"[CPU] 数据加载器创建完成。每轮训练步数: {len(train_loader)}, 验证步数: {len(val_loader)}")
    return train_loader, val_loader, train_dataset, valid_dataset

def setup_comet_logger(config):
    """
    设置Comet.ml日志记录器
    
    Args:
        config: 配置参数字典
        
    Returns:
        comet_ml.Experiment或None: Comet日志记录器实例
    """
    if not config.get('use_comet', False):
        return None
        
    try:
        import comet_ml
        logger.info("初始化Comet.ml日志记录器...")
        comet_logger = comet_ml.Experiment(
            api_key=config['comet_config']['api_key'],
            project_name=config['comet_config']['project_name'],
            workspace=config['comet_config']['workspace'],
        )
        comet_logger.add_tag(config['comet_tag'])
        comet_logger.set_name(config['comet_name'])
        comet_logger.log_parameters(config)
        logger.info("Comet.ml日志记录器初始化成功")
        return comet_logger
    except ImportError:
        logger.warning("无法导入comet_ml，将不使用Comet日志记录")
        return None
    except Exception as e:
        logger.error(f"设置Comet日志记录器时出错: {str(e)}")
        return None

def save_model_checkpoint(model, save_path, is_ddp=False):
    """
    保存模型检查点
    
    Args:
        model: 要保存的模型
        save_path: 保存路径
        is_ddp: 是否是DDP包装的模型
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if is_ddp:
            model.module.save_pretrained(save_path)
        else:
            model.save_pretrained(save_path)
        logger.info(f"模型已保存到 {save_path}")
        return True
    except Exception as e:
        logger.error(f"保存模型时出错: {str(e)}")
        return False

def save_training_summary(save_dir, summary_data):
    """
    保存训练摘要到JSON文件
    
    Args:
        save_dir: 保存目录
        summary_data: 要保存的摘要数据字典
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'training_summary.json'), 'w') as f:
            json.dump(summary_data, f, indent=4)
        logger.info(f"训练摘要已保存到 {os.path.join(save_dir, 'training_summary.json')}")
        return True
    except Exception as e:
        logger.error(f"保存训练摘要时出错: {str(e)}")
        return False

def predict_latest_data(model, tokenizer, data, config):
    """
    使用训练好的模型对最新数据进行预测
    
    Args:
        model: 预测模型
        tokenizer: 分词模型
        data: 要预测的数据
        config: 配置参数
        
    Returns:
        dict: 预测结果
    """
    try:
        model.eval()
        tokenizer.eval()
        
        with torch.no_grad():
            # 这里添加预测逻辑
            # 根据实际需求实现
            pass
            
        return {"status": "success", "message": "预测完成"}
    except Exception as e:
        logger.error(f"预测时出错: {str(e)}")
        return {"status": "error", "message": str(e)}

def save_pipeline_config(config, save_dir):
    """
    保存完整的流水线配置
    
    Args:
        config: 配置对象
        save_dir: 保存目录
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        
        # 处理不可序列化的对象
        for key in list(config_dict.keys()):
            try:
                json.dumps({key: config_dict[key]})
            except (TypeError, OverflowError):
                config_dict[key] = str(config_dict[key])
        
        with open(os.path.join(save_dir, 'pipeline_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=4)
        logger.info(f"流水线配置已保存到 {os.path.join(save_dir, 'pipeline_config.json')}")
        return True
    except Exception as e:
        logger.error(f"保存流水线配置时出错: {str(e)}")
        return False
