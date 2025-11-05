#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPU工具模块 - 智能GPU选择和管理
"""

import os
import logging
import subprocess
import torch
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def get_gpu_memory_info() -> List[Tuple[int, float, float]]:
    """
    获取所有GPU的内存使用信息
    
    Returns:
        List[Tuple[int, float, float]]: [(gpu_id, used_memory_mb, total_memory_mb), ...]
    """
    if not torch.cuda.is_available():
        return []
    
    gpu_info = []
    try:
        # 使用nvidia-smi获取GPU信息
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.strip().split(',')
                gpu_id = int(parts[0].strip())
                used_memory = float(parts[1].strip())
                total_memory = float(parts[2].strip())
                gpu_info.append((gpu_id, used_memory, total_memory))
    except (subprocess.SubprocessError, FileNotFoundError, ValueError) as e:
        logger.warning(f"无法通过nvidia-smi获取GPU信息: {e}，回退到PyTorch方法")
        # 回退到PyTorch方法
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                used_memory = torch.cuda.memory_allocated(i) / (1024 ** 2)  # MB
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)  # MB
                gpu_info.append((i, used_memory, total_memory))
            except Exception as e:
                logger.warning(f"获取GPU {i} 信息失败: {e}")
    
    return gpu_info


def select_best_gpu(min_free_memory_gb: float = 10.0) -> Optional[int]:
    """
    选择空闲内存最多的GPU
    
    Args:
        min_free_memory_gb: 最小空闲内存要求（GB）
        
    Returns:
        int or None: 最佳GPU的ID，如果没有合适的GPU则返回None
    """
    gpu_info = get_gpu_memory_info()
    
    if not gpu_info:
        return None
    
    # 计算每个GPU的空闲内存
    gpu_free_memory = []
    for gpu_id, used_memory, total_memory in gpu_info:
        free_memory = total_memory - used_memory
        free_memory_gb = free_memory / 1024  # 转换为GB
        gpu_free_memory.append((gpu_id, free_memory_gb, total_memory / 1024))
        logger.info(f"GPU {gpu_id}: 空闲 {free_memory_gb:.2f} GB / 总计 {total_memory / 1024:.2f} GB")
    
    # 按空闲内存排序，选择空闲内存最多的GPU
    gpu_free_memory.sort(key=lambda x: x[1], reverse=True)
    
    best_gpu_id, best_free_memory, best_total_memory = gpu_free_memory[0]
    
    if best_free_memory < min_free_memory_gb:
        logger.warning(f"所有GPU空闲内存不足 {min_free_memory_gb} GB，最佳GPU {best_gpu_id} 只有 {best_free_memory:.2f} GB 空闲")
        return None
    
    logger.info(f"选择GPU {best_gpu_id}，空闲内存: {best_free_memory:.2f} GB")
    return best_gpu_id


def get_available_gpus(min_free_memory_gb: float = 10.0) -> List[int]:
    """
    获取所有符合内存要求的GPU列表（按空闲内存从大到小排序）
    
    Args:
        min_free_memory_gb: 最小空闲内存要求（GB）
        
    Returns:
        List[int]: 可用GPU的ID列表（按空闲内存降序排列）
    """
    gpu_info = get_gpu_memory_info()
    
    if not gpu_info:
        return []
    
    # 收集符合条件的GPU及其空闲内存
    available_gpus_with_memory = []
    for gpu_id, used_memory, total_memory in gpu_info:
        free_memory_gb = (total_memory - used_memory) / 1024
        if free_memory_gb >= min_free_memory_gb:
            available_gpus_with_memory.append((gpu_id, free_memory_gb))
            logger.info(f"GPU {gpu_id} 可用，空闲内存: {free_memory_gb:.2f} GB")
        else:
            logger.info(f"GPU {gpu_id} 空闲内存不足: {free_memory_gb:.2f} GB < {min_free_memory_gb} GB")
    
    # 按空闲内存降序排序
    available_gpus_with_memory.sort(key=lambda x: x[1], reverse=True)
    
    # 只返回GPU ID列表
    return [gpu_id for gpu_id, _ in available_gpus_with_memory]


def set_cuda_visible_devices(gpu_ids: List[int]):
    """
    设置CUDA_VISIBLE_DEVICES环境变量
    
    Args:
        gpu_ids: GPU ID列表
    """
    if gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
        logger.info(f"设置CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        logger.warning("没有可用的GPU")


def setup_gpu_for_training(
    min_free_memory_gb: float = 10.0,
    use_all_available: bool = False
) -> Tuple[Optional[torch.device], bool, List[int]]:
    """
    为训练设置最佳GPU配置
    
    Args:
        min_free_memory_gb: 最小空闲内存要求（GB）
        use_all_available: 是否使用所有符合条件的GPU（用于DataParallel）
        
    Returns:
        Tuple[Optional[torch.device], bool, List[int]]: 
            (设备对象, 是否可以使用多GPU, 可用GPU列表)
    """
    if not torch.cuda.is_available():
        logger.info("CUDA不可用，使用CPU")
        return torch.device("cpu"), False, []
    
    # 获取可用GPU列表
    available_gpus = get_available_gpus(min_free_memory_gb)
    
    if not available_gpus:
        logger.warning(f"没有符合要求的GPU（最小空闲内存: {min_free_memory_gb} GB），回退到CPU")
        return torch.device("cpu"), False, []
    
    if use_all_available and len(available_gpus) > 1:
        # 使用所有可用GPU
        logger.info(f"检测到 {len(available_gpus)} 个可用GPU: {available_gpus}")
        # 设置主GPU为第一个可用GPU
        primary_device = torch.device(f"cuda:{available_gpus[0]}")
        return primary_device, True, available_gpus
    else:
        # 只使用单个最佳GPU
        best_gpu = available_gpus[0]  # get_available_gpus已按空闲内存排序
        logger.info(f"使用单个GPU: {best_gpu}")
        device = torch.device(f"cuda:{best_gpu}")
        return device, False, [best_gpu]


def clear_gpu_cache():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("已清理GPU缓存")


def print_gpu_memory_summary():
    """打印GPU内存使用摘要"""
    if not torch.cuda.is_available():
        logger.info("CUDA不可用")
        return
    
    logger.info("=" * 80)
    logger.info("GPU内存使用摘要:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)  # GB
        max_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)  # GB
        total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # GB
        logger.info(f"GPU {i}: 已分配 {allocated:.2f} GB, 已预留 {reserved:.2f} GB, "
                   f"峰值 {max_allocated:.2f} GB, 总计 {total:.2f} GB")
    logger.info("=" * 80)

