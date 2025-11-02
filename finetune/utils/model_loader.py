#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型加载器 - 支持从多个源加载模型
支持: Hugging Face, ModelScope (魔搭社区), 本地路径
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Union, Dict

logger = logging.getLogger('KronosPipeline')

# 尝试导入ModelScope
try:
    from modelscope import snapshot_download
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False
    logger.warning("ModelScope 未安装，无法从魔搭社区下载模型。安装: pip install modelscope")

# 加载预定义的模型配置
_MODEL_CONFIGS = None

def _load_model_configs() -> Dict:
    """加载预定义的模型配置"""
    global _MODEL_CONFIGS
    if _MODEL_CONFIGS is not None:
        return _MODEL_CONFIGS
    
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs',
        'kronos_models_config.json'
    )
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                _MODEL_CONFIGS = json.load(f)
            logger.info(f"成功加载模型配置: {config_path}")
        else:
            logger.warning(f"配置文件不存在: {config_path}，将使用默认配置")
            _MODEL_CONFIGS = {}
    except Exception as e:
        logger.warning(f"加载配置文件失败: {str(e)}，将使用默认配置")
        _MODEL_CONFIGS = {}
    
    return _MODEL_CONFIGS


def _get_model_config_from_path(model_path: str, model_type: str) -> Optional[Dict]:
    """
    从模型路径推断配置
    
    Args:
        model_path: 模型路径（如 'NeoQuasar/Kronos-Tokenizer-base'）
        model_type: 模型类型 ('tokenizer' 或 'predictor')
    
    Returns:
        配置字典，如果找不到则返回 None
    """
    configs = _load_model_configs()
    if not configs:
        return None
    
    # 从路径推断模型版本
    # 'NeoQuasar/Kronos-Tokenizer-base' -> 'base'
    # 'NeoQuasar/Kronos-Tokenizer-2k' -> 'mini'
    # 'NeoQuasar/Kronos-base' -> 'base'
    # 'NeoQuasar/Kronos-mini' -> 'mini'
    # 'NeoQuasar/Kronos-small' -> 'small'
    
    model_name = model_path.split('/')[-1] if '/' in model_path else model_path
    
    version_map = {
        'Kronos-Tokenizer-2k': 'mini',
        'Kronos-Tokenizer-base': 'base',  # base 和 small 都使用 Tokenizer-base
        'Kronos-mini': 'mini',
        'Kronos-small': 'small',
        'Kronos-base': 'base'
    }
    
    version = None
    for key, val in version_map.items():
        if key in model_name:
            version = val
            break
    
    # 特殊处理：如果无法确定版本，尝试从路径中提取
    if version is None:
        if '2k' in model_name.lower():
            version = 'mini'
        elif 'mini' in model_name.lower():
            version = 'mini'
        elif 'small' in model_name.lower():
            version = 'small'
        elif 'base' in model_name.lower():
            version = 'base'
    
    if version and version in configs:
        if model_type in configs[version]:
            return configs[version][model_type]
    
    return None


def detect_model_source(model_path: str) -> str:
    """
    检测模型路径的来源类型
    
    Args:
        model_path: 模型路径或标识符
        
    Returns:
        str: 'local', 'huggingface', 或 'modelscope'
    """
    # 检查是否是本地路径
    if os.path.exists(model_path):
        return 'local'
    
    # 检查是否是魔搭社区的格式（通常是用户名/模型名或特定前缀）
    if '/' in model_path:
        # ModelScope 通常使用中文或特定的命名格式
        # 也可以通过配置指定使用 ModelScope
        # 这里我们使用简单的启发式规则
        parts = model_path.split('/')
        if len(parts) == 2:
            # 如果路径中包含中文字符或特定关键字，可能是魔搭
            if any('\u4e00' <= char <= '\u9fff' for char in model_path):
                return 'modelscope'
            # 默认认为是 Hugging Face
            return 'huggingface'
    
    return 'huggingface'


def load_model_from_source(
    model_class,
    model_path: str,
    source: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    fallback_on_error: bool = True,
    **kwargs
):
    """
    从指定来源加载模型，支持智能回退
    
    Args:
        model_class: 模型类（如 KronosTokenizer, Kronos）
        model_path: 模型路径或标识符
        source: 模型来源 ('local', 'huggingface', 'modelscope')，None则自动检测
        cache_dir: 缓存目录
        local_files_only: 是否仅使用本地文件
        fallback_on_error: 如果指定来源加载失败，是否尝试其他来源（仅当source='local'时有效）
        **kwargs: 其他传递给模型加载的参数
        
    Returns:
        加载的模型实例
    """
    # 自动检测来源
    if source is None:
        source = detect_model_source(model_path)
        logger.info(f"自动检测模型来源: {source}")
    
    logger.info(f"从 {source} 加载模型: {model_path}")
    
    try:
        if source == 'local':
            # 从本地路径加载
            try:
                return _load_from_local(model_class, model_path, **kwargs)
            except (FileNotFoundError, OSError) as e:
                if fallback_on_error:
                    logger.warning(f"本地路径不存在或加载失败: {model_path}")
                    logger.info("尝试自动检测其他可用的模型来源...")
                    # 智能回退：尝试从 Hugging Face 或 ModelScope 加载
                    fallback_source = detect_model_source(model_path)
                    if fallback_source != 'local':
                        logger.info(f"回退到 {fallback_source} 加载模型")
                        return load_model_from_source(
                            model_class, model_path, source=fallback_source,
                            cache_dir=cache_dir, local_files_only=local_files_only,
                            fallback_on_error=False, **kwargs
                        )
                raise
            
        elif source == 'huggingface':
            # 从 Hugging Face 加载
            return _load_from_huggingface(
                model_class, model_path, cache_dir, local_files_only, **kwargs
            )
            
        elif source == 'modelscope':
            # 从魔搭社区加载
            if not MODELSCOPE_AVAILABLE:
                raise ImportError(
                    "ModelScope 未安装。请安装: pip install modelscope"
                )
            return _load_from_modelscope(
                model_class, model_path, cache_dir, **kwargs
            )
            
        else:
            raise ValueError(f"不支持的模型来源: {source}")
            
    except Exception as e:
        logger.error(f"从 {source} 加载模型失败: {str(e)}")
        raise


def _load_from_local(model_class, model_path: str, **kwargs):
    """从本地路径加载模型"""
    logger.info(f"从本地加载模型: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"本地模型路径不存在: {model_path}")
    
    # 使用 from_pretrained，设置 local_files_only=True
    return model_class.from_pretrained(
        model_path,
        local_files_only=True,
        **kwargs
    )


def _load_from_huggingface(
    model_class,
    model_path: str,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    **kwargs
):
    """从 Hugging Face 加载模型"""
    logger.info(f"从 Hugging Face 加载模型: {model_path}")
    
    # 构建加载参数
    load_kwargs = {
        'local_files_only': local_files_only,
        **kwargs
    }
    
    if cache_dir:
        load_kwargs['cache_dir'] = cache_dir
    
    try:
        # 首先尝试使用标准的 from_pretrained 方法
        return model_class.from_pretrained(model_path, **load_kwargs)
    except (TypeError, ValueError) as e:
        # 如果失败，可能是缺少配置文件，尝试使用预定义配置
        if "missing" in str(e).lower() and "required" in str(e).lower():
            logger.warning(f"从 Hugging Face 加载模型时缺少配置参数: {str(e)}")
            logger.info("尝试使用预定义配置加载模型...")
            
            # 确定模型类型
            model_type = 'tokenizer' if 'Tokenizer' in model_path else 'predictor'
            config = _get_model_config_from_path(model_path, model_type)
            
            if config:
                logger.info(f"使用预定义配置实例化模型: {model_type}")
                # 使用配置实例化模型
                model = model_class(**config)
                
                # 尝试加载权重
                try:
                    from huggingface_hub import hf_hub_download
                    import torch
                    
                    # 下载模型权重文件
                    weight_file = hf_hub_download(
                        repo_id=model_path,
                        filename="model.safetensors",
                        cache_dir=cache_dir,
                        local_files_only=local_files_only
                    )
                    
                    # 加载权重
                    from safetensors.torch import load_file
                    state_dict = load_file(weight_file)
                    model.load_state_dict(state_dict, strict=False)
                    logger.info("成功加载模型权重")
                    
                except Exception as weight_error:
                    logger.warning(f"加载模型权重失败: {str(weight_error)}，将使用未初始化的模型")
                
                return model
            else:
                logger.error(f"无法找到模型配置: {model_path} (类型: {model_type})")
                raise ValueError(f"无法加载模型 {model_path}: 缺少配置文件且无法找到预定义配置")
        else:
            # 其他类型的错误，直接抛出
            raise


def _load_from_modelscope(
    model_class,
    model_path: str,
    cache_dir: Optional[str] = None,
    **kwargs
):
    """从魔搭社区加载模型"""
    logger.info(f"从魔搭社区加载模型: {model_path}")
    
    # 设置缓存目录
    if cache_dir is None:
        cache_dir = os.path.expanduser('~/.cache/modelscope')
    
    try:
        # 使用 ModelScope 下载模型
        logger.info("正在从魔搭社区下载模型...")
        local_model_dir = snapshot_download(
            model_path,
            cache_dir=cache_dir
        )
        logger.info(f"模型已下载到: {local_model_dir}")
        
        # 从下载的本地路径加载模型
        return model_class.from_pretrained(
            local_model_dir,
            local_files_only=True,
            **kwargs
        )
        
    except Exception as e:
        logger.error(f"从魔搭社区下载模型失败: {str(e)}")
        raise


def load_tokenizer(
    tokenizer_path: str,
    source: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    fallback_on_error: bool = True
):
    """
    加载分词器（Tokenizer）
    
    Args:
        tokenizer_path: 分词器路径或标识符
        source: 模型来源
        cache_dir: 缓存目录
        local_files_only: 是否仅使用本地文件
        fallback_on_error: 如果本地加载失败，是否尝试其他来源
        
    Returns:
        KronosTokenizer 实例
    """
    from model.kronos import KronosTokenizer
    
    return load_model_from_source(
        KronosTokenizer,
        tokenizer_path,
        source=source,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        fallback_on_error=fallback_on_error
    )


def load_predictor(
    predictor_path: str,
    source: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    fallback_on_error: bool = True
):
    """
    加载预测器（Predictor）
    
    Args:
        predictor_path: 预测器路径或标识符
        source: 模型来源
        cache_dir: 缓存目录
        local_files_only: 是否仅使用本地文件
        fallback_on_error: 如果本地加载失败，是否尝试其他来源
        
    Returns:
        Kronos 实例
    """
    from model.kronos import Kronos
    
    return load_model_from_source(
        Kronos,
        predictor_path,
        source=source,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        fallback_on_error=fallback_on_error
    )


# 魔搭社区常用模型映射（可选）
MODELSCOPE_MODEL_MAP = {
    # Hugging Face -> ModelScope 映射
    # 'NeoQuasar/Kronos-Tokenizer-2k': 'modelscope_user/kronos-tokenizer-2k',
    # 'NeoQuasar/Kronos-mini': 'modelscope_user/kronos-mini',
    # 可以根据需要添加映射
}


def get_modelscope_path(huggingface_path: str) -> Optional[str]:
    """
    获取 Hugging Face 模型对应的魔搭社区路径
    
    Args:
        huggingface_path: Hugging Face 模型路径
        
    Returns:
        Optional[str]: 魔搭社区路径，如果没有映射则返回 None
    """
    return MODELSCOPE_MODEL_MAP.get(huggingface_path)


if __name__ == '__main__':
    # 测试模型加载器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    logger.info("=== 测试模型加载器 ===")
    
    # 测试来源检测
    test_paths = [
        './local/model/path',
        'NeoQuasar/Kronos-Tokenizer-2k',
        'user/model-name',
        '用户名/模型名'
    ]
    
    for path in test_paths:
        source = detect_model_source(path)
        logger.info(f"路径: {path} -> 来源: {source}")

