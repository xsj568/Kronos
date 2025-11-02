#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从Hugging Face Hub提取Kronos模型配置参数
简化版本：只支持从网上抽取mini、small、base模型配置
"""

import os
import json
import torch
import argparse
import sys
from typing import Dict, Any

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from model.kronos import KronosTokenizer, Kronos


def extract_from_huggingface(model_name: str, model_type: str) -> Dict[str, Any]:
    """
    从 Hugging Face Hub 下载并提取模型配置
    
    Args:
        model_name: Hugging Face 模型名称
        model_type: 模型类型，'tokenizer' 或 'predictor'
    
    Returns:
        dict: 模型配置参数
    """
    print(f"正在从 Hugging Face Hub 下载 {model_name} 并提取配置...")
    
    try:
        if model_type == 'tokenizer':
            model = KronosTokenizer.from_pretrained(model_name)
            config = {
                'd_in': model.d_in,
                'd_model': model.d_model,
                'n_heads': model.n_heads,
                'ff_dim': model.ff_dim,
                'n_enc_layers': model.enc_layers,
                'n_dec_layers': model.dec_layers,
                'ffn_dropout_p': model.ffn_dropout_p,
                'attn_dropout_p': model.attn_dropout_p,
                'resid_dropout_p': model.resid_dropout_p,
                's1_bits': model.s1_bits,
                's2_bits': model.s2_bits,
                'beta': model.tokenizer.bsq.beta,
                'gamma0': model.tokenizer.bsq.gamma0,
                'gamma': model.tokenizer.bsq.gamma,
                'zeta': model.tokenizer.bsq.zeta,
                'group_size': model.tokenizer.bsq.group_size,
            }
        else:  # predictor
            model = Kronos.from_pretrained(model_name)
            config = {
                's1_bits': model.s1_bits,
                's2_bits': model.s2_bits,
                'n_layers': model.n_layers,
                'd_model': model.d_model,
                'n_heads': model.n_heads,
                'ff_dim': model.ff_dim,
                'ffn_dropout_p': model.ffn_dropout_p,
                'attn_dropout_p': model.attn_dropout_p,
                'resid_dropout_p': model.resid_dropout_p,
                'token_dropout_p': model.token_dropout_p,
                'learn_te': model.learn_te,
            }
        
        print(f"{model_type} 模型配置提取完成")
        return config
        
    except Exception as e:
        print(f"从 Hugging Face Hub 提取配置时出错: {str(e)}")
        return None


def extract_all_models_from_hf(output_dir: str = './configs') -> Dict[str, Any]:
    """
    从Hugging Face Hub批量下载并提取所有模型配置
    
    Args:
        output_dir: 输出目录路径
    
    Returns:
        dict: 包含所有模型配置的字典
    """
    print("正在从 Hugging Face Hub 批量提取模型配置...")
    
    # 定义模型映射
    model_mappings = {
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
        }
    }
    
    all_configs = {}
    os.makedirs(output_dir, exist_ok=True)
    
    for model_size, models in model_mappings.items():
        print(f"\n处理 {model_size} 模型...")
        model_configs = {}
        
        # 提取tokenizer配置
        try:
            tokenizer_config = extract_from_huggingface(models['tokenizer'], 'tokenizer')
            if tokenizer_config:
                model_configs['tokenizer'] = tokenizer_config
        except Exception as e:
            print(f"提取 {model_size} tokenizer 配置失败: {str(e)}")
        
        # 提取predictor配置
        try:
            predictor_config = extract_from_huggingface(models['predictor'], 'predictor')
            if predictor_config:
                model_configs['predictor'] = predictor_config
        except Exception as e:
            print(f"提取 {model_size} predictor 配置失败: {str(e)}")
        
        if model_configs:
            all_configs[model_size] = model_configs
            print(f"{model_size} 模型配置提取完成")
    
    # 保存所有配置到一个统一的JSON文件
    unified_config_path = os.path.join(output_dir, 'kronos_models_config.json')
    with open(unified_config_path, 'w', encoding='utf-8') as f:
        json.dump(all_configs, f, indent=2, ensure_ascii=False)
    print(f"\n所有模型配置已保存到统一文件: {unified_config_path}")
    
    return all_configs


def create_model_from_config(config: Dict[str, Any], model_type: str = 'predictor'):
    """
    根据配置创建新的模型实例
    
    Args:
        config: 模型配置字典
        model_type: 模型类型，'tokenizer' 或 'predictor'
    
    Returns:
        创建的模型实例
    """
    print(f"正在根据配置创建 {model_type} 模型...")
    
    try:
        if model_type == 'tokenizer':
            model = KronosTokenizer(
                d_in=config['d_in'],
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                ff_dim=config['ff_dim'],
                n_enc_layers=config['n_enc_layers'],
                n_dec_layers=config['n_dec_layers'],
                ffn_dropout_p=config['ffn_dropout_p'],
                attn_dropout_p=config['attn_dropout_p'],
                resid_dropout_p=config['resid_dropout_p'],
                s1_bits=config['s1_bits'],
                s2_bits=config['s2_bits'],
                beta=config['beta'],
                gamma0=config['gamma0'],
                gamma=config['gamma'],
                zeta=config['zeta'],
                group_size=config['group_size']
            )
        else:  # predictor
            model = Kronos(
                s1_bits=config['s1_bits'],
                s2_bits=config['s2_bits'],
                n_layers=config['n_layers'],
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                ff_dim=config['ff_dim'],
                ffn_dropout_p=config['ffn_dropout_p'],
                attn_dropout_p=config['attn_dropout_p'],
                resid_dropout_p=config['resid_dropout_p'],
                token_dropout_p=config['token_dropout_p'],
                learn_te=config['learn_te']
            )
        
        print(f"成功创建 {model_type} 模型")
        return model
        
    except Exception as e:
        print(f"创建 {model_type} 模型失败: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description='从 Hugging Face Hub 提取 Kronos 模型配置')
    parser.add_argument('--output-dir', type=str, default='./configs', 
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 批量提取所有模型配置
    all_configs = extract_all_models_from_hf(args.output_dir)
    
    if all_configs:
        print(f"\n成功提取 {len(all_configs)} 个模型配置:")
        for model_size in all_configs.keys():
            print(f"  - {model_size}")
    else:
        print("未能提取任何模型配置")


if __name__ == '__main__':
    main()