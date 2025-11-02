#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kronos 金融模型完整训练流程
包括：配置，数据源，数据处理，分词模型训练，预测模型训练，预测最新的结果
"""

import os
import shutil
import sys
import json
import time
import logging
import argparse
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
import traceback
from time import strftime
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

# 确保项目根目录在路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimized_config import OptimizedConfig, create_config_from_args, parse_args
from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference
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
    predict_future_trends,
    save_pipeline_config,
    evaluate_tokenizer_on_test_data,
    evaluate_predictor_on_test_data,
    evaluate_model_on_test_data,
    evaluate_models_during_training,
    update_best_model_paths,
    get_shanghai_time
)
from utils.model_loader import load_tokenizer, load_predictor, load_model_from_source
from common_data_processor import DataProcessorFactory, FinancialDataset
from prediction_incremental_updater import PredictionIncrementalUpdater

# 全局日志记录器
logger = logging.getLogger('KronosPipeline')


class KronosTrainingPipeline:
    """
    Kronos模型训练完整流水线
    """
    
    def __init__(self, config, use_gpu=True, data_source='qlib', early_stopping_patience=3):
        """
        初始化训练流水线
        
        Args:
            config: 配置对象
            use_gpu: 是否使用GPU训练
            data_source: 数据源类型，'qlib'或'sina'
            early_stopping_patience: 提前终止的耐心值
        """
        self.config = config
        self.use_gpu = use_gpu
        self.data_source = data_source
        self.early_stopping_patience = early_stopping_patience
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        
        # 支持 Mac M4 MPS 和 NVIDIA CUDA
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                self.gpu_type = "cuda"
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.gpu_type = "mps"
            else:
                self.device = torch.device("cpu")
                self.gpu_type = "cpu"
        else:
            self.device = torch.device("cpu")
            self.gpu_type = "cpu"
            
        # CPU多核优化：设置PyTorch线程数
        if self.gpu_type == "cpu" and hasattr(config, 'torch_threads') and config.torch_threads > 0:
            torch.set_num_threads(config.torch_threads)
            torch.set_num_interop_threads(config.torch_threads)
            logger.info(f"设置PyTorch线程数: {config.torch_threads}")
        
        self.is_master = True  # 单进程或主进程
        
        # 设置分布式训练标志，避免重复检查
        self.use_ddp = (use_gpu and self.gpu_type == "cuda" and torch.cuda.device_count() > 1)
        
        # 日志已在main函数中初始化，这里只记录信息
        logger.info(f"初始化Kronos训练流水线 - GPU: {use_gpu}, GPU类型: {self.gpu_type}, 数据源: {data_source}, DDP: {self.use_ddp}")
        
        # 设置保存路径，使用config中定义的路径
        self.tokenizer_save_dir = os.path.join(config.save_path, config.tokenizer_save_folder_name)
        self.predictor_save_dir = os.path.join(config.save_path, config.predictor_save_folder_name)
        
        # 确保检查点目录存在
        tokenizer_checkpoint_dir = os.path.dirname(config.finetuned_tokenizer_path)
        predictor_checkpoint_dir = os.path.dirname(config.finetuned_predictor_path)
        os.makedirs(tokenizer_checkpoint_dir, exist_ok=True)
        os.makedirs(predictor_checkpoint_dir, exist_ok=True)
        
        # 保存流水线配置
        save_pipeline_config(config, config.save_path)
        
        # 设置随机种子
        set_seed(config.seed)
        
        # 记录模型评估的最佳损失
        self.best_tokenizer_test_loss = float('inf')
        self.best_predictor_test_loss = float('inf')
        
        # 初始化历史最佳模型路径
        # 如果配置中没有设置，则使用当前的最佳模型路径
        if not hasattr(config, 'his_best_tokenizer_path') or not config.his_best_tokenizer_path:
            config.his_best_tokenizer_path = config.finetuned_tokenizer_path
        if not hasattr(config, 'his_best_predictor_path') or not config.his_best_predictor_path:
            config.his_best_predictor_path = config.finetuned_predictor_path
            
        # 创建模型历史记录目录
        if hasattr(config, 'model_history_dir') and config.model_history_dir:
            os.makedirs(config.model_history_dir, exist_ok=True)
    
    def create_model_from_config(self, config_path: str, model_type: str):
        """
        根据配置文件创建模型实例
        
        Args:
            config_path: 配置文件路径
            model_type: 模型类型，'tokenizer' 或 'predictor'
        
        Returns:
            创建的模型实例
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
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
            
            logger.info(f"成功从配置文件创建 {model_type} 模型: {config_path}")
            return model
            
        except Exception as e:
            logger.error(f"从配置文件创建 {model_type} 模型失败: {str(e)}")
            return None
        
    def setup_distributed(self):
        """设置分布式训练环境"""
        if not self.use_gpu or self.gpu_type == "mps":
            if self.gpu_type == "mps":
                logger.info("使用MPS训练，跳过分布式设置（MPS不支持分布式训练）")
            else:
                logger.info("使用CPU训练，跳过分布式设置")
            return
        
        # 检查是否有分布式环境变量，如果没有则禁用DDP
        if not all(key in os.environ for key in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK']):
            logger.warning("未检测到分布式训练环境变量（RANK, WORLD_SIZE, LOCAL_RANK），禁用分布式训练，使用单GPU训练")
            self.use_ddp = False
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.device = torch.device("cuda:0")
            self.is_master = True
            set_seed(self.config.seed, 0)
            return
            
        try:
            self.rank, self.world_size, self.local_rank = setup_ddp()
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.is_master = (self.rank == 0)
            set_seed(self.config.seed, self.rank)
            logger.info(f"分布式训练环境设置完成 - Rank: {self.rank}, World Size: {self.world_size}")
        except Exception as e:
            logger.warning(f"设置分布式训练环境时出错: {str(e)}")
            logger.warning("回退到单GPU训练模式")
            self.use_ddp = False
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.device = torch.device("cuda:0")
            self.is_master = True
            set_seed(self.config.seed, 0)
    
    def process_data(self):
        """处理数据"""
        success = True
        
        if self.is_master:
            logger.info(f"开始处理{self.data_source}数据...")
            try:
                # 使用工厂创建数据处理器
                processor = DataProcessorFactory.create_processor(self.data_source, self.config)
                result = processor.run_pipeline()
                logger.info(f"数据处理完成: {result}")
                
                # 加载测试数据，用于每个训练阶段的评估
                self.load_test_data()
            except Exception as e:
                logger.error(f"处理数据时出错: {str(e)}")
                success = False
        
        # 同步所有进程，确保主进程完成数据处理后其他进程才继续
        if self.use_ddp:
            dist.barrier()
            # 广播主进程的处理结果到所有进程
            success_tensor = torch.tensor([1.0 if success else 0.0], device=self.device)
            dist.broadcast(success_tensor, src=0)
            success = bool(success_tensor.item() > 0.5)
        
        return success
        
    def load_test_data(self):
        """加载测试数据，用于模型评估"""
        if self.is_master:
            try:
                test_data_path = os.path.join(self.config.dataset_path, self.data_source, "test_data.pkl")
                logger.info(f"加载测试数据: {test_data_path}")
                if os.path.exists(test_data_path):
                    with open(test_data_path, 'rb') as f:
                        self.test_data = pickle.load(f)
                    logger.info(f"测试数据加载成功，包含 {len(self.test_data)} 支股票")
                else:
                    logger.warning(f"测试数据文件不存在: {test_data_path}")
                    self.test_data = None
            except Exception as e:
                logger.error(f"加载测试数据时出错: {str(e)}")
                self.test_data = None
    
    def _check_early_stopping_sync(self, early_stopping_counter, patience):
        """
        检查并同步提前终止决策（支持分布式训练）
        
        Args:
            early_stopping_counter: 当前的提前终止计数器
            patience: 耐心值
            
        Returns:
            bool: 是否应该停止训练
        """
        should_stop = False
        
        if self.is_master:
            should_stop = (early_stopping_counter >= patience)
        
        # 在分布式环境中同步提前终止决策
        if self.use_ddp:
            stop_tensor = torch.tensor([1.0 if should_stop else 0.0], device=self.device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item() > 0.5)
        
        return should_stop
    
    def train_tokenizer(self):
        """训练分词模型"""
        start_time = time.time()
        logger.info("开始训练分词模型...")
        
        # 初始化模型
        try:
            if self.config.model_version == 'customer':
                # 从自定义配置文件创建模型
                model = self.create_model_from_config(self.config.custom_tokenizer_config, 'tokenizer')
                if model is None:
                    return False
            else:
                # 使用新的模型加载器，默认从本地加载，支持智能回退
                model_source = getattr(self.config, 'model_source', 'local')
                logger.info(f"从 {model_source} 加载预训练分词模型: {self.config.pretrained_tokenizer_path}")
                # 如果指定为 'local'，启用智能回退；如果为 'auto'，则自动检测
                use_fallback = (model_source == 'local')
                model = load_tokenizer(
                    self.config.pretrained_tokenizer_path,
                    source=model_source if model_source != 'auto' else None,
                    fallback_on_error=use_fallback
                )
            
            model.to(self.device)
            logger.info(f"分词模型初始化完成 - 大小: {get_model_size(model)}")
            
            # CPU优化：使用torch.compile()加速（如果启用）
            if hasattr(self.config, 'use_torch_compile') and self.config.use_torch_compile:
                if hasattr(torch, 'compile'):
                    compile_mode = getattr(self.config, 'torch_compile_mode', 'default')
                    logger.info(f"启用torch.compile()加速 - 模式: {compile_mode}")
                    model = torch.compile(model, mode=compile_mode)
                else:
                    logger.warning("当前PyTorch版本不支持torch.compile()，跳过编译")
        except Exception as e:
            logger.error(f"初始化分词模型时出错: {str(e)}")
            return False
        
        # 设置DDP
        if self.use_ddp:
            model = DDP(model, device_ids=[self.local_rank], find_unused_parameters=False)
        
        # 创建数据加载器
        if self.use_ddp:
            train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders_ddp(
                self.config.__dict__, self.rank, self.world_size
            )
        else:
            train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders_cpu(
                self.config.__dict__
            )
        
        # 设置优化器和调度器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.tokenizer_learning_rate,
            weight_decay=self.config.adam_weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.config.tokenizer_learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=self.config.epochs,
            pct_start=0.03,
            div_factor=10
        )
        
        # 设置Comet日志记录器
        comet_logger = setup_comet_logger(self.config.__dict__) if self.is_master else None
        
        # 训练循环
        best_val_loss = float('inf')
        batch_idx_global_train = 0
        evaluation_history = []  # 记录每个epoch的评估信息
        early_stopping_counter = 0  # 提前终止计数器
        last_test_loss = float('inf')  # 上次测试损失
        
        for epoch_idx in range(self.config.epochs):
            epoch_start_time = time.time()
            model.train()
            
            # 设置数据集种子
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch_idx)
            train_dataset.set_epoch_seed(epoch_idx * 10000 + (self.rank if self.use_gpu else 0))
            valid_dataset.set_epoch_seed(0)  # 保持验证采样一致
            
            # 训练循环
            for i, (ori_batch_x, _) in enumerate(train_loader):
                ori_batch_x = ori_batch_x.squeeze(0).to(self.device)
                
                # 梯度累积循环
                current_batch_total_loss = 0.0
                for j in range(self.config.accumulation_steps):
                    start_idx = j * (ori_batch_x.shape[0] // self.config.accumulation_steps)
                    end_idx = (j + 1) * (ori_batch_x.shape[0] // self.config.accumulation_steps)
                    batch_x = ori_batch_x[start_idx:end_idx]
                    
                    # 前向传播
                    if self.use_ddp:
                        zs, bsq_loss, _, _ = model(batch_x)
                    else:
                        zs, bsq_loss, _, _ = model(batch_x)
                    z_pre, z = zs
                    
                    # 损失计算
                    recon_loss_pre = F.mse_loss(z_pre, batch_x)
                    recon_loss_all = F.mse_loss(z, batch_x)
                    recon_loss = recon_loss_pre + recon_loss_all
                    loss = (recon_loss + bsq_loss) / 2
                    
                    loss_scaled = loss / self.config.accumulation_steps
                    current_batch_total_loss += loss.item()
                    loss_scaled.backward()
                
                # 优化器步骤
                torch.nn.utils.clip_grad_norm_(
                    model.module.parameters() if self.use_ddp else model.parameters(), 
                    max_norm=2.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # 日志记录
                if self.is_master and (batch_idx_global_train + 1) % self.config.log_interval == 0:
                    avg_loss = current_batch_total_loss / self.config.accumulation_steps
                    logger.info(
                        f"[Epoch {epoch_idx + 1}/{self.config.epochs}, Step {i + 1}/{len(train_loader)}] "
                        f"LR {optimizer.param_groups[0]['lr']:.6f}, Loss: {avg_loss:.4f}"
                    )
                if self.is_master and comet_logger:
                    avg_loss = current_batch_total_loss / self.config.accumulation_steps
                    comet_logger.log_metric('train_tokenizer_loss_batch', avg_loss, step=batch_idx_global_train)
                    comet_logger.log_metric('tokenizer_learning_rate', optimizer.param_groups[0]["lr"], step=batch_idx_global_train)
                
                batch_idx_global_train += 1
            
            # 验证循环
            model.eval()
            tot_val_loss = 0.0
            val_sample_count = 0
            
            with torch.no_grad():
                for ori_batch_x, _ in val_loader:
                    ori_batch_x = ori_batch_x.squeeze(0).to(self.device)
                    if self.use_ddp:
                        zs, _, _, _ = model(ori_batch_x)
                    else:
                        zs, _, _, _ = model(ori_batch_x)
                    _, z = zs
                    val_loss_item = F.mse_loss(z, ori_batch_x)
                    
                    tot_val_loss += val_loss_item.item() * ori_batch_x.size(0)
                    val_sample_count += ori_batch_x.size(0)
            
            # 如果是分布式训练，收集所有进程的验证损失
            if self.use_ddp:
                val_loss_sum_tensor = torch.tensor(tot_val_loss, device=self.device)
                val_count_tensor = torch.tensor(val_sample_count, device=self.device)
                dist.all_reduce(val_loss_sum_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_count_tensor, op=dist.ReduceOp.SUM)
                
                tot_val_loss = val_loss_sum_tensor.item()
                val_sample_count = val_count_tensor.item()
            
            avg_val_loss = tot_val_loss / val_sample_count if val_sample_count > 0 else 0
            
            # 主进程进行摘要和检查点保存
            if self.is_master:
                logger.info(f"\n--- Epoch {epoch_idx + 1}/{self.config.epochs} Summary ---")
                logger.info(f"验证损失: {avg_val_loss:.4f}")
                logger.info(f"本轮用时: {format_time(time.time() - epoch_start_time)}")
                logger.info(f"总用时: {format_time(time.time() - start_time)}\n")
                
                if comet_logger:
                    comet_logger.log_metric('val_tokenizer_loss_epoch', avg_val_loss, epoch=epoch_idx)
                
                # 在测试集上评估当前模型
                if hasattr(self, 'test_data') and self.test_data is not None:
                    # 创建临时路径用于当前模型评估，包含epoch信息
                    temp_save_path = f"{self.tokenizer_save_dir}/checkpoints/current_model_epoch_{epoch_idx + 1}"
                    os.makedirs(temp_save_path, exist_ok=True)
                    
                    # 保存当前模型到临时路径用于评估
                    if self.use_ddp:
                        model.module.save_pretrained(temp_save_path)
                    else:
                        model.save_pretrained(temp_save_path)
                    
                    # 使用工具函数评估模型，使用config中定义的路径
                    self.best_tokenizer_test_loss, eval_info = evaluate_models_during_training(
                        epoch_idx=epoch_idx,
                        current_model_path=temp_save_path,
                        config=self.config,
                        test_data=self.test_data,
                        device=self.device,
                        model_type='tokenizer',
                        best_loss=self.best_tokenizer_test_loss,
                        save_path=self.config.finetuned_tokenizer_path
                    )
                    
                    # 记录评估信息
                    evaluation_history.append(eval_info)
                    
                    # 清理临时模型文件（节省磁盘空间）
                    # 只保留最佳模型，删除当前epoch的临时检查点
                    try:
                        shutil.rmtree(temp_save_path, ignore_errors=True)
                        logger.info(f"已清理临时检查点: {temp_save_path}")
                    except Exception as e:
                        logger.warning(f"清理临时文件失败: {e}")
                    
                    # 记录到Comet（如果启用）
                    if comet_logger and os.path.exists(self.config.finetuned_tokenizer_path):
                        comet_logger.log_model("best_model", self.config.finetuned_tokenizer_path)
                
                # 如果没有测试数据，则使用验证损失作为标准
                elif avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_path = self.config.finetuned_tokenizer_path
                    if self.use_ddp:
                        model.module.save_pretrained(save_path)
                    else:
                        model.save_pretrained(save_path)
                    logger.info(f"最佳模型已保存到 {save_path} (验证损失: {best_val_loss:.4f})")
                    if comet_logger:
                        comet_logger.log_model("best_model", save_path)
            
            # 基于测试集的提前终止逻辑
            if hasattr(self, 'test_data') and self.test_data is not None:
                # 如果有测试数据，检查测试损失是否改善
                if self.best_tokenizer_test_loss < float('inf'):
                    if self.best_tokenizer_test_loss >= last_test_loss:
                        early_stopping_counter += 1
                        if self.is_master:
                            logger.info(f"测试损失未改善，提前终止计数器: {early_stopping_counter}/{self.early_stopping_patience}")
                    else:
                        early_stopping_counter = 0  # 重置计数器
                        if self.is_master:
                            logger.info(f"测试损失改善，重置提前终止计数器")
                    
                    last_test_loss = self.best_tokenizer_test_loss
                    
                    # 检查是否需要提前终止（支持分布式同步）
                    if self._check_early_stopping_sync(early_stopping_counter, self.early_stopping_patience):
                        logger.info(f"提前终止训练：连续 {self.early_stopping_patience} 个epoch测试损失未改善")
                        break
            
            # 同步所有进程
            if self.use_ddp:
                dist.barrier()
        
        # 保存训练摘要
        if self.is_master:
            # 从评估历史中找出损失最小的模型
            best_eval = None
            if evaluation_history:
                best_eval = min(evaluation_history, key=lambda x: x.get('best_loss', float('inf')))
            
            shanghai_time = get_shanghai_time()
            summary = {
                'start_time': shanghai_time.strftime("%Y-%m-%dT%H-%M-%S"),
                'end_time': shanghai_time.strftime("%Y-%m-%dT%H-%M-%S"),
                'total_time': format_time(time.time() - start_time),
                'best_val_loss': best_val_loss,
                'best_test_loss': self.best_tokenizer_test_loss if hasattr(self, 'best_tokenizer_test_loss') else None,
                'epochs': self.config.epochs,
                'world_size': self.world_size,
                'device': str(self.device),
                'evaluation_history': evaluation_history,  # 添加评估历史
                'final_best_model': {
                    'epoch': best_eval['epoch'] if best_eval else None,
                    'name': best_eval['best_model_name'] if best_eval else None,
                    'path': best_eval['best_model_path'] if best_eval else None,
                    'loss': best_eval['best_loss'] if best_eval else None,
                }
            }
            save_training_summary(self.tokenizer_save_dir, summary)
            
            # 检查分词模型训练结果
            if os.path.exists(self.config.finetuned_tokenizer_path):
                logger.info(f"分词模型训练完成，最佳模型路径: {self.config.finetuned_tokenizer_path}")
                logger.info(f"最佳分词模型测试损失: {self.best_tokenizer_test_loss:.4f}")
            else:
                logger.warning(f"最佳模型路径不存在: {self.config.finetuned_tokenizer_path}")
            
            if comet_logger:
                comet_logger.end()
        
        return True
    
    def train_predictor(self):
        """训练预测模型"""
        start_time = time.time()
        logger.info("开始训练预测模型...")
        
        # 初始化分词模型和预测模型
        try:
            # 加载已训练好的分词模型（必须存在，不需要回退）
            tokenizer = load_tokenizer(
                self.config.finetuned_tokenizer_path, 
                source='local', 
                local_files_only=True,
                fallback_on_error=False  # 训练好的模型必须存在
            )
            tokenizer.eval().to(self.device)
            logger.info("分词模型加载完成")
            
            if self.config.model_version == 'customer':
                # 从自定义配置文件创建模型
                model = self.create_model_from_config(self.config.custom_predictor_config, 'predictor')
                if model is None:
                    return False
            else:
                # 使用新的模型加载器，默认从本地加载，支持智能回退
                model_source = getattr(self.config, 'model_source', 'local')
                logger.info(f"从 {model_source} 加载预训练预测模型: {self.config.pretrained_predictor_path}")
                # 如果指定为 'local'，启用智能回退；如果为 'auto'，则自动检测
                use_fallback = (model_source == 'local')
                # 修复：local_files_only应该与model_source对应
                local_files_only = (model_source == 'local')
                model = load_predictor(
                    self.config.pretrained_predictor_path,
                    source=model_source if model_source != 'auto' else None,
                    fallback_on_error=use_fallback,
                    local_files_only=local_files_only
                )
            
            model.to(self.device)
            logger.info(f"预测模型初始化完成 - 大小: {get_model_size(model)}")
            
            # CPU优化：使用torch.compile()加速（如果启用）
            if hasattr(self.config, 'use_torch_compile') and self.config.use_torch_compile:
                if hasattr(torch, 'compile'):
                    compile_mode = getattr(self.config, 'torch_compile_mode', 'default')
                    logger.info(f"启用torch.compile()加速 - 模式: {compile_mode}")
                    model = torch.compile(model, mode=compile_mode)
                else:
                    logger.warning("当前PyTorch版本不支持torch.compile()，跳过编译")
        except Exception as e:
            logger.error(f"初始化模型时出错: {str(e)}")
            return False
        
        # 设置DDP
        if self.use_ddp:
            model = DDP(model, device_ids=[self.local_rank], find_unused_parameters=False)
        
        # 创建数据加载器
        if self.use_ddp:
            train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders_ddp(
                self.config.__dict__, self.rank, self.world_size
            )
        else:
            train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders_cpu(
                self.config.__dict__
            )
        
        # 设置优化器和调度器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.predictor_learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.config.predictor_learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=self.config.epochs,
            pct_start=0.03,
            div_factor=10
        )
        
        # 设置Comet日志记录器
        comet_logger = setup_comet_logger(self.config.__dict__) if self.is_master else None
        
        # 训练循环
        best_val_loss = float('inf')
        batch_idx_global = 0
        evaluation_history = []  # 记录每个epoch的评估信息
        early_stopping_counter = 0  # 提前终止计数器
        last_test_loss = float('inf')  # 上次测试损失
        
        for epoch_idx in range(self.config.epochs):
            epoch_start_time = time.time()
            model.train()
            
            # 设置数据集种子
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch_idx)
            train_dataset.set_epoch_seed(epoch_idx * 10000 + (self.rank if self.use_gpu else 0))
            valid_dataset.set_epoch_seed(0)
            
            # 训练循环
            for i, (batch_x, batch_x_stamp) in enumerate(train_loader):
                batch_x = batch_x.squeeze(0).to(self.device)
                batch_x_stamp = batch_x_stamp.squeeze(0).to(self.device)
                
                # 使用分词模型对输入数据进行编码
                with torch.no_grad():
                    token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                
                # 准备输入和目标
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
                
                # 前向传播和损失计算
                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                if self.use_ddp:
                    loss, s1_loss, s2_loss = model.module.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
                else:
                    loss, s1_loss, s2_loss = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.module.parameters() if self.use_ddp else model.parameters(), 
                    max_norm=3.0
                )
                optimizer.step()
                scheduler.step()
                
                # 日志记录
                if self.is_master and (batch_idx_global + 1) % self.config.log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    logger.info(
                        f"[Epoch {epoch_idx + 1}/{self.config.epochs}, Step {i + 1}/{len(train_loader)}] "
                        f"LR {lr:.6f}, Loss: {loss.item():.4f}"
                    )
                if self.is_master and comet_logger:
                    lr = optimizer.param_groups[0]['lr']
                    comet_logger.log_metric('train_predictor_loss_batch', loss.item(), step=batch_idx_global)
                    comet_logger.log_metric('train_S1_loss_each_batch', s1_loss.item(), step=batch_idx_global)
                    comet_logger.log_metric('train_S2_loss_each_batch', s2_loss.item(), step=batch_idx_global)
                    comet_logger.log_metric('predictor_learning_rate', lr, step=batch_idx_global)
                
                batch_idx_global += 1
            
            # 验证循环
            model.eval()
            tot_val_loss = 0.0
            val_batches_processed = 0
            
            with torch.no_grad():
                for batch_x, batch_x_stamp in val_loader:
                    batch_x = batch_x.squeeze(0).to(self.device)
                    batch_x_stamp = batch_x_stamp.squeeze(0).to(self.device)
                    
                    token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                    token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                    token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
                    
                    logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                    if self.use_ddp:
                        val_loss, _, _ = model.module.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
                    else:
                        val_loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
                    
                    tot_val_loss += val_loss.item()
                    val_batches_processed += 1
            
            # 如果是分布式训练，收集所有进程的验证损失
            if self.use_ddp:
                val_loss_sum_tensor = torch.tensor(tot_val_loss, device=self.device)
                val_batches_tensor = torch.tensor(val_batches_processed, device=self.device)
                dist.all_reduce(val_loss_sum_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_batches_tensor, op=dist.ReduceOp.SUM)
                
                tot_val_loss = val_loss_sum_tensor.item()
                val_batches_processed = val_batches_tensor.item()
            
            avg_val_loss = tot_val_loss / val_batches_processed if val_batches_processed > 0 else 0
            
            # 主进程进行摘要和检查点保存
            if self.is_master:
                logger.info(f"\n--- Epoch {epoch_idx + 1}/{self.config.epochs} Summary ---")
                logger.info(f"验证损失: {avg_val_loss:.4f}")
                logger.info(f"本轮用时: {format_time(time.time() - epoch_start_time)}")
                logger.info(f"总用时: {format_time(time.time() - start_time)}\n")
                
                if comet_logger:
                    comet_logger.log_metric('val_predictor_loss_epoch', avg_val_loss, epoch=epoch_idx)
                
                # 在测试集上评估当前模型
                if hasattr(self, 'test_data') and self.test_data is not None:
                    # 创建临时路径用于当前模型评估，包含epoch信息
                    temp_save_path = f"{self.predictor_save_dir}/checkpoints/current_model_epoch_{epoch_idx + 1}"
                    os.makedirs(temp_save_path, exist_ok=True)
                    
                    # 保存当前模型到临时路径用于评估
                    if self.use_ddp:
                        model.module.save_pretrained(temp_save_path)
                    else:
                        model.save_pretrained(temp_save_path)
                    
                    # 使用工具函数评估模型，使用config中定义的路径
                    self.best_predictor_test_loss, eval_info = evaluate_models_during_training(
                        epoch_idx=epoch_idx,
                        current_model_path=temp_save_path,
                        config=self.config,
                        test_data=self.test_data,
                        device=self.device,
                        model_type='predictor',
                        best_loss=self.best_predictor_test_loss,
                        save_path=self.config.finetuned_predictor_path
                    )
                    
                    # 记录评估信息
                    evaluation_history.append(eval_info)
                    
                    # 清理临时模型文件（节省磁盘空间）
                    # 只保留最佳模型，删除当前epoch的临时检查点
                    try:
                        shutil.rmtree(temp_save_path, ignore_errors=True)
                        logger.info(f"已清理临时检查点: {temp_save_path}")
                    except Exception as e:
                        logger.warning(f"清理临时文件失败: {e}")
                    
                    # 记录到Comet（如果启用）
                    if comet_logger and os.path.exists(self.config.finetuned_predictor_path):
                        comet_logger.log_model("best_model", self.config.finetuned_predictor_path)
                
                # 如果没有测试数据，则使用验证损失作为标准
                elif avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_path = self.config.finetuned_predictor_path
                    if self.use_ddp:
                        model.module.save_pretrained(save_path)
                    else:
                        model.save_pretrained(save_path)
                    logger.info(f"最佳模型已保存到 {save_path} (验证损失: {best_val_loss:.4f})")
                    if comet_logger:
                        comet_logger.log_model("best_model", save_path)
            
            # 基于测试集的提前终止逻辑
            if hasattr(self, 'test_data') and self.test_data is not None:
                # 如果有测试数据，检查测试损失是否改善
                if self.best_predictor_test_loss < float('inf'):
                    if self.best_predictor_test_loss >= last_test_loss:
                        early_stopping_counter += 1
                        if self.is_master:
                            logger.info(f"测试损失未改善，提前终止计数器: {early_stopping_counter}/{self.early_stopping_patience}")
                    else:
                        early_stopping_counter = 0  # 重置计数器
                        if self.is_master:
                            logger.info(f"测试损失改善，重置提前终止计数器")
                    
                    last_test_loss = self.best_predictor_test_loss
                    
                    # 检查是否需要提前终止（支持分布式同步）
                    if self._check_early_stopping_sync(early_stopping_counter, self.early_stopping_patience):
                        logger.info(f"提前终止训练：连续 {self.early_stopping_patience} 个epoch测试损失未改善")
                        break
            
            # 同步所有进程
            if self.use_ddp:
                dist.barrier()
        
        # 保存训练摘要
        if self.is_master:
            # 从评估历史中找出损失最小的模型
            best_eval = None
            if evaluation_history:
                best_eval = min(evaluation_history, key=lambda x: x.get('best_loss', float('inf')))
            
            shanghai_time = get_shanghai_time()
            summary = {
                'start_time': shanghai_time.strftime("%Y-%m-%dT%H-%M-%S"),
                'end_time': shanghai_time.strftime("%Y-%m-%dT%H-%M-%S"),
                'total_time': format_time(time.time() - start_time),
                'best_val_loss': best_val_loss,
                'best_test_loss': self.best_predictor_test_loss if hasattr(self, 'best_predictor_test_loss') else None,
                'epochs': self.config.epochs,
                'world_size': self.world_size,
                'device': str(self.device),
                'evaluation_history': evaluation_history,  # 添加评估历史
                'final_best_model': {
                    'epoch': best_eval['epoch'] if best_eval else None,
                    'name': best_eval['best_model_name'] if best_eval else None,
                    'path': best_eval['best_model_path'] if best_eval else None,
                    'loss': best_eval['best_loss'] if best_eval else None,
                }
            }
            save_training_summary(self.predictor_save_dir, summary)
            
            # 检查预测模型训练结果
            if os.path.exists(self.config.finetuned_predictor_path):
                logger.info(f"预测模型训练完成，最佳模型路径: {self.config.finetuned_predictor_path}")
                logger.info(f"最佳预测模型测试损失: {self.best_predictor_test_loss:.4f}")
            else:
                logger.warning(f"最佳模型路径不存在: {self.config.finetuned_predictor_path}")
            
            if comet_logger:
                comet_logger.end()
        
        return True
    
    def evaluate_models(self):
        """验证最佳模型是否已经选择完成"""
        if not self.is_master:
            return True
            
        logger.info("验证最佳模型选择...")
        try:
            # 检查配置中的路径是否已经被更新（应该在训练完成时已更新）
            if not self.config.finetuned_tokenizer_path or not self.config.finetuned_predictor_path:
                logger.error("配置中的模型路径未设置，训练可能未正确完成")
                return False
            
            # 验证路径是否存在
            if not os.path.exists(self.config.finetuned_tokenizer_path):
                logger.error(f"最佳分词模型路径不存在: {self.config.finetuned_tokenizer_path}")
                return False
                
            if not os.path.exists(self.config.finetuned_predictor_path):
                logger.error(f"最佳预测模型路径不存在: {self.config.finetuned_predictor_path}")
                return False
            
            logger.info(f"✓ 最佳分词模型路径: {self.config.finetuned_tokenizer_path}")
            logger.info(f"✓ 最佳预测模型路径: {self.config.finetuned_predictor_path}")
            logger.info(f"✓ 最佳分词模型测试损失: {self.best_tokenizer_test_loss:.4f}")
            logger.info(f"✓ 最佳预测模型测试损失: {self.best_predictor_test_loss:.4f}")
            
            return True
        except Exception as e:
            logger.error(f"模型评估过程中出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _evaluate_predictor_on_test_data(self, predictor_path, tokenizer_path):
        """在测试数据上评估预测模型
        
        Args:
            predictor_path: 预测模型路径
            tokenizer_path: 分词模型路径
            
        Returns:
            float: 测试损失
        """
        # 使用工具函数评估预测模型
        return evaluate_predictor_on_test_data(predictor_path, tokenizer_path, self.test_data, self.config, self.device)
    
    def _evaluate_tokenizer_on_test_data(self, tokenizer_path):
        """在测试数据上评估分词模型
        
        Args:
            tokenizer_path: 分词模型路径
            
        Returns:
            float: 测试损失
        """
        # 使用工具函数评估分词模型
        return evaluate_tokenizer_on_test_data(tokenizer_path, self.test_data, self.config, self.device)
    
    def _evaluate_model_on_test_data(self, model, tokenizer, test_data):
        """在测试数据上评估模型"""
        # 使用工具函数评估模型
        return evaluate_model_on_test_data(model, tokenizer, test_data, self.config, self.device)

    def predict(self):
        """
        使用训练好的模型进行预测
        1. 预测未来10个工作日的股票走势（详细预测，保存到日期时间戳文件）
        2. 预测下一个工作日的涨跌幅（简化预测，增量更新到主Excel文件）
        """
        if not self.is_master:
            return True
            
        logger.info("=" * 60)
        logger.info("开始预测股票走势...")
        logger.info("=" * 60)
        
        try:
            # 加载最佳模型（训练好的模型，优先从本地加载）
            logger.info("加载最佳模型...")
            tokenizer = load_tokenizer(
                self.config.finetuned_tokenizer_path, 
                source='local', 
                local_files_only=True,
                fallback_on_error=False  # 训练好的模型必须存在，不需要回退
            )
            tokenizer.eval().to(self.device)
            
            model = load_predictor(
                self.config.finetuned_predictor_path, 
                source='local', 
                local_files_only=True,
                fallback_on_error=False  # 训练好的模型必须存在，不需要回退
            )
            model.eval().to(self.device)
            logger.info("✓ 模型加载成功")
            
            # 加载最新的测试数据
            test_data_path = os.path.join(self.config.dataset_path, self.data_source, "test_data.pkl")
            logger.info(f"加载最新数据: {test_data_path}")
            with open(test_data_path, 'rb') as f:
                test_data = pickle.load(f)
            logger.info(f"✓ 加载了 {len(test_data)} 支股票的数据")
            
            # ========== 1. 详细预测：预测未来10个工作日 ==========
            logger.info("\n" + "=" * 60)
            logger.info("执行详细预测：预测未来10个工作日...")
            logger.info("=" * 60)
            save_dir = self.config.save_path
            prediction_dfs = predict_future_trends(
                tokenizer, model, test_data, self.config, 
                self.device, save_dir
            )
            
            if prediction_dfs is None:
                logger.error("详细预测失败")
                return False
            
            logger.info("✓ 详细预测完成")
            
            # ========== 2. 简化预测：只预测下一个工作日并增量更新 ==========
            logger.info("\n" + "=" * 60)
            logger.info("执行简化预测：预测下一个工作日涨跌幅...")
            logger.info("=" * 60)
            
            # 从详细预测结果中提取第一天的预测
            if 'detailed' in prediction_dfs and not prediction_dfs['detailed'].empty:
                detailed_df = prediction_dfs['detailed']
                
                # 只保留第一天的预测数据
                next_day_columns = ['stock_code', 
                                   'current_open', 'current_high', 'current_low', 'current_close', 'current_volume',
                                   'day_1_date', 'day_1_open', 'day_1_high', 'day_1_low', 'day_1_close', 'day_1_volume',
                                   'day_1_open_change_pct', 'day_1_high_change_pct', 'day_1_low_change_pct', 
                                   'day_1_close_change_pct', 'day_1_volume_change_pct']
                
                next_day_prediction = detailed_df[next_day_columns].copy()
                
                # 初始化增量更新器
                master_excel_path = os.path.join(save_dir, "predictions_master.xlsx")
                updater = PredictionIncrementalUpdater(master_excel_path)
                
                # 获取预测日期（使用上海时间）
                shanghai_time = get_shanghai_time()
                prediction_date = shanghai_time.strftime('%Y-%m-%d')
                
                # 追加到主Excel文件
                logger.info(f"将预测结果追加到主Excel文件: {master_excel_path}")
                success = updater.append_daily_predictions(next_day_prediction, prediction_date)
                
                if success:
                    logger.info("✓ 增量更新成功")
                    logger.info(f"✓ 主Excel文件路径: {master_excel_path}")
                    logger.info(f"✓ 预测日期: {prediction_date}")
                    logger.info(f"✓ 预测股票数量: {len(next_day_prediction)}")
                    
                    # 导出CSV备份
                    csv_path = os.path.join(save_dir, "predictions_master.csv")
                    updater.export_to_csv(csv_path)
                    logger.info(f"✓ CSV备份已保存: {csv_path}")
                else:
                    logger.warning("增量更新失败，但详细预测已保存")
            else:
                logger.warning("未找到详细预测数据，跳过增量更新")
            
            logger.info("\n" + "=" * 60)
            logger.info("预测流程全部完成")
            logger.info("=" * 60)
            return True
            
        except Exception as e:
            logger.error(f"预测时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def run_pipeline(self):
        """运行完整训练流水线"""
        try:
            # 设置分布式环境（如果使用CUDA多GPU）
            if self.use_ddp:
                self.setup_distributed()
            
            # 处理数据并加载测试数据
            if not self.process_data():
                logger.error("数据处理失败，流水线终止")
                return False
            
            # 训练分词模型（每轮评估并保存最佳模型）
            if not self.train_tokenizer():
                logger.error("分词模型训练失败，流水线终止")
                return False
            
            # 训练预测模型（每轮评估并保存最佳模型）
            if not self.train_predictor():
                logger.error("预测模型训练失败，流水线终止")
                return False
            
            # 验证最佳模型是否已正确选择
            if not self.evaluate_models():
                logger.error("模型验证失败，流水线终止")
                return False
            
            # 使用最佳模型进行预测
            if not self.predict():
                logger.error("预测失败，流水线终止")
                return False
            
            if self.is_master:
                logger.info("完整训练流水线执行成功")
                logger.info(f"最佳分词模型测试损失: {self.best_tokenizer_test_loss:.4f}")
                logger.info(f"最佳预测模型测试损失: {self.best_predictor_test_loss:.4f}")
                logger.info(f"最佳分词模型路径: {self.config.finetuned_tokenizer_path}")
                logger.info(f"最佳预测模型路径: {self.config.finetuned_predictor_path}")
                
                # 更新历史最佳模型路径
                # 使用正确的model_history_subdir路径
                model_version = getattr(self.config, 'model_version', 'default')
                model_history_subdir = os.path.join(self.config.model_history_dir, f"{self.data_source}/{model_version}")
                success, tokenizer_path, predictor_path = update_best_model_paths(self.config, model_history_subdir)
                if success:
                    logger.info("已更新历史最佳模型路径")
                    if tokenizer_path:
                        logger.info(f"历史最佳分词模型路径: {tokenizer_path}")
                    if predictor_path:
                        logger.info(f"历史最佳预测模型路径: {predictor_path}")
                else:
                    logger.warning("更新历史最佳模型路径失败")
            
            # 清理分布式环境
            if self.use_ddp:
                cleanup_ddp()
                
            return True
        except Exception as e:
            logger.error(f"执行训练流水线时出错: {str(e)}")
            # 已在文件顶部导入
            logger.error(traceback.format_exc())
            
            # 清理分布式环境
            if self.use_ddp:
                cleanup_ddp()
                
            return False


# parse_args 函数已移至 optimized_config.py 中


def main():
    """主函数"""
    args = parse_args()
    
    # 优先设置日志，避免重复
    top_k = getattr(args, 'top_k_stocks', None)
    model_ver = getattr(args, 'model_version', None)
    data_source = getattr(args, 'data_source', None)
    setup_logging(top_k_stocks=top_k, data_source=data_source, model_version=model_ver)
    
    # 使用优化配置类创建配置
    try:
        config = create_config_from_args(args)
        logger.info(f"配置初始化完成: {config}")
    except Exception as e:
        logger.error(f"配置初始化失败: {str(e)}")
        return 1

    # 创建并运行流水线（不再重复调用setup_logging）
    pipeline = KronosTrainingPipeline(
        config=config,
        use_gpu=not args.cpu,
        data_source=args.data_source,
        early_stopping_patience=args.early_stopping_patience
    )
    success = pipeline.run_pipeline()
    
    return 0 if success else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(traceback.format_exc())
