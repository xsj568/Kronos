#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kronos 金融模型完整训练流程
包括：配置，数据源，数据处理，分词模型训练，预测模型训练，预测最新的结果
"""

import os
import sys
import json
import time
import logging
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
from time import gmtime, strftime
from pathlib import Path

# 确保项目根目录在路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from model.kronos import KronosTokenizer, Kronos
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
    predict_latest_data,
    save_pipeline_config
)
from common_data_processor import DataProcessorFactory

# 全局日志记录器
logger = logging.getLogger('KronosPipeline')


class KronosTrainingPipeline:
    """
    Kronos模型训练完整流水线
    """
    
    def __init__(self, config, use_gpu=True, data_source='qlib'):
        """
        初始化训练流水线
        
        Args:
            config: 配置对象
            use_gpu: 是否使用GPU训练
            data_source: 数据源类型，'qlib'或'sina'
        """
        self.config = config
        self.use_gpu = use_gpu
        self.data_source = data_source
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
        self.is_master = True  # 单进程或主进程
        
        # 初始化日志
        setup_logging()
        logger.info(f"初始化Kronos训练流水线 - GPU: {use_gpu}, 数据源: {data_source}")
        
        # 设置保存路径
        self.tokenizer_save_dir = os.path.join(config.save_path, config.tokenizer_save_folder_name)
        self.predictor_save_dir = os.path.join(config.save_path, config.predictor_save_folder_name)
        os.makedirs(os.path.join(self.tokenizer_save_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.predictor_save_dir, 'checkpoints'), exist_ok=True)
        
        # 保存流水线配置
        save_pipeline_config(config, config.save_path)
        
        # 设置随机种子
        set_seed(config.seed)
        
    def setup_distributed(self):
        """设置分布式训练环境"""
        if not self.use_gpu:
            logger.info("使用CPU训练，跳过分布式设置")
            return
            
        try:
            self.rank, self.world_size, self.local_rank = setup_ddp()
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.is_master = (self.rank == 0)
            set_seed(self.config.seed, self.rank)
            logger.info(f"分布式训练环境设置完成 - Rank: {self.rank}, World Size: {self.world_size}")
        except Exception as e:
            logger.error(f"设置分布式训练环境时出错: {str(e)}")
            raise
    
    def process_data(self):
        """处理数据"""
        if self.is_master:
            logger.info(f"开始处理{self.data_source}数据...")
            try:
                # 使用工厂创建数据处理器
                processor = DataProcessorFactory.create_processor(self.data_source, self.config)
                result = processor.run_pipeline()
                logger.info(f"数据处理完成: {result}")
                return True
            except Exception as e:
                logger.error(f"处理数据时出错: {str(e)}")
                return False
        return True  # 非主进程直接返回成功
    
    def train_tokenizer(self):
        """训练分词模型"""
        start_time = time.time()
        logger.info("开始训练分词模型...")
        
        # 初始化模型
        try:
            model = KronosTokenizer.from_pretrained(self.config.pretrained_tokenizer_path)
            model.to(self.device)
            logger.info(f"分词模型初始化完成 - 大小: {get_model_size(model)}")
        except Exception as e:
            logger.error(f"初始化分词模型时出错: {str(e)}")
            return False
        
        # 设置DDP
        if self.use_gpu and torch.cuda.device_count() > 1:
            model = DDP(model, device_ids=[self.local_rank], find_unused_parameters=False)
        
        # 创建数据加载器
        if self.use_gpu and torch.cuda.device_count() > 1:
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
                    if self.use_gpu and torch.cuda.device_count() > 1:
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
                    model.parameters() if not self.use_gpu or torch.cuda.device_count() <= 1 else model.module.parameters(), 
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
                    if self.use_gpu and torch.cuda.device_count() > 1:
                        zs, _, _, _ = model(ori_batch_x)
                    else:
                        zs, _, _, _ = model(ori_batch_x)
                    _, z = zs
                    val_loss_item = F.mse_loss(z, ori_batch_x)
                    
                    tot_val_loss += val_loss_item.item() * ori_batch_x.size(0)
                    val_sample_count += ori_batch_x.size(0)
            
            # 如果是分布式训练，收集所有进程的验证损失
            if self.use_gpu and torch.cuda.device_count() > 1:
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
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_path = f"{self.tokenizer_save_dir}/checkpoints/best_model"
                    if self.use_gpu and torch.cuda.device_count() > 1:
                        model.module.save_pretrained(save_path)
                    else:
                        model.save_pretrained(save_path)
                    logger.info(f"最佳模型已保存到 {save_path} (验证损失: {best_val_loss:.4f})")
                    if comet_logger:
                        comet_logger.log_model("best_model", save_path)
            
            # 同步所有进程
            if self.use_gpu and torch.cuda.device_count() > 1:
                dist.barrier()
        
        # 保存训练摘要
        if self.is_master:
            summary = {
                'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
                'end_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
                'total_time': format_time(time.time() - start_time),
                'best_val_loss': best_val_loss,
                'epochs': self.config.epochs,
                'world_size': self.world_size,
                'device': str(self.device),
            }
            save_training_summary(self.tokenizer_save_dir, summary)
            logger.info("分词模型训练完成")
            
            if comet_logger:
                comet_logger.end()
        
        return True
    
    def train_predictor(self):
        """训练预测模型"""
        start_time = time.time()
        logger.info("开始训练预测模型...")
        
        # 初始化分词模型和预测模型
        try:
            tokenizer = KronosTokenizer.from_pretrained(self.config.finetuned_tokenizer_path)
            tokenizer.eval().to(self.device)
            logger.info("分词模型加载完成")
            
            model = Kronos.from_pretrained(self.config.pretrained_predictor_path)
            model.to(self.device)
            logger.info(f"预测模型初始化完成 - 大小: {get_model_size(model)}")
        except Exception as e:
            logger.error(f"初始化模型时出错: {str(e)}")
            return False
        
        # 设置DDP
        if self.use_gpu and torch.cuda.device_count() > 1:
            model = DDP(model, device_ids=[self.local_rank], find_unused_parameters=False)
        
        # 创建数据加载器
        if self.use_gpu and torch.cuda.device_count() > 1:
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
                if self.use_gpu and torch.cuda.device_count() > 1:
                    logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                    loss, s1_loss, s2_loss = model.module.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
                else:
                    logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                    loss, s1_loss, s2_loss = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters() if not self.use_gpu or torch.cuda.device_count() <= 1 else model.module.parameters(), 
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
                    
                    if self.use_gpu and torch.cuda.device_count() > 1:
                        logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                        val_loss, _, _ = model.module.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
                    else:
                        logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                        val_loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
                    
                    tot_val_loss += val_loss.item()
                    val_batches_processed += 1
            
            # 如果是分布式训练，收集所有进程的验证损失
            if self.use_gpu and torch.cuda.device_count() > 1:
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
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_path = f"{self.predictor_save_dir}/checkpoints/best_model"
                    if self.use_gpu and torch.cuda.device_count() > 1:
                        model.module.save_pretrained(save_path)
                    else:
                        model.save_pretrained(save_path)
                    logger.info(f"最佳模型已保存到 {save_path} (验证损失: {best_val_loss:.4f})")
            
            # 同步所有进程
            if self.use_gpu and torch.cuda.device_count() > 1:
                dist.barrier()
        
        # 保存训练摘要
        if self.is_master:
            summary = {
                'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
                'end_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
                'total_time': format_time(time.time() - start_time),
                'best_val_loss': best_val_loss,
                'epochs': self.config.epochs,
                'world_size': self.world_size,
                'device': str(self.device),
            }
            save_training_summary(self.predictor_save_dir, summary)
            logger.info("预测模型训练完成")
            
            if comet_logger:
                comet_logger.end()
        
        return True
    
    def predict(self):
        """使用训练好的模型进行预测"""
        if not self.is_master:
            return True
            
        logger.info("开始进行预测...")
        try:
            # 加载模型
            tokenizer = KronosTokenizer.from_pretrained(self.config.finetuned_tokenizer_path)
            tokenizer.eval().to(self.device)
            
            model = Kronos.from_pretrained(self.config.finetuned_predictor_path)
            model.eval().to(self.device)
            
            # 这里实现预测逻辑
            # 根据实际需求实现
            
            logger.info("预测完成")
            return True
        except Exception as e:
            logger.error(f"预测时出错: {str(e)}")
            return False
    
    def run_pipeline(self):
        """运行完整训练流水线"""
        try:
            # 设置分布式环境（如果使用GPU）
            if self.use_gpu and torch.cuda.device_count() > 1:
                self.setup_distributed()
            
            # 处理数据
            if not self.process_data():
                logger.error("数据处理失败，流水线终止")
                return False
            
            # 训练分词模型
            if not self.train_tokenizer():
                logger.error("分词模型训练失败，流水线终止")
                return False
            
            # 训练预测模型
            if not self.train_predictor():
                logger.error("预测模型训练失败，流水线终止")
                return False
            
            # 进行预测
            if not self.predict():
                logger.error("预测失败，流水线终止")
                return False
            
            if self.is_master:
                logger.info("完整训练流水线执行成功")
            
            # 清理分布式环境
            if self.use_gpu and torch.cuda.device_count() > 1:
                cleanup_ddp()
                
            return True
        except Exception as e:
            logger.error(f"执行训练流水线时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 清理分布式环境
            if self.use_gpu and torch.cuda.device_count() > 1:
                cleanup_ddp()
                
            return False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Kronos模型训练流水线')
    parser.add_argument('--cpu', action='store_true', default=False, help='使用CPU训练')
    parser.add_argument('--data-source', type=str, default='qlib', choices=['qlib', 'sina'], help='数据源类型')
    parser.add_argument('--config-path', type=str, default=None, help='配置文件路径')
    parser.add_argument('--force-download', action='store_true', default=False, help='强制重新下载数据')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置配置
    config = Config()
    if args.config_path:
        # 从文件加载配置
        pass

    config.qlib_data_path = './qlib_bin'
    config.batch_size = 4
    config.n_train_iter = 8
    config.n_val_iter = 4
    config.use_comet = False
    config.save_path = f"./outputs/{args.data_source}/models"
    config.pretrained_tokenizer_path = 'NeoQuasar/Kronos-Tokenizer-2k'
    config.pretrained_predictor_path = 'NeoQuasar/Kronos-mini'
    config.finetuned_tokenizer_path = f"{config.save_path}/{config.tokenizer_save_folder_name}/checkpoints/best_model"
    config.finetuned_predictor_path = f"{config.save_path}/{config.predictor_save_folder_name}/checkpoints/best_model"
    config.force_download_data = args.force_download

    # 创建并运行流水线
    pipeline = KronosTrainingPipeline(
        config=config,
        use_gpu=not args.cpu,
        data_source=args.data_source
    )
    success = pipeline.run_pipeline()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
