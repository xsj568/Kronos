import os
import sys
import pickle
import json
import time
import logging
import requests
import random
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import trange
from abc import ABC, abstractmethod
from torch.utils.data import Dataset

# 添加项目路径
sys.path.append("../")
from optimized_config import OptimizedConfig as Config

# 全局logger
logger = logging.getLogger('KronosPipeline')


class BaseDataProcessor(ABC):
    """
    数据处理的抽象基类，定义了数据处理的通用接口
    """
    
    def __init__(self, config):
        """初始化数据处理器"""
        self.config = config
        self.dataset_path = Path(config.dataset_path)
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        self.data = {}  # 存储处理后的数据
    
    @abstractmethod
    def download_data(self):
        """下载数据"""
        pass
    
    @abstractmethod
    def process_raw_data(self):
        """处理原始数据"""
        pass
    
    def save_processed_data(self, data, data_type, data_source=None):
        """保存处理后的数据
        
        Args:
            data: 要保存的数据
            data_type: 数据类型（train/val/test）
            data_source: 数据来源（qlib/sina等）
        """
        # 获取数据来源
        if data_source is None:
            if hasattr(self, 'data_source'):
                data_source = self.data_source
            else:
                data_source = self.__class__.__name__.replace('DataProcessor', '').lower()
        
        # 创建数据来源特定的目录
        source_dir = self.dataset_path / data_source
        source_dir.mkdir(exist_ok=True, parents=True)
        
        # 构建文件路径，包含数据来源信息
        file_path = source_dir / f"{data_type}_data.pkl"
        
        # 记录数据统计信息
        symbol_count = len(data)
        total_rows = sum(len(df) for df in data.values())
        logger.info(f"保存{data_source}/{data_type}数据集: {symbol_count}支股票, 共{total_rows}行数据")
        
        # 使用最高级别的pickle协议保存
        start_time = time.time()
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 计算和记录统计信息
        elapsed = time.time() - start_time
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        logger.info(f"数据保存完成，耗时 {elapsed:.2f} 秒, 文件大小: {file_size:.2f} MB")
        
        return file_path
    
    def run_pipeline(self):
        """运行完整的数据处理流程"""
        # 获取数据来源名称
        data_source = self.__class__.__name__.replace('DataProcessor', '').lower()
        self.data_source = data_source
        
        logger.info(f"运行{data_source}数据处理流程")
        
        # 下载和处理数据
        start_time = time.time()
        self.download_data()
        processed_data = self.process_raw_data()
        
        # 保存处理后的数据
        train_path = self.save_processed_data(processed_data['train'], 'train', data_source)
        val_path = self.save_processed_data(processed_data['val'], 'val', data_source)
        test_path = self.save_processed_data(processed_data['test'], 'test', data_source)
        
        # 计算总耗时
        total_time = time.time() - start_time
        logger.info(f"数据处理完成，总耗时: {total_time:.2f} 秒")
        logger.info(f"数据存储路径:")
        logger.info(f"  - 训练数据: {train_path}")
        logger.info(f"  - 验证数据: {val_path}")
        logger.info(f"  - 测试数据: {test_path}")
        
        return {'train': train_path, 'val': val_path, 'test': test_path}


class SinaDataProcessor(BaseDataProcessor):
    """
    从新浪财经获取股票数据并处理成与Qlib相同的格式
    """
    
    def __init__(self, config):
        """初始化新浪数据处理器"""
        super().__init__(config)
        self.url_base = "http://stock.finance.sina.com.cn/usstock/api/json_v2.php/US_MinKService.getDailyK?symbol=%s&___qn=3n"
        
        # 加载股票代码列表（支持缓存）
        self.symbols = self._load_or_select_stocks()
        logger.info(f"使用{len(self.symbols)}支股票进行训练: {self.symbols[:5]}... 等")
        
        self.data_fields = ['open', 'close', 'high', 'low', 'volume']
        self.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']
    
    def _load_or_select_stocks(self):
        """
        加载或选择股票列表（支持缓存）
        
        流程：
        1. 如果启用缓存且缓存文件存在，从缓存加载
        2. 否则，从CSV读取所有股票，选择TopK活跃股票
        3. 将选中的股票保存到缓存文件
        
        缓存文件名根据top_k自动生成（如selected_stocks_3000.json），
        避免不同股票数量的测试和正式训练互相覆盖。
        
        Returns:
            list: 股票代码列表
        """
        # 获取配置
        top_k = getattr(self.config, 'top_k_stocks', 3000)
        use_cache = getattr(self.config, 'use_stock_cache', True)
        cache_file = getattr(self.config, 'stock_cache_file', None)
        selection_days = getattr(self.config, 'stock_selection_days', 365)
        
        # 如果没有指定缓存文件名，根据top_k自动生成
        if cache_file is None:
            cache_file = f'selected_stocks_{top_k}.json'
        
        # 构建缓存文件完整路径
        cache_path = Path(__file__).parent / cache_file
        
        # 1. 尝试从缓存加载
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                cached_symbols = cached_data.get('symbols', [])
                cached_top_k = cached_data.get('top_k', 0)
                cached_days = cached_data.get('selection_days', 0)
                cached_date_str = cached_data.get('selection_date', 'unknown')
                
                # 验证缓存是否有效
                if cached_symbols and len(cached_symbols) > 0:
                    logger.info(f"从缓存文件加载股票列表: {cache_path}")
                    logger.info(f"  缓存参数: top_k={cached_top_k}, selection_days={cached_days}")
                    logger.info(f"  缓存日期: {cached_date_str}")
                    logger.info(f"  加载 {len(cached_symbols)} 支股票")
                    
                    # 如果配置的top_k比缓存的少，只返回前top_k个
                    if top_k < len(cached_symbols):
                        logger.info(f"  当前配置top_k={top_k}，从缓存中取前{top_k}支股票")
                        return cached_symbols[:top_k]
                    
                    return cached_symbols
            except Exception as e:
                logger.warning(f"加载缓存文件失败: {e}，将重新选择股票")
        
        # 2. 从CSV读取所有股票代码
        logger.info(f"{'重新' if use_cache else ''}选择TopK={top_k}活跃股票（基于最近{selection_days}天数据）")
        csv_path = Path(__file__).parent / "data" / "stock_code_US.csv"
        
        if not csv_path.exists():
            logger.warning(f"CSV文件不存在: {csv_path}，使用默认股票列表")
            default_symbols = self._get_default_symbols()
            return default_symbols[:min(len(default_symbols), top_k)]
        
        try:
            # 读取CSV文件 - 只使用symbol列
            df = pd.read_csv(csv_path)
            
            if 'symbol' not in df.columns:
                logger.error("CSV文件中缺少'symbol'列")
                return self._get_default_symbols()[:top_k]
            
            # 过滤有效股票代码
            df = df[df['symbol'].notna()]
            df = df[df['symbol'].apply(lambda x: isinstance(x, str) and len(x) > 0 and not x.startswith('-'))]
            all_symbols = df['symbol'].tolist()
            
            logger.info(f"从CSV读取到 {len(all_symbols)} 支股票代码")
            
            # 如果CSV中的股票数量少于top_k，直接返回所有股票
            if len(all_symbols) <= top_k:
                logger.info(f"CSV中股票数量({len(all_symbols)})未超过top_k({top_k})，使用全部股票")
                selected_symbols = all_symbols
            else:
                # 3. 通过实时API选择TopK活跃股票
                logger.info(f"开始从 {len(all_symbols)} 支股票中选择 {top_k} 支最活跃的股票")
                selected_symbols = self._select_top_k_active_stocks(all_symbols, top_k, selection_days)
            
            # 4. 保存到缓存文件
            cache_data = {
                'symbols': selected_symbols,
                'top_k': top_k,
                'selection_days': selection_days,
                'selection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_candidates': len(all_symbols)
            }
            
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2, ensure_ascii=False)
                logger.info(f"股票列表已保存到缓存文件: {cache_path}")
            except Exception as e:
                logger.warning(f"保存缓存文件失败: {e}")
            
            return selected_symbols
            
        except Exception as e:
            logger.error(f"选择股票失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return self._get_default_symbols()[:top_k]
    
    def _select_top_k_active_stocks(self, symbols, top_k, selection_days):
        """
        从给定股票列表中选择TopK最活跃的股票
        
        基于最近N天的平均成交量和数据完整性进行评分
        
        Args:
            symbols: 候选股票列表
            top_k: 需要选择的股票数量
            selection_days: 用于评估的天数
            
        Returns:
            list: 选中的TopK股票列表
        """
        logger.info(f"基于最近{selection_days}天的数据评估股票活跃度...")
        
        stock_scores = {}
        checked_count = 0
        failed_count = 0
        debug_sample_shown = False
        
        # 随机打乱顺序，避免总是检查前面的股票
        random.shuffle(symbols)
        
        # 限制检查的股票数量（最多检查top_k的3倍，以提高效率）
        symbols_to_check = symbols[:min(len(symbols), top_k * 3)]
        total_to_check = len(symbols_to_check)
        
        # 计算打印进度的间隔，确保总共打印约10次
        log_interval = max(1, total_to_check // 10)
        
        for symbol in symbols_to_check:
            try:
                url = self.url_base % symbol
                response = requests.get(url, timeout=10)  # 增加超时时间到10秒
                
                if response.status_code == 200 and response.text:
                    # 尝试解析JSON
                    try:
                        data = json.loads(response.text)
                    except json.JSONDecodeError as je:
                        # 如果是第一次，打印调试信息
                        if not debug_sample_shown:
                            logger.warning(f"JSON解析失败，股票 {symbol}，响应内容: {response.text[:200]}")
                            debug_sample_shown = True
                        failed_count += 1
                        continue
                    
                    # 打印第一个成功的数据样本用于调试
                    if not debug_sample_shown and data:
                        logger.info(f"API响应示例（股票{symbol}）: 数据长度={len(data)}, 示例={data[:2] if len(data) >= 2 else data}")
                        debug_sample_shown = True
                    
                    # 检查是否有足够的数据
                    if data and isinstance(data, list) and len(data) >= min(30, selection_days):  # 至少30天数据
                        # 取最近N天的数据
                        recent_data = data[-selection_days:] if len(data) >= selection_days else data
                        
                        # 计算平均成交量和数据完整性
                        volumes = []
                        for d in recent_data:
                            if isinstance(d, dict) and 'volume' in d:
                                try:
                                    vol = float(d['volume'])
                                    if vol > 0:  # 过滤无效数据
                                        volumes.append(vol)
                                except:
                                    continue
                            elif isinstance(d, dict) and 'v' in d:  # 尝试另一个可能的字段名
                                try:
                                    vol = float(d['v'])
                                    if vol > 0:
                                        volumes.append(vol)
                                except:
                                    continue
                        
                        if len(volumes) >= min(20, selection_days // 2):  # 至少一半天数有数据
                            # 计算评分：平均成交量 × 数据完整性权重
                            avg_volume = sum(volumes) / len(volumes)
                            completeness = len(volumes) / len(recent_data)  # 数据完整性
                            score = avg_volume * (1 + completeness)  # 成交量越大，数据越完整，分数越高
                            
                            stock_scores[symbol] = {
                                'score': score,
                                'avg_volume': avg_volume,
                                'completeness': completeness,
                                'data_points': len(volumes)
                            }
                    else:
                        failed_count += 1
                else:
                    failed_count += 1
                
                checked_count += 1
                
                # 定期输出进度 - 改为打印约10次
                if checked_count % log_interval == 0 or checked_count == total_to_check:
                    logger.info(f"  已评估 {checked_count}/{total_to_check} 支股票，找到 {len(stock_scores)} 支有效股票，失败 {failed_count} 支")
                    
            except Exception as e:
                # 跳过无法获取数据的股票
                if not debug_sample_shown:
                    logger.warning(f"获取股票 {symbol} 数据失败: {str(e)}")
                    debug_sample_shown = True
                failed_count += 1
                continue
        
        logger.info(f"评估完成：从 {len(symbols_to_check)} 支中找到 {len(stock_scores)} 支有效股票")
        
        # 如果有效股票不足top_k，用未检查的股票随机补充
        if len(stock_scores) < top_k:
            logger.warning(f"有效股票数量({len(stock_scores)})少于top_k({top_k})，随机补充")
            remaining_symbols = [s for s in symbols if s not in stock_scores]
            random.shuffle(remaining_symbols)
            additional_count = top_k - len(stock_scores)
            additional = remaining_symbols[:additional_count]
            
            # 给补充的股票一个较低的默认分数
            for sym in additional:
                stock_scores[sym] = {'score': 0, 'avg_volume': 0, 'completeness': 0, 'data_points': 0}
            
            logger.info(f"  随机补充了 {len(additional)} 支股票")
        
        # 按评分排序，选择TopK
        sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        selected_symbols = [s[0] for s in sorted_stocks[:top_k]]
        
        # 输出TopK统计信息
        if len(sorted_stocks) > 0:
            top_scores = [s[1] for s in sorted_stocks[:min(10, len(sorted_stocks))]]
            top_volumes = [f"{s['avg_volume']:.0f}" for s in top_scores]
            logger.info(f"Top{min(10, len(sorted_stocks))}股票平均成交量: {top_volumes}")
        
        logger.info(f"最终选择 {len(selected_symbols)} 支股票")
        return selected_symbols
    
    def _get_default_symbols(self):
        """
        获取默认的股票代码列表
        
        Returns:
            list: 默认股票代码列表
        """
        default_symbols = [
            # 科技股
            'AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'INTC', 'AMD', 
            # 金融股
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'BLK',
            # 医疗保健
            'JNJ', 'PFE', 'MRK', 'ABBV', 'UNH', 'CVS', 'ABT', 'LLY', 'AMGN', 'BMY',
            # 消费品
            'PG', 'KO', 'PEP', 'WMT', 'MCD', 'SBUX', 'NKE', 'DIS', 'HD', 'LOW',
            # 能源
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'BP',
            # 工业
            'GE', 'HON', 'MMM', 'CAT', 'DE', 'BA', 'LMT', 'RTX', 'UPS', 'FDX',
            # 电信
            'T', 'VZ', 'TMUS', 'CMCSA', 'NFLX', 'CHTR', 'DISH', 'LUMN', 'ATVI', 'EA',
            # 半导体
            'TSM', 'AVGO', 'QCOM', 'TXN', 'MU', 'AMAT', 'KLAC', 'LRCX', 'ADI', 'MCHP',
            # 中国股票
            'BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'LI', 'XPEV', 'TME', 'BILI', 'NTES'
        ]
        
        # 如果配置中指定了sina_symbols，则使用配置中的列表
        if hasattr(self.config, 'sina_symbols') and self.config.sina_symbols:
            return self.config.sina_symbols
        
        return default_symbols
    
    def http_get(self, url, params=None, headers=None, retry=3, timeout=10):
        """HTTP GET请求，支持重试"""
        req_count = 0
        while req_count < retry:
            try:
                resp = requests.get(url=url, params=params, headers=headers, timeout=timeout)
                if resp.status_code == 200 or resp.status_code == 206:
                    return resp
            except Exception as e:
                logger.warning(f"HTTP请求失败: {e}")
            req_count += 1
            time.sleep(0.5)
        return None
    
    def download_data(self):
        """从新浪财经下载股票数据"""
        total_symbols = len(self.symbols)
        logger.info(f"开始从新浪财经下载数据，共 {total_symbols} 支股票")
        
        # 计算日志打印间隔，确保总共打印约10次
        log_interval = max(1, total_symbols // 10)
        success_count = 0
        failed_count = 0
        
        # 详细的失败统计
        fail_reasons = {
            'http_failed': 0,      # HTTP请求失败
            'empty_data': 0,       # API返回空数据
            'insufficient_data': 0, # 数据不足（<10条）
            'exception': 0         # 其他异常
        }
        
        for symbol_idx in range(total_symbols):
            symbol_code = self.symbols[symbol_idx]
            try:
                url = self.url_base % symbol_code
                response = self.http_get(url=url, timeout=10)
                if response is None:
                    failed_count += 1
                    fail_reasons['http_failed'] += 1
                    continue
                    
                data_json = response.json()
                if not data_json:
                    failed_count += 1
                    fail_reasons['empty_data'] += 1
                    continue
                
                # 转换为DataFrame
                df = self._json_to_dataframe(data_json, symbol_code)
                if df is None or len(df) < 10:  # 至少需要10条数据
                    failed_count += 1
                    fail_reasons['insufficient_data'] += 1
                    continue
                    
                self.data[symbol_code] = df
                success_count += 1
                
                # 定期输出进度 - 只打印约10次
                if (symbol_idx + 1) % log_interval == 0 or (symbol_idx + 1) == total_symbols:
                    logger.info(f"下载进度: {symbol_idx + 1}/{total_symbols} - 成功: {success_count}, 失败: {failed_count} "
                              f"(请求失败:{fail_reasons['http_failed']}, 空数据:{fail_reasons['empty_data']}, "
                              f"数据不足:{fail_reasons['insufficient_data']}, 异常:{fail_reasons['exception']})")
                    
            except Exception as e:
                failed_count += 1
                fail_reasons['exception'] += 1
                # 只在前3次错误时打印详细信息
                if fail_reasons['exception'] <= 3:
                    logger.error(f"处理股票 {symbol_code} 时出错: {str(e)}")
        
        logger.info(f"数据下载完成 - 总计: {total_symbols}支, 成功: {success_count}支, 失败: {failed_count}支")
        logger.info(f"失败原因统计: HTTP请求失败:{fail_reasons['http_failed']}, "
                   f"API返回空数据:{fail_reasons['empty_data']}, "
                   f"数据不足(<10条):{fail_reasons['insufficient_data']}, "
                   f"异常错误:{fail_reasons['exception']}")
    
    def _json_to_dataframe(self, data_json, symbol_code):
        """将JSON数据转换为DataFrame"""
        if not data_json:
            return None
        
        try:
            # 提取数据并过滤无效日期
            valid_data = []
            for item in data_json:
                # 检查日期是否有效
                date_str = item.get('d', '')
                if not date_str or date_str.startswith('0000-00-00'):
                    continue  # 跳过无效日期
                
                try:
                    # 尝试提取所有字段
                    valid_data.append({
                        'd': date_str,
                        'o': float(item['o']),
                        'c': float(item['c']),
                        'h': float(item['h']),
                        'l': float(item['l']),
                        'v': int(item['v'])
                    })
                except (KeyError, ValueError, TypeError):
                    # 跳过数据不完整或无效的记录
                    continue
            
            if not valid_data:
                return None
            
            # 提取有效数据
            dates = [item['d'] for item in valid_data]
            opens = [item['o'] for item in valid_data]
            closes = [item['c'] for item in valid_data]
            highs = [item['h'] for item in valid_data]
            lows = [item['l'] for item in valid_data]
            volumes = [item['v'] for item in valid_data]
            
            # 创建DataFrame，不显示warning
            dates_pd = pd.to_datetime(dates, errors='coerce')
            # 移除无法解析的日期
            valid_mask = dates_pd.notna()
            if not valid_mask.any():
                return None
            
            dates_pd = dates_pd[valid_mask]
            opens = [opens[i] for i, v in enumerate(valid_mask) if v]
            closes = [closes[i] for i, v in enumerate(valid_mask) if v]
            highs = [highs[i] for i, v in enumerate(valid_mask) if v]
            lows = [lows[i] for i, v in enumerate(valid_mask) if v]
            volumes = [volumes[i] for i, v in enumerate(valid_mask) if v]
            
            df = pd.DataFrame({
                'datetime': dates_pd,
                'open': opens,
                'close': closes,
                'high': highs,
                'low': lows,
                'volume': volumes
            }, index=dates_pd)
        except Exception as e:
            # 捕获任何异常并返回None
            return None
        
        # 转换日期为整数格式
        df['date'] = df['datetime'].dt.strftime('%Y%m%d').astype(int)
        
        # 计算额外特征
        df['vol'] = df['volume']  # 与Qlib保持一致
        df['amt'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4 * df['vol']  # 成交金额估计
        
        # 过滤数据，只保留指定时间范围内的数据
        start_date = pd.Timestamp(self.config.dataset_begin_time)
        end_date = pd.Timestamp(self.config.dataset_end_time)
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        return df
    
    def process_raw_data(self):
        """处理原始数据，分割为训练集、验证集和测试集"""
        logger.info("分割数据为训练集、验证集和测试集...")
        train_data, val_data, test_data = {}, {}, {}
        
        symbols_list = list(self.data.keys())
        total_symbols = len(symbols_list)
        log_interval = max(1, total_symbols // 10)
        
        for idx, symbol in enumerate(symbols_list):
            df = self.data[symbol]
            # 确保数据按时间排序
            df = df.sort_index()
            
            # 定义时间范围
            train_start, train_end = self.config.train_time_range
            val_start, val_end = self.config.val_time_range
            test_start, test_end = self.config.test_time_range
            
            # 创建布尔掩码
            train_mask = (df.index >= pd.Timestamp(train_start)) & (df.index <= pd.Timestamp(train_end))
            val_mask = (df.index >= pd.Timestamp(val_start)) & (df.index <= pd.Timestamp(val_end))
            test_mask = (df.index >= pd.Timestamp(test_start)) & (df.index <= pd.Timestamp(test_end))
            
            # 应用掩码创建最终数据集
            train_data[symbol] = df[train_mask]
            val_data[symbol] = df[val_mask]
            test_data[symbol] = df[test_mask]
            
            # 定期输出进度 - 只打印约10次
            if (idx + 1) % log_interval == 0 or (idx + 1) == total_symbols:
                logger.info(f"数据分割进度: {idx + 1}/{total_symbols} - 最新: {symbol} (训练:{len(train_data[symbol])}, 验证:{len(val_data[symbol])}, 测试:{len(test_data[symbol])})")
        
        # 输出汇总统计
        total_train = sum(len(df) for df in train_data.values())
        total_val = sum(len(df) for df in val_data.values())
        total_test = sum(len(df) for df in test_data.values())
        logger.info(f"数据分割完成 - 训练集:{total_train}条, 验证集:{total_val}条, 测试集:{total_test}条")
        
        return {'train': train_data, 'val': val_data, 'test': test_data}


class QlibDataProcessor(BaseDataProcessor):
    """
    处理Qlib格式的金融数据
    """
    
    def __init__(self, config):
        """初始化Qlib数据处理器"""
        super().__init__(config)
        self.data_fields = ['open', 'close', 'high', 'low', 'volume', 'vwap']
        self.force_download = getattr(self.config, 'force_download_data', False)
    
    def download_data(self):
        """初始化Qlib环境并加载数据"""
        try:
            import qlib
            from qlib.config import REG_CN
            from qlib.data import D
            from qlib.data.dataset.loader import QlibDataLoader
            import os
            import subprocess
            import shutil
            
            # 确保目录存在
            data_dir = os.path.expanduser(self.config.qlib_data_path)
            os.makedirs(os.path.dirname(data_dir), exist_ok=True)
            
            # 检查是否需要下载数据
            need_download = self.force_download
            
            # 检查数据目录是否存在并且有数据文件
            if not need_download and not os.path.exists(data_dir):
                logger.info(f"数据目录 {data_dir} 不存在，需要下载数据")
                need_download = True
            elif not need_download:
                # 检查目录是否为空或缺少关键文件
                if not os.path.exists(data_dir) or not os.listdir(data_dir):
                    logger.info(f"数据目录 {data_dir} 为空，需要下载数据")
                    need_download = True
                elif not (os.path.exists(os.path.join(data_dir, 'calendars')) and 
                         os.path.exists(os.path.join(data_dir, 'instruments'))):
                    logger.info(f"数据目录 {data_dir} 缺少关键文件，需要下载数据")
                    need_download = True
                else:
                    logger.info(f"数据目录 {data_dir} 已存在并包含数据文件，跳过下载步骤")
            
            if need_download:
                logger.info("从GitHub下载最新的Qlib数据...")
                
                # 创建临时目录用于下载和解压
                temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_qlib_data")
                os.makedirs(temp_dir, exist_ok=True)
                
                # 下载最新的qlib数据
                logger.info(f"下载qlib数据到临时目录: {temp_dir}")
                data_url = "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"
                tar_path = os.path.join(temp_dir, "qlib_bin.tar.gz")
                
                try:
                    logger.info(f"从 {data_url} 下载数据...")
                    response = requests.get(data_url, stream=True)
                    response.raise_for_status()
                    
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    last_percent = -1
                    with open(tar_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    percent = int(100 * downloaded / total_size)
                                    if percent % 10 == 0 and percent != last_percent:
                                        logger.info(f"下载进度: {percent}%")
                                        last_percent = percent
                    
                    logger.info("数据下载成功")
                except Exception as e:
                    logger.error(f"下载数据失败: {str(e)}")
                    raise
                
                # 解压数据到qlib目录
                logger.info(f"解压数据到: {data_dir}")
                
                try:
                    import tarfile
                    with tarfile.open(tar_path, "r:gz") as tar:
                        # 解压所有文件到目标目录
                        members = tar.getmembers()
                        total_members = len(members)
                        logger.info(f"开始解压 {total_members} 个文件...")
                        
                        # 只在关键百分比时显示进度
                        progress_points = [0, 25, 50, 75, 100]
                        next_point_idx = 0
                        
                        for i, member in enumerate(members):
                            # 处理路径以去除第一级目录
                            if member.name.find('/') != -1:
                                member.name = '/'.join(member.name.split('/')[1:])
                            if member.name:
                                tar.extract(member, path=data_dir)
                            
                            # 显示进度
                            percent_done = int((i+1) * 100 / total_members)
                            if next_point_idx < len(progress_points) and percent_done >= progress_points[next_point_idx]:
                                logger.info(f"解压进度: {percent_done}% ({i+1}/{total_members})")
                                next_point_idx += 1
                    
                    logger.info("数据解压成功")
                except Exception as e:
                    logger.error(f"解压数据失败: {str(e)}")
                    raise
                
                # 清理临时文件
                logger.info("清理临时文件")
                shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                logger.info("使用现有的qlib数据")
            
            # 更新配置中的qlib数据路径
            self.config.qlib_data_path = data_dir
            
            # 初始化Qlib环境
            logger.info("初始化Qlib环境...")
            try:
                qlib.init(provider_uri=self.config.qlib_data_path, region=REG_CN)
                logger.info("Qlib环境初始化成功")
            except Exception as e:
                logger.error(f"Qlib环境初始化失败: {str(e)}")
                # 尝试使用默认路径
                try:
                    qlib.init(region=REG_CN)
                    logger.info("使用默认路径初始化Qlib环境成功")
                except Exception as e2:
                    logger.error(f"使用默认路径初始化Qlib环境也失败: {str(e2)}")
                    raise
            
            logger.info("从Qlib加载数据...")
            data_fields_qlib = ['$' + f for f in self.data_fields]
            cal = D.calendar()

            # 确定实际的开始和结束时间
            start_index = cal.searchsorted(pd.Timestamp(self.config.dataset_begin_time))
            end_index = cal.searchsorted(pd.Timestamp(self.config.dataset_end_time))

            # 处理边界条件
            adjusted_start_index = max(start_index - self.config.lookback_window, 0)
            real_start_time = cal[adjusted_start_index]

            if end_index >= len(cal):
                end_index = len(cal) - 1
            elif cal[end_index] != pd.Timestamp(self.config.dataset_end_time):
                end_index -= 1

            adjusted_end_index = min(end_index + self.config.predict_window, len(cal) - 1)
            real_end_time = cal[adjusted_end_index]

            # 加载数据
            logger.info(f"加载数据时间范围: {real_start_time} 至 {real_end_time}")
            data_df = QlibDataLoader(config=data_fields_qlib).load(
                self.config.instrument, real_start_time, real_end_time
            )
            data_df = data_df.stack().unstack(level=1)
            
            symbol_list = list(data_df.columns)
            logger.info(f"处理 {len(symbol_list)} 个股票代码...")
            
            for i in range(len(symbol_list)):
                symbol = symbol_list[i]
                symbol_df = data_df[symbol]

                # 透视表
                symbol_df = symbol_df.reset_index().rename(columns={'level_1': 'field'})
                symbol_df = pd.pivot(symbol_df, index='datetime', columns='field', values=symbol)
                symbol_df = symbol_df.rename(columns={f'${field}': field for field in self.data_fields})

                # 计算额外特征
                symbol_df['vol'] = symbol_df['volume']
                symbol_df['amt'] = (symbol_df['open'] + symbol_df['high'] + symbol_df['low'] + symbol_df['close']) / 4 * symbol_df['vol']
                symbol_df = symbol_df[self.config.feature_list]

                # 过滤数据
                symbol_df = symbol_df.dropna()
                if len(symbol_df) < self.config.lookback_window + self.config.predict_window + 1:
                    continue

                self.data[symbol] = symbol_df
                
                # 定期输出进度 - 只打印约10次
                log_interval = max(1, len(symbol_list) // 10)
                if (i + 1) % log_interval == 0 or (i + 1) == len(symbol_list):
                    logger.info(f"处理进度: {i + 1}/{len(symbol_list)} - 当前有效股票: {len(self.data)} 支")
            
            logger.info(f"数据加载完成，共 {len(self.data)} 个有效股票代码")
            return True
        except Exception as e:
            logger.error(f"加载Qlib数据失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def process_raw_data(self):
        """处理原始数据，分割为训练集、验证集和测试集"""
        logger.info("分割数据为训练集、验证集和测试集...")
        train_data, val_data, test_data = {}, {}, {}

        symbol_list = list(self.data.keys())
        total_symbols = len(symbol_list)
        log_interval = max(1, total_symbols // 10)
        
        for i in range(total_symbols):
            symbol = symbol_list[i]
            symbol_df = self.data[symbol]

            # 定义时间范围
            train_start, train_end = self.config.train_time_range
            val_start, val_end = self.config.val_time_range
            test_start, test_end = self.config.test_time_range

            # 创建布尔掩码
            train_mask = (symbol_df.index >= pd.Timestamp(train_start)) & (symbol_df.index <= pd.Timestamp(train_end))
            val_mask = (symbol_df.index >= pd.Timestamp(val_start)) & (symbol_df.index <= pd.Timestamp(val_end))
            test_mask = (symbol_df.index >= pd.Timestamp(test_start)) & (symbol_df.index <= pd.Timestamp(test_end))

            # 应用掩码创建最终数据集
            train_data[symbol] = symbol_df[train_mask]
            val_data[symbol] = symbol_df[val_mask]
            test_data[symbol] = symbol_df[test_mask]
            
            # 定期输出进度 - 只打印约10次
            if (i + 1) % log_interval == 0 or (i + 1) == total_symbols:
                logger.info(f"数据分割进度: {i + 1}/{total_symbols} - 最新: {symbol} (训练:{len(train_data[symbol])}, 验证:{len(val_data[symbol])}, 测试:{len(test_data[symbol])})")
        
        # 输出汇总统计
        total_train = sum(len(df) for df in train_data.values())
        total_val = sum(len(df) for df in val_data.values())
        total_test = sum(len(df) for df in test_data.values())
        logger.info(f"数据分割完成 - 训练集:{total_train}条, 验证集:{total_val}条, 测试集:{total_test}条")

        return {'train': train_data, 'val': val_data, 'test': test_data}


class FinancialDataset(Dataset):
    """
    A PyTorch Dataset for handling financial time series data from various sources.

    This dataset pre-computes all possible start indices for sliding windows
    and then randomly samples from them during training/validation.

    Args:
        data_type (str): The type of dataset to load, either 'train' or 'val'.

    Raises:
        ValueError: If `data_type` is not 'train' or 'val'.
    """

    def __init__(self, data_type: str = 'train', config=None):
        # 处理config参数，可以是Config对象、字典或None
        if config is None:
            self.config = Config()
        elif isinstance(config, dict):
            # 如果是字典，创建一个Config对象并更新属性
            self.config = Config()
            for key, value in config.items():
                setattr(self.config, key, value)
        else:
            self.config = config
            
        if data_type not in ['train', 'val']:
            raise ValueError("data_type must be 'train' or 'val'")
        self.data_type = data_type

        # Use a dedicated random number generator for sampling to avoid
        # interfering with other random processes (e.g., in model initialization).
        self.py_rng = random.Random(self.config.seed)

        # 获取数据来源
        self.data_source = getattr(self.config, 'data_source', 'qlib')
        
        # Set paths and number of samples based on the data type.
        if data_type == 'train':
            self.data_path = f"{self.config.dataset_path}/{self.data_source}/train_data.pkl"
            self.n_samples = self.config.n_train_iter
        else:
            self.data_path = f"{self.config.dataset_path}/{self.data_source}/val_data.pkl"
            self.n_samples = self.config.n_val_iter
            
        logger.info(f"加载{self.data_source}/{data_type}数据集: {self.data_path}")

        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.window = self.config.lookback_window + self.config.predict_window + 1

        self.symbols = list(self.data.keys())
        self.feature_list = self.config.feature_list
        self.time_feature_list = self.config.time_feature_list

        # Pre-compute all possible (symbol, start_index) pairs.
        self.indices = []
        logger.info(f"[{data_type.upper()}] Pre-computing sample indices...")
        for symbol in self.symbols:
            df = self.data[symbol].reset_index()
            series_len = len(df)
            num_samples = series_len - self.window + 1

            if num_samples > 0:
                # Generate time features and store them directly in the dataframe.
                df['minute'] = df['datetime'].dt.minute
                df['hour'] = df['datetime'].dt.hour
                df['weekday'] = df['datetime'].dt.weekday
                df['day'] = df['datetime'].dt.day
                df['month'] = df['datetime'].dt.month
                # Keep only necessary columns to save memory.
                self.data[symbol] = df[self.feature_list + self.time_feature_list]

                # Add all valid starting indices for this symbol to the global list.
                for i in range(num_samples):
                    self.indices.append((symbol, i))

        # The effective dataset size is the minimum of the configured iterations
        # and the total number of available samples.
        self.n_samples = min(self.n_samples, len(self.indices))
        logger.info(f"[{data_type.upper()}] Found {len(self.indices)} possible samples. Using {self.n_samples} per epoch.")

    def set_epoch_seed(self, epoch: int):
        """
        Sets a new seed for the random sampler for each epoch. This is crucial
        for reproducibility in distributed training.

        Args:
            epoch (int): The current epoch number.
        """
        epoch_seed = self.config.seed + epoch
        self.py_rng.seed(epoch_seed)

    def __len__(self) -> int:
        """Returns the number of samples per epoch."""
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a random sample from the dataset.

        Note: The `idx` argument is ignored. Instead, a random index is drawn
        from the pre-computed `self.indices` list using `self.py_rng`. This
        ensures random sampling over the entire dataset for each call.

        Args:
            idx (int): Ignored.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x_tensor (torch.Tensor): The normalized feature tensor.
                - x_stamp_tensor (torch.Tensor): The time feature tensor.
        """
        # Select a random sample from the entire pool of indices.
        random_idx = self.py_rng.randint(0, len(self.indices) - 1)
        symbol, start_idx = self.indices[random_idx]

        # Extract the sliding window from the dataframe.
        df = self.data[symbol]
        end_idx = start_idx + self.window
        win_df = df.iloc[start_idx:end_idx]

        # Separate main features and time features.
        x = win_df[self.feature_list].values.astype(np.float32)
        x_stamp = win_df[self.time_feature_list].values.astype(np.float32)

        # Perform instance-level normalization.
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        # Convert to PyTorch tensors.
        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)

        return x_tensor, x_stamp_tensor


class DataProcessorFactory:
    """
    数据处理器工厂，用于创建不同类型的数据处理器
    """
    
    @staticmethod
    def create_processor(data_source_type: str, config, **kwargs):
        """
        创建数据处理器
        
        Args:
            data_source_type: 数据源类型，'qlib'或'sina'
            config: 配置对象
            **kwargs: 额外参数
            
        Returns:
            数据处理器实例
        """
        if data_source_type.lower() == 'qlib':
            return QlibDataProcessor(config)
        elif data_source_type.lower() == 'sina':
            return SinaDataProcessor(config)
        else:
            raise ValueError(f"未知的数据源类型: {data_source_type}")


if __name__ == '__main__':
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger('KronosPipeline')
    
    # 测试数据处理器
    config = Config()
    config.sina_symbols = ['AAPL', 'MSFT', 'GOOG']  # 示例股票代码

    # 使用工厂创建处理器
    processor = DataProcessorFactory.create_processor('sina', config)
    processor.run_pipeline()
