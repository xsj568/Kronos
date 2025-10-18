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
from config import Config

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
        
        # 加载股票代码列表
        self.symbols = self._load_symbols_from_csv()
        logger.info(f"使用{len(self.symbols)}支股票进行训练: {self.symbols[:5]}... 等")
        
        self.data_fields = ['open', 'close', 'high', 'low', 'volume']
        self.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']
    
    def _load_symbols_from_csv(self):
        """
        从stock_code_US.csv文件中加载股票代码列表
        
        Returns:
            list: 股票代码列表
        """
        csv_path = Path(__file__).parent / "stock_code_US.csv"
        
        if not csv_path.exists():
            logger.warning(f"CSV文件不存在: {csv_path}，使用默认股票代码列表")
            return self._get_default_symbols()
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_path)
            
            # 检查必要的列是否存在
            if 'symbol' not in df.columns:
                logger.error("CSV文件中缺少'symbol'列")
                return self._get_default_symbols()
            
            # 获取配置中的过滤条件
            filters = getattr(self.config, 'sina_symbol_filters', {})
            
            # 应用过滤条件
            filtered_df = df.copy()

            # 限制股票数量
            max_symbols = getattr(self.config, 'max_sina_symbols', None)
            if max_symbols and len(filtered_df) > max_symbols:
                # 随机选择指定数量的股票
                filtered_df = filtered_df.sample(n=max_symbols, random_state=42)
                logger.info(f"随机选择 {max_symbols} 支股票")
            
            # 提取股票代码
            symbols = filtered_df['symbol'].dropna().tolist()
            
            # 过滤掉无效的股票代码
            symbols = [s for s in symbols if isinstance(s, str) and len(s) > 0 and not s.startswith('-')]
            
            if not symbols:
                logger.warning("过滤后没有有效的股票代码，使用默认列表")
                return self._get_default_symbols()
            
            logger.info(f"从CSV文件加载了 {len(symbols)} 支股票代码")
            return symbols
            
        except Exception as e:
            logger.error(f"读取CSV文件失败: {str(e)}，使用默认股票代码列表")
            return self._get_default_symbols()
    
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
        logger.info(f"开始从新浪财经下载数据，股票代码: {self.symbols}")
        for symbol in trange(len(self.symbols), desc="下载股票数据"):
            symbol_code = self.symbols[symbol]
            try:
                url = self.url_base % symbol_code
                response = self.http_get(url=url, timeout=10)
                if response is None:
                    logger.warning(f"无法获取股票数据: {symbol_code}")
                    continue
                    
                data_json = response.json()
                if not data_json:
                    logger.warning(f"股票 {symbol_code} 没有数据")
                    continue
                
                # 转换为DataFrame
                df = self._json_to_dataframe(data_json, symbol_code)
                if df is None or len(df) < 10:  # 至少需要10条数据
                    logger.warning(f"股票 {symbol_code} 数据不足")
                    continue
                    
                self.data[symbol_code] = df
                logger.info(f"成功下载股票 {symbol_code} 数据，共 {len(df)} 条记录")
            except Exception as e:
                logger.error(f"处理股票 {symbol_code} 时出错: {str(e)}")
        
        logger.info(f"数据下载完成，共 {len(self.data)} 个股票")
    
    def _json_to_dataframe(self, data_json, symbol_code):
        """将JSON数据转换为DataFrame"""
        if not data_json:
            return None
            
        # 提取数据
        dates = [item['d'] for item in data_json]
        opens = [float(item['o']) for item in data_json]
        closes = [float(item['c']) for item in data_json]
        highs = [float(item['h']) for item in data_json]
        lows = [float(item['l']) for item in data_json]
        volumes = [int(item['v']) for item in data_json]
        
        # 创建DataFrame
        dates_pd = pd.to_datetime(dates)
        df = pd.DataFrame({
            'datetime': dates_pd,  # Use 'datetime' instead of 'date'
            'open': opens,
            'close': closes,
            'high': highs,
            'low': lows,
            'volume': volumes
        }, index=dates_pd)
        
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
        
        for symbol, df in self.data.items():
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
            
            logger.info(f"股票 {symbol} - 训练: {len(train_data[symbol])}条, 验证: {len(val_data[symbol])}条, 测试: {len(test_data[symbol])}条")
        
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
            
            for i in trange(len(symbol_list), desc="处理股票数据"):
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
        for i in trange(len(symbol_list), desc="准备数据集"):
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
