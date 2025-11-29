#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
为predictions_master.xlsx添加涨幅计算列
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

def add_price_change_columns(excel_path: str):
    """
    为Excel文件添加涨幅计算列
    
    添加的列：
    1. 预测最高价相对开盘价涨幅(%) = (预测最高价 - 预测开盘价) / 预测开盘价 * 100
    2. 真实最高价相对开盘价涨幅(%) = (真实最高价 - 真实开盘价) / 真实开盘价 * 100 (如果有真实数据)
    3. 涨幅误差(%) = 预测涨幅 - 真实涨幅
    """
    
    print(f"正在读取Excel文件: {excel_path}")
    
    # 读取Excel文件
    try:
        df = pd.read_excel(excel_path, sheet_name='预测历史')
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        # 尝试读取第一个sheet
        df = pd.read_excel(excel_path, sheet_name=0)
    
    print(f"原始数据行数: {len(df)}")
    print(f"原始列数: {len(df.columns)}")
    
    # 检查必要的列是否存在
    required_cols = ['预测开盘价', '预测最高价']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"错误：缺少必要的列: {missing_cols}")
        return False
    
    # 1. 计算预测的最高价和预测的开盘价的涨幅
    print("\n计算预测的最高价相对开盘价涨幅...")
    df['预测最高价相对开盘价涨幅(%)'] = np.where(
        df['预测开盘价'] > 0,
        (df['预测最高价'] - df['预测开盘价']) / df['预测开盘价'] * 100,
        np.nan
    )
    
    # 2. 计算真实的最高价和真实的开盘价的涨幅
    print("计算真实的最高价相对开盘价涨幅...")
    
    # 检查是否有真实数据列
    has_real_open = '真实开盘价' in df.columns
    has_real_high = '真实最高价' in df.columns
    
    if has_real_open and has_real_high:
        # 如果有真实开盘价和真实最高价，直接计算
        df['真实最高价相对开盘价涨幅(%)'] = np.where(
            (df['真实开盘价'].notna()) & (df['真实开盘价'] > 0),
            (df['真实最高价'] - df['真实开盘价']) / df['真实开盘价'] * 100,
            np.nan
        )
    else:
        print("警告：未找到真实开盘价或真实最高价列")
        df['真实最高价相对开盘价涨幅(%)'] = np.nan
    
    # 3. 计算两个涨幅之间的误差
    print("计算涨幅误差...")
    df['涨幅误差(%)'] = np.where(
        (df['预测最高价相对开盘价涨幅(%)'].notna()) & 
        (df['真实最高价相对开盘价涨幅(%)'].notna()),
        df['预测最高价相对开盘价涨幅(%)'] - df['真实最高价相对开盘价涨幅(%)'],
        np.nan
    )
    
    # 移除不需要的辅助列（如果存在）
    columns_to_remove = ['最新日期收盘价', '真实开盘价缺失']
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"已移除列: {col}")
    
    # 显示统计信息
    print("\n=== 统计信息 ===")
    total_rows = len(df)
    has_prediction = df['预测最高价相对开盘价涨幅(%)'].notna().sum()
    has_real = df['真实最高价相对开盘价涨幅(%)'].notna().sum()
    has_error = df['涨幅误差(%)'].notna().sum()
    
    print(f"总行数: {total_rows}")
    print(f"有预测涨幅的行数: {has_prediction}")
    print(f"有真实涨幅的行数: {has_real}")
    print(f"有涨幅误差的行数: {has_error}")
    
    if has_error > 0:
        print(f"\n涨幅误差统计:")
        print(f"  平均误差: {df['涨幅误差(%)'].mean():.4f}%")
        print(f"  中位数误差: {df['涨幅误差(%)'].median():.4f}%")
        print(f"  最大正误差: {df['涨幅误差(%)'].max():.4f}%")
        print(f"  最大负误差: {df['涨幅误差(%)'].min():.4f}%")
        print(f"  标准差: {df['涨幅误差(%)'].std():.4f}%")
    
    # 保存更新后的Excel文件
    print(f"\n正在保存更新后的Excel文件...")
    try:
        # 使用openpyxl引擎以保持格式
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='预测历史', index=False)
            
            # 如果原文件有其他sheet，尝试保留
            try:
                original_excel = pd.ExcelFile(excel_path)
                for sheet_name in original_excel.sheet_names:
                    if sheet_name != '预测历史':
                        sheet_df = pd.read_excel(excel_path, sheet_name=sheet_name)
                        sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
            except:
                pass
        
        print(f"✓ 成功保存到: {excel_path}")
        print(f"✓ 新增列数: 3")
        print(f"  - 预测最高价相对开盘价涨幅(%)")
        print(f"  - 真实最高价相对开盘价涨幅(%)")
        print(f"  - 涨幅误差(%)")
        
        return True
        
    except Exception as e:
        print(f"保存Excel文件失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # 默认处理base_k100目录下的文件
    default_path = '/root/zouning/Kronos/finetune/outputs/sina/base_k100/predictions_master.xlsx'
    
    if len(sys.argv) > 1:
        excel_path = sys.argv[1]
    else:
        excel_path = default_path
    
    if not os.path.exists(excel_path):
        print(f"错误：文件不存在: {excel_path}")
        sys.exit(1)
    
    success = add_price_change_columns(excel_path)
    
    if success:
        print("\n✓ 处理完成！")
        sys.exit(0)
    else:
        print("\n✗ 处理失败！")
        sys.exit(1)

