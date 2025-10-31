#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
美股交易日历
判断美国股市的交易日和假期
"""

import logging
from datetime import datetime, timedelta, date
from typing import List, Set

logger = logging.getLogger('KronosPipeline')


def get_us_market_holidays(year: int) -> Set[date]:
    """
    获取指定年份的美股市场假期
    
    美股主要假期：
    - 新年 (New Year's Day): 1月1日
    - 马丁·路德·金纪念日 (Martin Luther King Jr. Day): 1月第三个星期一
    - 总统日 (Presidents' Day): 2月第三个星期一
    - 耶稣受难日 (Good Friday): 复活节前的星期五
    - 阵亡将士纪念日 (Memorial Day): 5月最后一个星期一
    - 独立日 (Independence Day): 7月4日
    - 劳动节 (Labor Day): 9月第一个星期一
    - 感恩节 (Thanksgiving Day): 11月第四个星期四
    - 圣诞节 (Christmas Day): 12月25日
    
    Args:
        year: 年份
        
    Returns:
        Set[date]: 假期日期集合
    """
    holidays = set()
    
    # 1. 新年 (1月1日)
    new_year = date(year, 1, 1)
    holidays.add(adjust_for_weekend(new_year))
    
    # 2. 马丁·路德·金纪念日 (1月第三个星期一)
    mlk_day = get_nth_weekday(year, 1, 0, 3)  # 0=周一
    holidays.add(mlk_day)
    
    # 3. 总统日 (2月第三个星期一)
    presidents_day = get_nth_weekday(year, 2, 0, 3)
    holidays.add(presidents_day)
    
    # 4. 耶稣受难日 (复活节前的星期五)
    good_friday = get_good_friday(year)
    holidays.add(good_friday)
    
    # 5. 阵亡将士纪念日 (5月最后一个星期一)
    memorial_day = get_last_weekday(year, 5, 0)
    holidays.add(memorial_day)
    
    # 6. 独立日 (7月4日)
    independence_day = date(year, 7, 4)
    holidays.add(adjust_for_weekend(independence_day))
    
    # 7. 劳动节 (9月第一个星期一)
    labor_day = get_nth_weekday(year, 9, 0, 1)
    holidays.add(labor_day)
    
    # 8. 感恩节 (11月第四个星期四)
    thanksgiving = get_nth_weekday(year, 11, 3, 4)  # 3=周四
    holidays.add(thanksgiving)
    
    # 9. 圣诞节 (12月25日)
    christmas = date(year, 12, 25)
    holidays.add(adjust_for_weekend(christmas))
    
    logger.debug(f"{year}年美股假期: {sorted(holidays)}")
    return holidays


def get_nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """
    获取指定月份的第N个指定星期几
    
    Args:
        year: 年份
        month: 月份 (1-12)
        weekday: 星期几 (0=周一, 6=周日)
        n: 第几个 (1, 2, 3, 4)
        
    Returns:
        date: 日期
    """
    first_day = date(year, month, 1)
    first_weekday = first_day.weekday()
    
    # 计算第一个指定星期几的日期
    days_ahead = (weekday - first_weekday) % 7
    first_target = first_day + timedelta(days=days_ahead)
    
    # 加上 (n-1) 周
    target_date = first_target + timedelta(weeks=(n - 1))
    
    return target_date


def get_last_weekday(year: int, month: int, weekday: int) -> date:
    """
    获取指定月份的最后一个指定星期几
    
    Args:
        year: 年份
        month: 月份 (1-12)
        weekday: 星期几 (0=周一, 6=周日)
        
    Returns:
        date: 日期
    """
    # 获取下个月的第一天
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    
    # 回退一天到本月最后一天
    last_day = next_month - timedelta(days=1)
    last_weekday = last_day.weekday()
    
    # 计算最后一个指定星期几
    days_back = (last_weekday - weekday) % 7
    target_date = last_day - timedelta(days=days_back)
    
    return target_date


def get_easter(year: int) -> date:
    """
    计算复活节日期（使用Meeus/Jones/Butcher算法）
    
    Args:
        year: 年份
        
    Returns:
        date: 复活节日期
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    
    return date(year, month, day)


def get_good_friday(year: int) -> date:
    """
    计算耶稣受难日（复活节前的星期五）
    
    Args:
        year: 年份
        
    Returns:
        date: 耶稣受难日日期
    """
    easter = get_easter(year)
    good_friday = easter - timedelta(days=2)
    return good_friday


def adjust_for_weekend(holiday: date) -> date:
    """
    如果假期是周末，调整到最近的工作日
    周六的假期调整到周五，周日的假期调整到周一
    
    Args:
        holiday: 假期日期
        
    Returns:
        date: 调整后的日期
    """
    weekday = holiday.weekday()
    
    if weekday == 5:  # 周六
        return holiday - timedelta(days=1)
    elif weekday == 6:  # 周日
        return holiday + timedelta(days=1)
    else:
        return holiday


def is_us_trading_day(check_date: date) -> bool:
    """
    判断指定日期是否为美股交易日
    
    Args:
        check_date: 要检查的日期
        
    Returns:
        bool: True表示是交易日，False表示不是
    """
    # 检查是否是周末
    if check_date.weekday() >= 5:  # 5=周六, 6=周日
        return False
    
    # 检查是否是假期
    year = check_date.year
    holidays = get_us_market_holidays(year)
    
    if check_date in holidays:
        return False
    
    return True


def get_next_trading_day(start_date: date, include_start: bool = False) -> date:
    """
    获取下一个交易日
    
    Args:
        start_date: 起始日期
        include_start: 是否包含起始日期
        
    Returns:
        date: 下一个交易日
    """
    if include_start and is_us_trading_day(start_date):
        return start_date
    
    current = start_date + timedelta(days=1)
    while not is_us_trading_day(current):
        current += timedelta(days=1)
    
    return current


def get_previous_trading_day(start_date: date, include_start: bool = False) -> date:
    """
    获取上一个交易日
    
    Args:
        start_date: 起始日期
        include_start: 是否包含起始日期
        
    Returns:
        date: 上一个交易日
    """
    if include_start and is_us_trading_day(start_date):
        return start_date
    
    current = start_date - timedelta(days=1)
    while not is_us_trading_day(current):
        current -= timedelta(days=1)
    
    return current


def get_future_trading_days(start_date: date, num_days: int) -> List[date]:
    """
    获取未来N个交易日
    
    Args:
        start_date: 起始日期（不包含）
        num_days: 需要的交易日数量
        
    Returns:
        List[date]: 交易日列表
    """
    trading_days = []
    current = start_date
    
    while len(trading_days) < num_days:
        current = get_next_trading_day(current, include_start=False)
        trading_days.append(current)
    
    return trading_days


def count_trading_days_between(start_date: date, end_date: date) -> int:
    """
    计算两个日期之间的交易日数量（不包含起始和结束日期）
    
    Args:
        start_date: 起始日期
        end_date: 结束日期
        
    Returns:
        int: 交易日数量
    """
    count = 0
    current = start_date + timedelta(days=1)
    
    while current < end_date:
        if is_us_trading_day(current):
            count += 1
        current += timedelta(days=1)
    
    return count


def is_market_open_time() -> bool:
    """
    判断当前时间是否在美股交易时间内
    美股交易时间: 美东时间 9:30 AM - 4:00 PM
    转换为北京时间: 晚上9:30 PM - 次日凌晨4:00 AM (夏令时)
                   晚上10:30 PM - 次日凌晨5:00 AM (冬令时)
    
    Returns:
        bool: True表示市场开盘，False表示市场关闭
    """
    import pytz
    
    # 获取当前美东时间
    et_tz = pytz.timezone('America/New_York')
    now_et = datetime.now(et_tz)
    
    # 检查是否是交易日
    if not is_us_trading_day(now_et.date()):
        return False
    
    # 检查是否在交易时间内 (9:30 AM - 4:00 PM ET)
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now_et <= market_close


def get_market_status() -> dict:
    """
    获取市场状态信息
    
    Returns:
        dict: 包含市场状态的字典
    """
    import pytz
    
    # 获取当前时间
    et_tz = pytz.timezone('America/New_York')
    now_et = datetime.now(et_tz)
    today = now_et.date()
    
    # 判断今天是否是交易日
    is_trading = is_us_trading_day(today)
    
    # 判断市场是否开盘
    is_open = is_market_open_time()
    
    # 获取下一个交易日
    next_trading = get_next_trading_day(today, include_start=False)
    
    return {
        'current_time_et': now_et.strftime('%Y-%m-%d %H:%M:%S %Z'),
        'is_trading_day': is_trading,
        'is_market_open': is_open,
        'next_trading_day': next_trading.strftime('%Y-%m-%d'),
        'weekday': now_et.strftime('%A')
    }


if __name__ == '__main__':
    # 测试交易日历
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    logger.info("=== 测试美股交易日历 ===")
    
    # 测试2025年的假期
    year = 2025
    holidays = get_us_market_holidays(year)
    logger.info(f"\n{year}年美股假期:")
    for h in sorted(holidays):
        logger.info(f"  {h.strftime('%Y-%m-%d %A')}")
    
    # 测试今天是否是交易日
    today = date.today()
    is_trading = is_us_trading_day(today)
    logger.info(f"\n今天 ({today}) 是否是交易日: {is_trading}")
    
    # 获取下一个交易日
    next_trading = get_next_trading_day(today, include_start=False)
    logger.info(f"下一个交易日: {next_trading}")
    
    # 获取未来10个交易日
    future_days = get_future_trading_days(today, 10)
    logger.info(f"\n未来10个交易日:")
    for i, d in enumerate(future_days, 1):
        logger.info(f"  {i}. {d.strftime('%Y-%m-%d %A')}")
    
    # 获取市场状态
    status = get_market_status()
    logger.info(f"\n市场状态:")
    for key, value in status.items():
        logger.info(f"  {key}: {value}")

