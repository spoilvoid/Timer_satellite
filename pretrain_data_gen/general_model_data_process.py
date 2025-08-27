import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any
from datetime import datetime
import argparse

from scipy.interpolate import interp1d


def convert_time_num(time_str):
    # start_time = '2006-01-01T00:00:00.000Z'
    # start_num = 1136044800.0
    time_str = time_str[:-1]
    dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%f")
    time_num = float(dt.timestamp() - 1136044800.0)
    return time_num


#载入数据
def load_satellite_data(raw_data_root, satellite_name, time_day):
    path_in = f'{raw_data_root}/{satellite_name}/{time_day}/'
    final_df = pd.DataFrame()
    for file in os.listdir(path_in):
            # 数据加载
            part_data = pd.read_orc(path_in + file)
            part_data['id'] = part_data['satellite_code'] + '_' + part_data['satellite_time']
            part_data['parsed_value'] = pd.to_numeric(part_data['parsed_value'], errors='coerce')
            pivot_data = part_data.pivot_table(index='id',columns='param_code', values='parsed_value', fill_value=None)
            pivot_data = pivot_data.sort_values(by='id')  
            pivot_data.dropna(axis=0,how="all",inplace=True)  
            pivot_data['time']=pivot_data.index.str.split("_").str[1].map(convert_time_num)
            final_df = pd.concat([final_df,pivot_data],axis=0)
            del part_data
            del pivot_data
    final_df=final_df.sort_values("time")
    return final_df


def find_valid_timeindex(selected_data_df, patch_time=10, density_threshold=0.5):
    """基于数据密度检测有效数据段
    
    Args:
        selected_data_df: 输入数据框
        density_threshold: 数据密度阈值（百分比），默认0.5
        patch_time: 每个patch的长度，默认10s
    
    Returns:
        list: 包含元组的列表，每个元组为(段起始时间, 段结束时间)，时间点来自原始数据框的索引
    """
    # 计算每个时间点的数据密度
    feat_num = len(selected_data_df.columns)
    density_series = pd.Series(0, index=selected_data_df.index)
    
    # 计算每个时间点的有效数据点数量
    for col in selected_data_df.columns:
        valid_data = selected_data_df[col].notna()
        density_series[valid_data] += 1
    
    # 计算平均密度百分比（每个时间点的有效特征数/总特征数）
    density_percentage = density_series / feat_num
    
    # 将时间序列按patch_len长度分段
    time_index = density_percentage.index
    patch_densities = []
    patch_time_index = []
    
    for idx, time_point in enumerate(time_index):
        if idx == 0:
            density_summary = density_percentage[time_point]
            n_timepoint = 1
            start_time_point = time_point
        elif start_time_point + patch_time >= time_point:
            density_summary += density_percentage[time_point]
            n_timepoint += 1
        else:
            patch_densities.append(density_summary / n_timepoint)
            patch_time_index.append((start_time_point, time_index[idx - 1]))
            density_summary = density_percentage[time_point]
            n_timepoint = 1
            start_time_point = time_point        
    patch_densities.append(density_summary / n_timepoint)
    patch_time_index.append((start_time_point, time_index[len(time_index) - 1]))
    
    # 找出密度大于阈值的patch
    previous_valid = False
    valid_patches = []
    
    for i, (density, (start_time, end_time)) in enumerate(zip(patch_densities, patch_time_index)):
        if density < density_threshold and not previous_valid:
            continue
        elif density < density_threshold and previous_valid:
            patch_end_time = patch_time_index[i-1][1]
            valid_patches.append((patch_start_time, patch_end_time))
            previous_valid = False
        elif density >= density_threshold and not previous_valid:
            patch_start_time = start_time
            previous_valid = True
        elif density >= density_threshold and previous_valid:
            continue
    
    if previous_valid:
        valid_patches.append((patch_start_time, patch_time_index[len(patch_time_index)-1][1]))
    
    return valid_patches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./general_data", help="输入数据文件根目录")
    parser.add_argument("--delta_t", type=int, default=60, help="时间间隔偏移")
    parser.add_argument("--n_offset", type=int, default=6, help="时间间隔偏移数量")
    parser.add_argument("--patch_time", type=int, default=90, help="单时间段最小间隔")
    parser.add_argument("--density_threshold", type=float, default=0.25, help="数据密度阈值")
    parser.add_argument("--output_dir", type=str, default="./general_data_processed", help="输出数据文件根目录")
    args = parser.parse_args()
    
    # 从配置中获取参数
    data_tuple_path = os.path.join(args.input_dir, "params/data_tuple.json") 
    params_path = os.path.join(args.input_dir, "params/X_range.json") 
    delta_t = args.delta_t
    n_offset = args.n_offset
    patch_time = args.patch_time
    density_threshold = args.density_threshold
    os.makedirs(args.output_dir, exist_ok=True)

    if not all([data_tuple_path, params_path]):
        error_msg = "缺少必要参数，需提供数据ID对路径和相关参数路径"
        raise ValueError(error_msg)
    
    print("载入待处理数据列表")
    # 载入待处理数据列表
    try:
        with open(data_tuple_path, "r", encoding="utf-8") as f:
            data_tuple_list = json.load(f)
        print(f"待处理数据列表: {data_tuple_list[:5]}...(共{len(data_tuple_list)}个)")
    except FileNotFoundError:
        raise FileNotFoundError(f"待处理数据列表文件 {data_tuple_path} 不存在")

    print(f"载入参数列表")
    try:
        # 相关参数载入
        with open(params_path, 'r', encoding='utf-8') as f:
            params_data = json.load(f)
        params_list = list(params_data['state'].keys()) + list(params_data['continuous'].keys()) + list(params_data['multilabel'].keys()) + list(params_data['values'].keys())
    except FileNotFoundError as e:
        raise ValueError(f"参数列表文件不存在: {str(e)}")
    
    for satellite_name, time_day in data_tuple_list:
        print(f"开始处理卫星{satellite_name}在{time_day}的数据")
        try:
            # 加载卫星数据
            final_df = load_satellite_data(args.input_dir, satellite_name, time_day)
            print(f"成功加载卫星数据，共{len(final_df)}条记录")
        except Exception as e:
            raise ValueError(f"加载卫星数据失败: {str(e)}")
        
        final_df = final_df[["time"] + params_list]
        final_df.set_index('time', inplace=True)
        general_time_index_range_list = find_valid_timeindex(final_df, patch_time=patch_time, density_threshold=density_threshold)
        time_index_range_list, final_df_subset_list = [], []
        for start_time, end_time in general_time_index_range_list:
            final_df_subset = final_df.loc[final_df.index.to_series().between(start_time, end_time)]
            if(len(final_df_subset) > 0):
                time_index_range_list.append((start_time, end_time))
                final_df_subset_list.append(final_df_subset)
            
        for patch_idx, final_df_subset in enumerate(final_df_subset_list):
            final_df_subset_file_path = os.path.join(args.output_dir, f"raw_data/{satellite_name}_{time_day}_patch{patch_idx}-final_df.csv")
            os.makedirs(os.path.dirname(final_df_subset_file_path), exist_ok=True)
            final_df_subset.to_csv(final_df_subset_file_path, index=True, index_label="time", encoding='utf-8-sig')
            print(f"原始数据已保存到: {final_df_subset_file_path}")

        print(f"开始对所有的卫星{satellite_name}在{time_day}的数据进行时间偏移处理")
        time_offsets = np.arange(0, delta_t, int(delta_t / n_offset))
        for patch_idx, (final_df_subset, (min_time, max_time)) in enumerate(zip(final_df_subset_list, time_index_range_list)):
            offset_df_file_path = os.path.join(args.output_dir, f"raw_data/{satellite_name}_{time_day}_patch{patch_idx}-final_df.csv")
            offset_df =pd.read_csv(offset_df_file_path, index_col=0)
            start_time = offset_df.index[0]
            end_time = offset_df.index[-1]
            new_index = np.arange(start_time, end_time, delta_t)
            # 创建多个时间偏移版本的数据集
            for offset in time_offsets:
                # 创建带偏移的新时间索引
                offset_index = new_index + offset
                # 创建新的DataFrame用于存储插值结果
                offset_df = pd.DataFrame(index=offset_index, columns=offset_df.columns)
                # 对每个特征列应用不同的插值方法
                for i, column in enumerate(offset_df.columns):
                    old_times = offset_df[column].index.values
                    values = offset_df[column].values
                    if column in params_data['values'].keys():
                        interp_func = interp1d(old_times, values, kind='linear', bounds_error=False, fill_value='extrapolate')
                    elif column in params_data['state'].keys():
                        values = np.where(np.abs(values) > 0.5, 1, 0)
                        interp_func = interp1d(old_times, values, kind='nearest', bounds_error=False, fill_value='extrapolate')
                    elif column in params_data['continuous'].keys():
                        values = np.where(np.abs(values) > params_data['continuous'][column], 1, 0)
                        interp_func = interp1d(old_times, values, kind='nearest', bounds_error=False, fill_value='extrapolate')
                    elif column in params_data['multilabel'].keys():
                        interp_func = interp1d(old_times, values, kind='nearest', bounds_error=False, fill_value='extrapolate')
                    # 应用插值函数到偏移后的时间索引
                    offset_df[column] = interp_func(offset_index)
                offset_df = offset_df[offset_df.index.to_series(). between(min_time, max_time)]
                offset_df_filepath = os.path.join(args.output_dir, f"processed_data/{satellite_name}_{time_day}_interval{delta_t}_patch{patch_idx}_offset{offset}.csv")
                os.makedirs(os.path.dirname(offset_df_filepath), exist_ok=True)
                offset_df.to_csv(offset_df_filepath, index=True, index_label="time", encoding='utf-8-sig')

    print(f"卫星数据处理完成，保存到{args.output_dir}")