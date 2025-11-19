#!/usr/bin/env python3
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import seaborn as sns
from scipy import stats

# 硬件规格配置
HARDWARE_SPECS = {
    "l20": {"name": "NVIDIA_L20", "mem_bandwidth": 864 * (1024**3), "FP16": 119.5e12, "memsize": 48 * (1024**3)},
    "h20": {"name": "NVIDIA_H20_SXM5_96GB", "mem_bandwidth": 4022 * (1024**3), "FP16": 148e12, "memsize": 96 * (1024**3)},
    "H20": {"name": "NVIDIA_H20_SXM5_96GB", "mem_bandwidth": 4022 * (1024**3), "FP16": 148e12, "memsize": 96 * (1024**3)},
    "a800": {"name": "NVIDIA_A800_SXM4_80GB", "mem_bandwidth": 2039 * (1024**3), "FP16": 312e12, "memsize": 80 * (1024**3)},
}

# Base model 映射和参数量信息
BASE_MODEL_MAPPING = {
    "qwen2-7b": {"name": "Qwen2-7B", "params": 7.0},
    "qwen3-4b": {"name": "Qwen3-4B", "params": 4.0},
    "qwen3-32b": {"name": "Qwen3-32B", "params": 32.0},
}

# 指定要分析的4个代表性案例
SELECTED_CASES = {
    "qwen2-7b_l20": {"model": "qwen2-7b", "gpu_card": "l20", "quadrant": "high_qps_high_util"},
    "qwen3-4b_h20": {"model": "qwen3-4b", "gpu_card": "h20", "quadrant": "high_qps_low_util"},
    "qwen3-32b_H20": {"model": "qwen3-32b", "gpu_card": "H20", "quadrant": "low_qps_high_util"},
    "qwen2-7b_a800": {"model": "qwen2-7b", "gpu_card": "a800", "quadrant": "low_qps_low_util"}
}

def parse_diff_file_selective(filename: str) -> List[Dict]:
    """解析diff.txt文件，只提取指定的4个代表性案例"""
    services = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    service_blocks = content.strip().split('\n\n')
    
    for block in service_blocks:
        if not block.strip():
            continue
            
        lines = block.strip().split('\n')
        if len(lines) < 4:
            continue
            
        service_name = lines[0]
        
        # 解析基本信息
        model = None
        gpu_card = None
        gpu_num = None
        sm_util = None
        gpu_util = None
        
        for line in lines[1:]:
            if line.startswith('model '):
                model = line.split(' ', 1)[1]
            elif line.startswith('gpu_card '):
                gpu_card = line.split(' ', 1)[1]
            elif line.startswith('gpu_num '):
                gpu_num = int(line.split(' ')[1])
            elif line.startswith('sm ') and 'gpu ' in line:
                parts = line.split()
                sm_util = float(parts[1])
                gpu_util = float(parts[3])
            elif line == 'qps':
                qps_start_idx = lines.index(line) + 1
                break
        
        # 检查是否为选定的案例之一
        case_key = f"{model}_{gpu_card}"
        if case_key not in SELECTED_CASES:
            continue
        
        # 解析QPS时间序列数据
        qps_values = []
        for i in range(qps_start_idx, len(lines), 2):
            if i + 1 < len(lines):
                try:
                    qps_value = float(lines[i + 1])
                    qps_values.append(qps_value)
                except ValueError:
                    continue
        
        if qps_values and sm_util is not None and gpu_util is not None:
            avg_qps = np.mean(qps_values)
            
            # 获取base model和硬件规格
            base_model_info = BASE_MODEL_MAPPING.get(model, {"name": model, "params": 0.0})
            base_model = base_model_info["name"]
            model_params = base_model_info["params"]  # 模型参数量(B)
            # 处理H20和h20的兼容性
            hw_spec_key = gpu_card.lower() if gpu_card.lower() in HARDWARE_SPECS else gpu_card
            hw_spec = HARDWARE_SPECS.get(hw_spec_key, {})
            
            # 计算硬件效率指标
            theoretical_fp16_tflops = hw_spec.get('FP16', 0) / 1e12  # TFLOPS
            memory_gb = hw_spec.get('memsize', 0) / (1024**3)  # GB
            bandwidth_gbps = hw_spec.get('mem_bandwidth', 0) / (1024**3)  # GB/s
            
            # 计算效率指标
            qps_per_billion_params = avg_qps / model_params if model_params > 0 else 0
            sm_util_per_billion_params = sm_util / model_params if model_params > 0 else 0
            
            services.append({
                'service_name': service_name,
                'model': model,
                'base_model': base_model,
                'model_params': model_params,
                'gpu_card': gpu_card,
                'gpu_num': gpu_num,
                'sm_util': sm_util,
                'gpu_util': gpu_util,
                'avg_qps': avg_qps,
                'qps_values': qps_values,
                'theoretical_tflops': theoretical_fp16_tflops,
                'memory_gb': memory_gb,
                'bandwidth_gbps': bandwidth_gbps,
                'hw_spec_name': hw_spec.get('name', gpu_card),
                'qps_per_billion_params': qps_per_billion_params,
                'sm_util_per_billion_params': sm_util_per_billion_params,
                'quadrant': SELECTED_CASES[case_key]['quadrant']
            })
    
    return services

def classify_four_quadrants(services: List[Dict]) -> Dict[str, List[Dict]]:
    """将4个服务分类到四个象限"""
    qps_values = [s['avg_qps'] for s in services]
    sm_util_values = [s['sm_util'] for s in services]
    
    # 使用中位数作为分界线
    qps_median = np.median(qps_values)
    sm_util_median = np.median(sm_util_values)
    
    quadrants = {
        'high_qps_high_util': [],
        'high_qps_low_util': [],
        'low_qps_high_util': [],
        'low_qps_low_util': []
    }
    
    for service in services:
        quadrant = service['quadrant']
        quadrants[quadrant].append(service)
    
    return quadrants, qps_median, sm_util_median

def create_four_cases_visualization(services: List[Dict], quadrants: Dict, qps_median: float, sm_util_median: float, use_gpu_util: bool = False):
    """创建4个代表性案例的可视化图表"""
    plt.style.use('default')
    # 进一步减小图表尺寸，等比例缩小
    fig = plt.figure(figsize=(10, 6))
    
    # 调整子图布局，为右侧信息框留出空间
    ax_main = fig.add_subplot(1, 1, 1)
    
    # 选择使用的利用率指标
    util_field = 'gpu_util' if use_gpu_util else 'sm_util'
    util_label = 'GPU Utilization (%)' if use_gpu_util else 'SM Utilization (%)'
    util_name = 'GPU' if use_gpu_util else 'SM'
    
    # 为四个象限设置不同的颜色和标记
    quadrant_styles = {
        'high_qps_high_util': {'color': '#2E8B57', 'marker': 'o', 'label': 'High QPS, High Utilization'},
        'high_qps_low_util': {'color': '#FF6347', 'marker': 's', 'label': 'High QPS, Low Utilization'},
        'low_qps_high_util': {'color': '#4169E1', 'marker': '^', 'label': 'Low QPS, High Utilization'},
        'low_qps_low_util': {'color': '#DC143C', 'marker': 'D', 'label': 'Low QPS, Low Utilization'}
    }
    
    # 获取数据范围用于设置紧凑的坐标轴
    qps_values = [s['avg_qps'] for s in services]
    util_values = [s[util_field] for s in services]
    
    qps_min, qps_max = min(qps_values), max(qps_values)
    util_min, util_max = min(util_values), max(util_values)
    
    # 计算更小的边距（5%的数据范围作为边距）
    qps_margin = (qps_max - qps_min) * 0.05
    util_margin = (util_max - util_min) * 0.05
    
    # 设置更紧凑的坐标轴范围
    ax_main.set_xlim(qps_min - qps_margin, qps_max + qps_margin)
    # 纵轴范围略高于100，但刻度仍为0-100
    ax_main.set_ylim(0, 105)
    
    # 设置固定的刻度间距：每20个单位
    qps_tick_interval = 20
    util_tick_interval = 20
    
    # 计算刻度起始点，确保包含数据范围
    qps_start = int((qps_min - qps_margin) / qps_tick_interval) * qps_tick_interval
    qps_end = int((qps_max + qps_margin) / qps_tick_interval + 1) * qps_tick_interval
    
    util_start = int((util_min - util_margin) / util_tick_interval) * util_tick_interval
    util_end = int((util_max + util_margin) / util_tick_interval + 1) * util_tick_interval
    
    # 设置X轴刻度（每20个单位）
    qps_ticks = np.arange(qps_start, qps_end + qps_tick_interval, qps_tick_interval)
    ax_main.set_xticks(qps_ticks)
    
    # 设置Y轴刻度（每20个单位，但限制在0-100范围内）
    util_ticks = np.arange(0, 101, util_tick_interval)  # 0, 20, 40, 60, 80, 100
    ax_main.set_yticks(util_ticks)
    
    # 绘制散点图
    for service in services:
        quadrant = service['quadrant']
        style = quadrant_styles[quadrant]
        size = 80 + service['model_params'] * 8 
        ax_main.scatter(service['avg_qps'], service[util_field], 
                       c=style['color'], marker=style['marker'], s=size, alpha=0.8, 
                       edgecolors='black', linewidth=1,
                       label=style['label'])
    
    # 重新计算基于选择利用率的中位数
    util_median = np.median(util_values)
    
    # 添加分界线
    ax_main.axvline(x=qps_median, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, 
                   label=f'QPS Median: {qps_median:.1f}')
    ax_main.axhline(y=util_median, color='gray', linestyle='--', alpha=0.7, linewidth=1.5,
                   label=f'{util_name} Util Median: {util_median:.1f}%')
    
    # 调整象限标签位置，避免与数据点重叠
    ax_main.text(0.82, 0.82, 'High QPS\nHigh Utilization', transform=ax_main.transAxes, 
                fontsize=9, ha='center', va='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
    ax_main.text(0.82, 0.08, 'High QPS\nLow Utilization', transform=ax_main.transAxes, 
                fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8))
    ax_main.text(0.18, 0.82, 'Low QPS\nHigh Utilization', transform=ax_main.transAxes, 
                fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    ax_main.text(0.18, 0.08, 'Low QPS\nLow Utilization', transform=ax_main.transAxes, 
                fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightpink', alpha=0.8))
    
    # 计算相关性分析
    df = pd.DataFrame(services)
    qps_util_pearson = df['avg_qps'].corr(df[util_field], method='pearson')
    qps_params_pearson = df['avg_qps'].corr(df['model_params'], method='pearson')
    util_params_pearson = df[util_field].corr(df['model_params'], method='pearson')
    
    # 添加回归拟合线
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['avg_qps'], df[util_field])
    line_x = np.linspace(df['avg_qps'].min(), df['avg_qps'].max(), 100)
    line_y = slope * line_x + intercept
    ax_main.plot(line_x, line_y, 'r--', alpha=0.8, linewidth=2, 
                label=f'Linear Fit (R²={r_value**2:.3f})')
    
    # 设置坐标轴标签，调整字体大小
    ax_main.set_xlabel('Average QPS', fontsize=12)
    ax_main.set_ylabel(util_label, fontsize=12)
    ax_main.grid(True, alpha=0.3)
    
    # 调整刻度标签字体大小
    ax_main.tick_params(axis='both', which='major', labelsize=9)
    
    # 添加服务标签 - 智能调整位置避免重叠
    for service in services:
        qps = service['avg_qps']
        util = service[util_field]
        
        # 根据数据点位置智能调整标签偏移方向，并针对特定情况优化
        if qps < qps_median and util > util_median:
            # 左上象限 - 针对Qwen3-32B H20特殊处理，避免与象限标签重叠
            if 'qwen3-32b' in service['service_name'].lower():
                xytext = (15, -20)  # 向右下偏移更多，避免与象限标签重叠
            else:
                xytext = (12, -12)
            ha = 'left'
        elif qps > qps_median and util > util_median:
            # 右上象限 - 标签放在左下方
            xytext = (-12, -12)
            ha = 'right'
        elif qps < qps_median and util < util_median:
            # 左下象限 - 缩短距离
            if 'qwen2-7b' in service['service_name'].lower() and 'a800' in service['service_name'].lower():
                xytext = (8, 8)  # 缩短距离
            else:
                xytext = (12, 12)
            ha = 'left'
        else:
            # 右下象限 - 缩短距离
            if 'qwen3-4b' in service['service_name'].lower() and 'h20' in service['service_name'].lower():
                xytext = (-8, 8)  # 缩短距离
            else:
                xytext = (-12, 12)
            ha = 'right'
        
        ax_main.annotate(f"{service['base_model']}\n{service['gpu_card'].upper()}", 
                        (qps, util),
                        xytext=xytext, textcoords='offset points',
                        fontsize=8, fontweight='bold', ha=ha,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # 创建图例 - 放在图外部右边上方，调整字体大小
    handles, labels = ax_main.get_legend_handles_labels()
    legend = ax_main.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.8), 
                           fontsize=9, framealpha=0.9)
    
    # 添加统计信息文本框 - 放在图外部右边下方，调整字体大小
    stats_text = (f'Four Representative Cases Analysis:\n'
                 f'• Sample size: {len(services)} services\n'
                 f'• QPS range: {df["avg_qps"].min():.1f} - {df["avg_qps"].max():.1f}\n'
                 f'• {util_name} Util range: {df[util_field].min():.1f}% - {df[util_field].max():.1f}%\n'
                 f'• Parameter range: {df["model_params"].min():.1f}B - {df["model_params"].max():.1f}B\n'
                 f'• Correlation strength: {"Strong" if abs(qps_util_pearson) > 0.7 else "Moderate" if abs(qps_util_pearson) > 0.3 else "Weak"}')
    
    # 使用figtext在图外部添加统计信息框，调整字体大小
    fig.text(0.72, 0.35, stats_text, fontsize=8, verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # 调整布局以适应外部信息框，增加紧凑度
    plt.subplots_adjust(right=0.7, left=0.12, bottom=0.12, top=0.92)
    
    # 根据使用的指标调整文件名
    filename = f'four_cases_analysis_2_3_1_{"gpu" if use_gpu_util else "sm"}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Four cases visualization saved: {filename}")

def generate_four_cases_report(services: List[Dict], quadrants: Dict, qps_median: float, sm_util_median: float):
    """生成4个代表性案例的分析报告"""
    
    df = pd.DataFrame(services)
    qps_sm_corr = df['avg_qps'].corr(df['sm_util'])
    qps_gpu_corr = df['avg_qps'].corr(df['gpu_util'])
    
    report = []
    report.append("=" * 80)
    report.append("2.3.1 Service-level Observation: Four Representative Cases Analysis")
    report.append("=" * 80)
    report.append("")
    
    # 案例选择说明
    report.append("CASE SELECTION METHODOLOGY:")
    report.append("Selected 4 representative cases to demonstrate the four quadrants:")
    report.append("• High QPS + High Utilization: Optimal efficiency target")
    report.append("• High QPS + Low Utilization: Resource underutilization")
    report.append("• Low QPS + High Utilization: Compute-intensive workload")
    report.append("• Low QPS + Low Utilization: Primary optimization target")
    report.append("")
    
    # 案例概述
    report.append("SELECTED CASES OVERVIEW:")
    report.append(f"• Total cases analyzed: {len(services)}")
    report.append(f"• QPS range: {min([s['avg_qps'] for s in services]):.1f} - {max([s['avg_qps'] for s in services]):.1f}")
    report.append(f"• SM utilization range: {min([s['sm_util'] for s in services]):.1f}% - {max([s['sm_util'] for s in services]):.1f}%")
    report.append(f"• GPU utilization range: {min([s['gpu_util'] for s in services]):.1f}% - {max([s['gpu_util'] for s in services]):.1f}%")
    report.append(f"• Model parameter range: {min([s['model_params'] for s in services]):.1f}B - {max([s['model_params'] for s in services]):.1f}B")
    report.append("")
    
    # 相关性分析
    report.append("CORRELATION ANALYSIS:")
    report.append(f"• QPS vs SM Utilization: r = {qps_sm_corr:.3f}")
    report.append(f"• QPS vs GPU Utilization: r = {qps_gpu_corr:.3f}")
    report.append("")
    
    # 详细案例分析
    report.append("DETAILED CASE ANALYSIS:")
    report.append("")
    
    for i, service in enumerate(services, 1):
        quadrant_desc = {
            'high_qps_high_util': 'High QPS + High Utilization (Optimal)',
            'high_qps_low_util': 'High QPS + Low Utilization (Underutilized)',
            'low_qps_high_util': 'Low QPS + High Utilization (Compute-intensive)',
            'low_qps_low_util': 'Low QPS + Low Utilization (Optimization needed)'
        }
        
        report.append(f"Case {i}: {service['base_model']} on {service['hw_spec_name']}")
        report.append(f"  Quadrant: {quadrant_desc[service['quadrant']]}")
        report.append(f"  Service: {service['service_name']}")
        report.append(f"  Model Parameters: {service['model_params']:.1f}B")
        report.append(f"  GPU Configuration: {service['gpu_num']}x {service['hw_spec_name']}")
        report.append(f"  Performance Metrics:")
        report.append(f"    - Average QPS: {service['avg_qps']:.1f}")
        report.append(f"    - SM Utilization: {service['sm_util']:.1f}%")
        report.append(f"    - GPU Utilization: {service['gpu_util']:.1f}%")
        report.append(f"  Efficiency Metrics:")
        report.append(f"    - QPS per Billion Parameters: {service['qps_per_billion_params']:.2f}")
        report.append(f"    - SM Util per Billion Parameters: {service['sm_util_per_billion_params']:.2f}")
        report.append(f"  Hardware Specifications:")
        report.append(f"    - Theoretical FP16 Performance: {service['theoretical_tflops']:.1f} TFLOPS")
        report.append(f"    - Memory: {service['memory_gb']:.0f} GB")
        report.append(f"    - Memory Bandwidth: {service['bandwidth_gbps']:.0f} GB/s")
        report.append("")
    
    # 保存报告
    with open('four_cases_analysis_2_3_1_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("Four cases analysis report saved: four_cases_analysis_2_3_1_report.txt")

def main():
    print("Starting Four Representative Cases Analysis for 2.3.1...")
    
    # 解析数据
    services = parse_diff_file_selective('diff.txt')
    
    if len(services) == 0:
        print("No services found matching the selected cases!")
        return
    
    print(f"Parsed {len(services)} representative services with hardware specifications")
    if len(services) != 4:
        print(f"Warning: Expected 4 services, but got {len(services)}")
        for service in services:
            print(f"  - {service['model']} on {service['gpu_card']}")
    
    # 分类到四个象限
    quadrants, qps_median, sm_util_median = classify_four_quadrants(services)
    
    print("\nGenerating SM Utilization analysis...")
    # 创建SM利用率可视化
    create_four_cases_visualization(services, quadrants, qps_median, sm_util_median, use_gpu_util=False)
    
    print("Generating GPU Utilization analysis...")
    # 创建GPU利用率可视化
    create_four_cases_visualization(services, quadrants, qps_median, sm_util_median, use_gpu_util=True)
    
    # 生成分析报告
    generate_four_cases_report(services, quadrants, qps_median, sm_util_median)
    
    print("\nFour Cases Analysis completed!")
    print("• SM Utilization visualization: four_cases_analysis_2_3_1_sm.png")
    print("• GPU Utilization visualization: four_cases_analysis_2_3_1_gpu.png")
    print("• Analysis report: four_cases_analysis_2_3_1_report.txt")

if __name__ == "__main__":
    main()