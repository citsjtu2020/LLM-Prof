#!/usr/bin/env python3
"""
生成四种控制变量对比的综合表格
"""

import pandas as pd
import numpy as np
import os

def standardize_model_name(model_name):
    """标准化模型名称"""
    model_name_lower = model_name.lower()
    
    if 'qwen3' in model_name_lower or 'qwen-3' in model_name_lower:
        if '32b' in model_name_lower:
            return 'Qwen3-32B'
        elif '14b' in model_name_lower:
            return 'Qwen3-14B'
        elif '8b' in model_name_lower:
            return 'Qwen3-8B'
        elif '4b' in model_name_lower:
            return 'Qwen3-4B'
    
    if 'qwen2.5' in model_name_lower or 'qwen-2.5' in model_name_lower:
        if '32b' in model_name_lower:
            return 'Qwen2.5-32B'
        elif '14b' in model_name_lower:
            return 'Qwen2.5-14B'
        elif '7b' in model_name_lower:
            return 'Qwen2.5-7B'
        elif '3b' in model_name_lower:
            return 'Qwen2.5-3B'
    
    if 'qwq' in model_name_lower and '32b' in model_name_lower:
        return 'QwQ-32B'
    
    if 'llama' in model_name_lower:
        if '8b' in model_name_lower:
            return 'Llama-3.1-8B'
        elif '3b' in model_name_lower:
            return 'Llama-3.2-3B'
    
    return model_name

def extract_model_size(model_name):
    """提取模型参数量"""
    if '32b' in model_name.lower():
        return '32B'
    elif '14b' in model_name.lower():
        return '14B'
    elif '8b' in model_name.lower():
        return '8B'
    elif '7b' in model_name.lower():
        return '7B'
    elif '4b' in model_name.lower():
        return '4B'
    elif '3b' in model_name.lower():
        return '3B'
    return 'Unknown'

def load_data():
    """加载数据"""
    base_path = 'Evaluation'
    
    frameworks = {
        'vLLM': os.path.join(base_path, 'vllm_integrated_data.csv'),
        'SGLang': os.path.join(base_path, 'sglang_integrated_data.csv'),
        'RTP-LLM': os.path.join(base_path, 'rtp-llm_integrated_data.csv')
    }
    
    all_data = []
    for fw_name, file_path in frameworks.items():
        df = pd.read_csv(file_path)
        df['framework'] = fw_name
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df['standardized_model_name'] = combined_df['model_name'].apply(standardize_model_name)
    combined_df['model_size'] = combined_df['standardized_model_name'].apply(extract_model_size)
    combined_df['hw_config'] = combined_df['gpu_num'].astype(str) + '×' + combined_df['gpu_type']
    
    return combined_df

def analyze_comparison_1(df):
    """对比1：同参数量模型，同硬件，跨框架"""
    print("\n" + "="*80)
    print("对比1：同参数量模型 + 同硬件 → 跨框架对比")
    print("="*80)
    
    results = []
    
    # 找出同模型同硬件但不同框架的案例
    for model in df['standardized_model_name'].unique():
        model_df = df[df['standardized_model_name'] == model]
        model_size = model_df['model_size'].iloc[0]
        
        for hw in model_df['hw_config'].unique():
            hw_df = model_df[model_df['hw_config'] == hw]
            
            if len(hw_df) >= 2:  # 至少2个框架
                frameworks = hw_df['framework'].tolist()
                mie_values = hw_df['mea_mie'].tolist()
                
                max_mie = max(mie_values)
                min_mie = min(mie_values)
                variance = max_mie / min_mie
                
                result = {
                    'Model': model,
                    'Size': model_size,
                    'Hardware': hw,
                    'Frameworks': ' vs '.join(frameworks),
                    'MIE_Range': f"{min_mie:.2f}--{max_mie:.2f}",
                    'Variance': variance,  # 存储为数值
                    'Variance_Str': f"{variance:.2f}×"
                }
                results.append(result)
                
                print(f"\n{model} @ {hw}:")
                for fw, mie in zip(frameworks, mie_values):
                    print(f"  {fw}: MIE={mie:.3f}")
                print(f"  框架效应: {variance:.2f}×")
    
    return pd.DataFrame(results)

def analyze_comparison_2(df):
    """对比2：同参数量模型，不同硬件，跨框架"""
    print("\n" + "="*80)
    print("对比2：同参数量模型 + 不同硬件 → 跨框架对比")
    print("="*80)
    
    results = []
    
    # 按模型大小分组
    for size in df['model_size'].unique():
        if size == 'Unknown':
            continue
            
        size_df = df[df['model_size'] == size]
        
        # 找出该参数量下的跨框架案例
        models_in_size = size_df['standardized_model_name'].unique()
        
        for model in models_in_size:
            model_df = size_df[size_df['standardized_model_name'] == model]
            
            if model_df['framework'].nunique() >= 2:
                # 统计不同框架-硬件组合
                best_config = model_df.loc[model_df['mea_mie'].idxmin()]
                worst_config = model_df.loc[model_df['mea_mie'].idxmax()]
                
                variance = worst_config['mea_mie'] / best_config['mea_mie']
                
                result = {
                    'Model': model,
                    'Size': size,
                    'Best_Config': f"{best_config['framework']}@{best_config['hw_config']}",
                    'Best_MIE': best_config['mea_mie'],
                    'Worst_Config': f"{worst_config['framework']}@{worst_config['hw_config']}",
                    'Worst_MIE': worst_config['mea_mie'],
                    'Variance': variance,  # 存储为数值
                    'Variance_Str': f"{variance:.2f}×"
                }
                results.append(result)
                
                print(f"\n{model} ({size}):")
                print(f"  最优: {result['Best_Config']} (MIE={result['Best_MIE']:.2f})")
                print(f"  最差: {result['Worst_Config']} (MIE={result['Worst_MIE']:.2f})")
                print(f"  跨框架+硬件效应: {variance:.2f}×")
    
    return pd.DataFrame(results)

def analyze_comparison_3(df):
    """对比3：同参数量模型 + 同框架 → 不同硬件"""
    print("\n" + "="*80)
    print("对比3：同参数量模型 + 同框架 → 不同硬件对比")
    print("="*80)
    
    results = []
    
    for fw in df['framework'].unique():
        fw_df = df[df['framework'] == fw]
        
        for model in fw_df['standardized_model_name'].unique():
            model_df = fw_df[fw_df['standardized_model_name'] == model]
            model_size = model_df['model_size'].iloc[0]
            
            if len(model_df) >= 2:  # 至少2种硬件
                best_hw = model_df.loc[model_df['mea_mie'].idxmin()]
                worst_hw = model_df.loc[model_df['mea_mie'].idxmax()]
                
                variance = worst_hw['mea_mie'] / best_hw['mea_mie']
                
                result = {
                    'Framework': fw,
                    'Model': model,
                    'Size': model_size,
                    'Best_HW': best_hw['hw_config'],
                    'Best_MIE': best_hw['mea_mie'],
                    'Worst_HW': worst_hw['hw_config'],
                    'Worst_MIE': worst_hw['mea_mie'],
                    'HW_Variance': variance,  # 存储为数值
                    'HW_Variance_Str': f"{variance:.2f}×"
                }
                results.append(result)
                
                print(f"\n{fw} - {model} ({model_size}):")
                print(f"  最优硬件: {result['Best_HW']} (MIE={result['Best_MIE']:.2f})")
                print(f"  最差硬件: {result['Worst_HW']} (MIE={result['Worst_MIE']:.2f})")
                print(f"  硬件效应: {variance:.2f}×")
    
    return pd.DataFrame(results)

def analyze_comparison_4(df):
    """对比4：同框架 + 同硬件 → 不同参数量模型"""
    print("\n" + "="*80)
    print("对比4：同框架 + 同硬件 → 不同参数量模型对比")
    print("="*80)
    
    results = []
    
    for fw in df['framework'].unique():
        fw_df = df[df['framework'] == fw]
        
        for hw in fw_df['hw_config'].unique():
            hw_df = fw_df[fw_df['hw_config'] == hw]
            
            if hw_df['model_size'].nunique() >= 2:  # 至少2种参数量
                sizes = hw_df['model_size'].unique()
                
                for size in sizes:
                    size_df = hw_df[hw_df['model_size'] == size]
                    avg_mie = size_df['mea_mie'].mean()
                    
                    result = {
                        'Framework': fw,
                        'Hardware': hw,
                        'Model_Size': size,
                        'Avg_MIE': avg_mie,
                        'Cases': len(size_df)
                    }
                    results.append(result)
                
                print(f"\n{fw} @ {hw}:")
                for size in sorted(sizes):
                    size_df = hw_df[hw_df['model_size'] == size]
                    print(f"  {size}: Avg MIE={size_df['mea_mie'].mean():.2f} ({len(size_df)} cases)")
    
    return pd.DataFrame(results)

def main():
    print("加载数据...")
    df = load_data()
    
    print(f"\n数据集概览:")
    print(f"  总案例数: {len(df)}")
    print(f"  框架数: {df['framework'].nunique()}")
    print(f"  模型数: {df['standardized_model_name'].nunique()}")
    print(f"  硬件类型数: {df['gpu_type'].nunique()}")
    
    # 四种对比分析
    comp1_df = analyze_comparison_1(df)
    comp2_df = analyze_comparison_2(df)
    comp3_df = analyze_comparison_3(df)
    comp4_df = analyze_comparison_4(df)
    
    # 保存结果
    print("\n" + "="*80)
    print("保存分析结果...")
    print("="*80)
    
    comp1_df.to_csv('comparison1_same_model_same_hw.csv', index=False)
    print("  已保存: comparison1_same_model_same_hw.csv")
    
    comp2_df.to_csv('comparison2_same_model_diff_hw.csv', index=False)
    print("  已保存: comparison2_same_model_diff_hw.csv")
    
    comp3_df.to_csv('comparison3_same_fw_same_model.csv', index=False)
    print("  已保存: comparison3_same_fw_same_model.csv")
    
    comp4_df.to_csv('comparison4_same_fw_same_hw.csv', index=False)
    print("  已保存: comparison4_same_fw_same_hw.csv")
    
    # 生成统计摘要
    print("\n" + "="*80)
    print("四种对比的统计摘要:")
    print("="*80)
    
    print(f"\n对比1（同模型同硬件跨框架）:")
    print(f"  案例数: {len(comp1_df)}")
    if len(comp1_df) > 0:
        variances = comp1_df['Variance'].tolist()
        print(f"  平均框架效应: {np.mean(variances):.2f}×")
        print(f"  最大框架效应: {np.max(variances):.2f}×")
    
    print(f"\n对比2（同模型不同硬件跨框架）:")
    print(f"  案例数: {len(comp2_df)}")
    if len(comp2_df) > 0:
        variances = comp2_df['Variance'].tolist()
        print(f"  平均效应: {np.mean(variances):.2f}×")
        print(f"  最大效应: {np.max(variances):.2f}×")
    
    print(f"\n对比3（同框架同模型不同硬件）:")
    print(f"  案例数: {len(comp3_df)}")
    if len(comp3_df) > 0:
        variances = comp3_df['HW_Variance'].tolist()
        print(f"  平均硬件效应: {np.mean(variances):.2f}×")
        print(f"  最大硬件效应: {np.max(variances):.2f}×")
        
        # 按框架统计
        for fw in comp3_df['Framework'].unique():
            fw_variances = comp3_df[comp3_df['Framework']==fw]['HW_Variance'].tolist()
            print(f"    {fw}: 平均{np.mean(fw_variances):.2f}×, 最大{np.max(fw_variances):.2f}×")
    
    print(f"\n对比4（同框架同硬件不同模型）:")
    print(f"  案例数: {len(comp4_df)}")
    print(f"  覆盖的框架-硬件组合数: {comp4_df.groupby(['Framework', 'Hardware']).ngroups}")
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()