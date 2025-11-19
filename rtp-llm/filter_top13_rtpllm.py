#!/usr/bin/env python3
"""
筛选RTP-LLM数据中FPR最大的前13个案例
根据论文SEA层P85阈值规则：85个案例的15%约为13个
"""

import pandas as pd
import json

# 读取CSV数据
print("Reading llm_prof_integrated_data.csv...")
df = pd.read_csv('llm_prof_integrated_data.csv')
print(f"Total cases in original file: {len(df)}")
print(f"\nFPR (sea_fpr) range: {df['sea_fpr'].min():.6f} - {df['sea_fpr'].max():.6f}")

# 按sea_fpr降序排序，保留前13个
df_sorted = df.sort_values('sea_fpr', ascending=False)
df_top13 = df_sorted.head(13)

print(f"\nSelected top 13 cases with highest FPR:")
print(df_top13[['case_id', 'model_name', 'gpu_type', 'sea_fpr']].to_string(index=False))

# 保存到新的CSV文件
output_csv = 'rtp-llm_integrated_data.csv'
df_top13.to_csv(output_csv, index=False)
print(f"\n✓ Saved {len(df_top13)} cases to {output_csv}")

# 读取JSON数据并筛选
print("\nReading llm_prof_integrated_data.json...")
try:
    with open('llm_prof_integrated_data.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    print(f"Total cases in original JSON: {len(json_data)}")
    
    # 获取top13的case_id列表（转换为字符串以匹配JSON中的格式）
    top13_case_ids = set(str(cid) for cid in df_top13['case_id'].tolist())
    
    # 筛选JSON数据 - case_id在sea_data中
    filtered_json = []
    for item in json_data:
        if 'sea_data' in item and 'case_id' in item['sea_data']:
            if str(item['sea_data']['case_id']) in top13_case_ids:
                filtered_json.append(item)
    
    # 按sea_fpr降序排序
    filtered_json_sorted = sorted(
        filtered_json, 
        key=lambda x: x.get('sea_data', {}).get('fpr', 0), 
        reverse=True
    )
    
    # 保存到新的JSON文件
    output_json = 'rtp-llm_integrated_data.json'
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(filtered_json_sorted, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(filtered_json_sorted)} cases to {output_json}")
    
    # 显示筛选出的案例
    print(f"\nFiltered JSON cases:")
    for item in filtered_json_sorted:
        case_id = item.get('sea_data', {}).get('case_id', 'N/A')
        fpr = item.get('sea_data', {}).get('fpr', 0)
        model = item.get('sea_data', {}).get('model_name', 'N/A')
        print(f"  Case {case_id}: {model} - FPR={fpr:.6f}")
    
except FileNotFoundError:
    print("Warning: llm_prof_integrated_data.json not found, skipping JSON processing")
except Exception as e:
    print(f"Error processing JSON: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Summary:")
print(f"  Original cases: 22")
print(f"  Selected cases: 13 (top 15% based on 85 total RTP-LLM cases)")
print(f"  Selection criterion: Highest sea_fpr values")
print(f"  Output files:")
print(f"    - rtp-llm_integrated_data.csv")
print(f"    - rtp-llm_integrated_data.json")
print("="*60)