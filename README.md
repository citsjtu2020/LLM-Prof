# LLM-Prof

A comprehensive performance profiling and analysis framework for comparing mainstream LLM inference frameworks across different hardware platforms.

## Overview

LLM-Prof provides systematic performance evaluation and analysis for three major LLM inference frameworks:
- **RTP-LLM** - High-performance inference framework
- **SGLang** - Structured generation language framework
- **vLLM** - Fast and memory-efficient inference engine

The project conducts multi-dimensional performance analysis across various LLM models and GPU hardware configurations.

## Project Structure

```
LLM-Prof/
├── rtp-llm/              # RTP-LLM framework experiments and analysis
│   ├── sea_experiment/   # Static Experiment Analysis
│   ├── mea_experiment/   # Memory Experiment Analysis
│   ├── oea_experiment/   # Optimization Experiment Analysis
│   └── data_integration_script.py
├── sglang/               # SGLang framework experiments and analysis
│   ├── sea_fpr_cdf_analysis/
│   ├── mea_analysis_for_sglang/
│   ├── oea_analysis_for_sglang/
│   └── sglang_data_integration_script.py
├── vllm/                 # vLLM framework experiments and analysis
│   ├── sea_fpr_cdf_analysis/
│   ├── mea_analysis_for_vllm/
│   ├── oea_analysis_for_vllm/
│   └── vllm_data_integration_script.py
├── Evaluation/           # Cross-framework comparative analysis
│   ├── mea_evaluation/   # Memory analysis comparisons
│   ├── oea_evaluation/   # Optimization analysis comparisons
│   └── combined_framework_fpr_cdf_analysis.py
├── observation-before-experiment/  # Pre-experiment observations
├── hardware_spec.md      # Hardware specifications database
└── README.md
```

## Supported Models

The framework supports comprehensive analysis for the following LLM families:

### Llama Series
- Llama-3.1-8B
- Llama-3.2-3B

### Qwen2.5 Series
- Qwen2.5-3B
- Qwen2.5-7B
- Qwen2.5-14B
- Qwen2.5-32B

### Qwen3 Series
- Qwen3-4B
- Qwen3-8B
- Qwen3-14B
- Qwen3-32B
- Qwen3-30B-A3B (MoE)

## Supported Hardware

### NVIDIA GPUs
- A10 (24GB, 600GB/s bandwidth)
- L20 (48GB, 864GB/s bandwidth)
- H20 (96GB/141GB variants)
- A100 SXM4 (80GB)
- A800 SXM4 (80GB)
- H800 (80GB)
- H100 (80GB)

See `hardware_spec.md` for detailed specifications including memory bandwidth, peak FLOPS (FP16/INT8), and interconnect bandwidth.

## Key Features

### Visualization
- Cumulative Distribution Function (CDF) plots
- Roofline model diagrams
- Box plots for cross-framework comparison
- Multi-framework quadrant analysis

### Data Integration
- Automated data collection and integration scripts
- CSV-based data storage for reproducibility
- Framework-specific data processing pipelines

### Cross-Framework Evaluation

Generate comparative analysis across all frameworks:

```bash
cd Evaluation

# Combined FPR CDF analysis
python combined_framework_fpr_cdf_analysis.py

# MEA comparison tables
cd mea_evaluation
python generate_comparison_table.py

# OEA efficiency analysis
cd oea_evaluation
python generate_enhanced_cdf.py
python generate_boxplot_by_framework.py
```


## Data Format

All experimental data is stored in CSV format with the following key fields:
- Pod name/identifier
- Model name and configuration
- GPU type and specifications
- Token size and batch information
- Performance metrics (FPR, efficiency scores)
- Timing breakdowns

## Requirements

### Python Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- pathlib


## Contributing

This is a research project for systematic LLM inference framework performance analysis. Contributions for additional frameworks, models, or hardware platforms are welcome.

## Analysis Methodology

The project follows a multi-stage analysis pipeline:

1. **Data Collection**: Run inference workloads across frameworks
2. **Data Integration**: Consolidate results using integration scripts
3. **Framework Analysis**: Individual framework performance profiling
4. **Cross-Framework Evaluation**: Comparative analysis and visualization
5. **Bottleneck Identification**: Operator-level performance debugging

## License

## Citation

## Contact
