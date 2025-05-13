# Enhancing Process Mining with LLMs by Applying Instruction Tuning

## Description
This repository contains the dataset preparation, training and evaluation scripts as described in Master's Thesis "Enhancing Process Mining with LLMs by Applying Instruction Tuning".

## Structure
```
.
├── dataset_analysis                        
│   └── dataset_analysis.ipynb             # Python notebook for the analysis of datasets features mentioned in Section 5.1 of the thesis.
├── datasets                               # initial datasets must be added here (see below, Setup)
│   └── train_val_test.pkl                 # used for reproducible creation of dataset splits
├── eval                                   # Python notebooks for the in-depth analysis of results as described in Sections 5.3.2-5.3.3 of the thesis.
│   ├── classified_domains.csv             # classification of domains with confidence scores used for Doman-Specific Analysis
│   └── ... 
├── prompts_per_cluster                    # prompt variations (paraphrasing) per cluster, used during training
│   └── ... 
├── data_loading_util.py                   # utility to load and clean the initial datasets
├── evaluation_pipeline.py                 # main file for running evaluation experiments
├── evaluation_util.py                     # utility for evaluation, mainly to handle complex . adapted from [a-rebmann](https://github.com/a-rebmann/llms4pm/blob/main/eval_util.py)
├── instruction_tuning_pipeline.py         # main file for running instruction tuning experiments
├── instruction_tuning_util.py             # utility to mix training samples into one dataset and fill prompt templates with real data, as well as convert into LLM-compatible format
├── prompt_builder_util.py                 # utility to build prompts with optional few-shot setting, used during evaluation
└── ...
```

## Running the Experiments

### Hardware Requirements
Nvidia GPU with at least 80GB of memory is required to run the experiments (e.g., A100).

### Setup 

1. Set up a virtual env:

```shell
pip install -r requirements.txt
```

2. Download initial datasets, available via: [datasets](https://zenodo.org/records/14273161). Place A_SAD.csv, T_SAD.csv, S-NAP.csv, S-PMD.csv files in datasets/ folder.

### Instruction Tuning

1. Adjust parameters in instruction_tuning_pipeline.py.

2. Run instruction_tuning_pipeline.py.

### Evaluation

1. Adjust parameters in evaluation_pipeline.py.

2. Run evaluation_pipeline.py.
