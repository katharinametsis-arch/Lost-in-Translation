# Lost-in-Translation

This repository contains the code and data used for a bachelor thesis project on automatic translation of semantically annotated English FrameNet sentences into Swedish.

The project evaluates how well large language models preserve FrameNet-style semantic annotations during translation, with a focus on Target preservation.

## Models

- OpenAI GPT-4.1 Mini
- Google Gemini 2.5 Flash

## Data

The dataset consists of 635 annotated English sentences extracted from FrameNet 1.7 via NLTK. The selected frames are:

- `Change_event_time`
- `Relative_time`
- `Taking_time`

The repository includes:

- Input English FrameNet sentences  
- Manually corrected Swedish gold standard translations  
- Model outputs from GPT-4.1 Mini and Gemini 2.5 Flash  
- Evaluation scripts  

## Evaluation

The evaluation includes:

- Target preservation accuracy using exact matching of `[Target ...]`  
- SacreBLEU scores comparing model output to the gold standard  
- MQM-inspired manual error analysis  

## Repository structure

```
data/
scripts/
results/
```

## Installation

Python 3.9 or later is required.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Reproducibility

All scripts required to reproduce the experiments are included:

- Prompting scripts for both models  
- Target preservation evaluation script  
- SacreBLEU calculation script  

The data required to replicate the results is also provided in the `data/` directory.

