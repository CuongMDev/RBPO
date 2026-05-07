# PRISM: Multi-sampling Clustering with Consensus for Reliable Black-box Prompt Optimization

## Project Description

PRISM is a framework for optimizing prompts for black-box language models through multi-sampling strategies and consensus-based clustering. The project implements several prompt optimization methods:

- **BPO**: Black-box Prompt Optimization
- **RBPO**: Reliable Black-box Prompt Optimization  
- **MEPO**: Merit-Guided Prompt Optimization model
- **RMEPO**: Reliable Merit-Guided Prompt Optimization model

The framework uses clustering consensus to identify the most effective prompts and evaluates them across multiple language models.

## Reproduction Pipeline

The complete workflow consists of 5 sequential steps:

### Step 1: Prompt Generation & Inference
**File:** `src/step1-1_infer_bpo_rbpo_prompt.py` & `src/step1-2_infer_mepo_rmepo_prompt.py`

Generate and infer optimized prompts using:
- BPO/RBPO methods with various language models
- MEPO/RMEPO ensemble methods
- Stores prompt variations for subsequent clustering

```bash
python src/step1-1_infer_bpo_rbpo_prompt.py
python src/step1-2_infer_mepo_rmepo_prompt.py
```

### Step 2: Clustering & Selection  
**File:** `src/step2_clustering_and_selecting.py`

Perform consensus-based clustering on generated prompts:
- Uses embeddings model for semantic similarity (MiniLM, Gemma)
- Applies hierarchical clustering (Agglomerative Clustering)
- Selects representative prompts from clusters
- Reduces prompt diversity while preserving quality

```bash
python src/step2_clustering_and_selecting.py
```

### Step 3: Response Generation
**File:** `src/step3_response_generation.py`

Generate responses using selected prompts:
- Evaluate prompts on multiple base LLMs (Llama-2, Vicuna, Gemma)
- Test on multiple evaluation datasets (Dolly, Vicuna, Self-instruct, BPO-test)
- Batch processing for efficiency

```bash
python src/step3_response_generation.py
```

### Step 4: Response Verification
**File:** `src/step4_verify_response.py`

Evaluate and verify response quality:
- Use DeepSeek API for multi-criteria evaluation
- Score responses on 8 dimensions: Correctness, Relevance, Completeness, Clarity & Coherence, Usefulness & Helpfulness, Style & Tone, Conciseness, Safety & Compliance
- Identify consistent/mismatched predictions

```bash
python src/step4_verify_response.py
```

### Step 5: Post-Processing & Aggregation
**File:** `src/step5_post_processing.py`

Aggregate results and generate final comparisons:
- Compare method pairs (RBPO vs BPO, RBPO vs MEPO, RMEPO vs MEPO)
- Aggregate evaluation scores
- Generate statistical summaries

```bash
python src/step5_post_processing.py
```

## Quick Start

### Prerequisites
```bash
pip install -r src/requirements.txt
```

### Run Full Pipeline
```bash
cd src
python step1-1_infer_bpo_rbpo_prompt.py
python step1-2_infer_mepo_rmepo_prompt.py
python step2_clustering_and_selecting.py
python step3_response_generation.py
python step4_verify_response.py
python step5_post_processing.py
```

## Directory Structure

```
├── src/
│   ├── step1-1_infer_bpo_rbpo_prompt.py    # BPO/RBPO inference
│   ├── step1-2_infer_mepo_rmepo_prompt.py  # MEPO/RMEPO inference
│   ├── step2_clustering_and_selecting.py   # Clustering & selection
│   ├── step3_response_generation.py        # Response generation
│   ├── step4_verify_response.py            # Response verification
│   ├── step5_post_processing.py            # Aggregation
│   ├── config.py                           # Configuration
│   ├── helper.py                           # Helper functions
│   ├── utils.py                            # Utility functions
│   └── requirements.txt                    # Dependencies
├── evaluation/                             # Evaluation results
├── results/                                # Output results
└── testset/                                # Test datasets
```

## Output

- **Clustered prompts**: Selected representative prompts from each cluster
- **Generated responses**: Model outputs for each prompt-dataset combination
- **Evaluation scores**: Multi-criteria evaluation results
- **Comparison reports**: Method performance comparisons

## Configuration

Edit `src/config.py` & `src/helper.py`  to customize:
- Model cache paths
- Evaluation models and datasets
- Clustering parameters
- Batch sizes and device settings

## Lookout

**Important:** Each step requires loading large language models which consume significant GPU memory. To avoid out-of-memory (OOM) errors:

1. **Run each step separately** - Do not run all steps in a single script execution
2. **Reset memory between steps** - After completing each step, clear GPU cache and restart the Python kernel
3. **Monitor memory usage** - Track GPU memory consumption during execution
4. **Adjust batch sizes** - Reduce batch sizes in `config.py` if encountering OOM errors

**Recommended workflow:**
```bash
# Step 1
python src/step1-1_infer_bpo_rbpo_prompt.py
# [Clear GPU memory, restart kernel]

python src/step1-2_infer_mepo_rmepo_prompt.py
# [Clear GPU memory, restart kernel]

# Step 2
python src/step2_clustering_and_selecting.py
# [Clear GPU memory, restart kernel]

# ... continue for remaining steps
```
