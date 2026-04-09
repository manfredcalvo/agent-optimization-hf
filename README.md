# Prompt Optimization with Judge Alignment

A hands-on workshop demonstrating how to build better LLM applications using **MLflow 3** on **Databricks** — combining judge alignment with automated prompt optimization.

## Use Case: Banking Customer Complaint Classification

A two-stage classification pipeline for routing bank customer complaints:

1. **Complaint Analyzer** — extracts financial product, issue type, sentiment, and urgency from raw complaint text
2. **Complaint Router** — classifies into Department (Credit Cards, Mortgages, Personal Banking, Investment, Insurance) and Priority (Critical, High, Medium, Low)

## Notebooks

| Notebook | Description |
|----------|-------------|
| [00_overview](src/notebooks/00_overview.py) | Workshop overview — architecture, workflow, and key concepts |
| [01_align_judge](src/notebooks/01_align_judge.py) | Create and align an LLM judge with human feedback using **MemAlign** |
| [02_optimize_prompts](src/notebooks/02_optimize_prompts.py) | Optimize two chained prompts using **GEPA** with the aligned judge |

## Architecture

```
                  ┌─────────────────────────────────────────────┐
                  │          Notebook 1: Judge Alignment         │
                  │                                             │
                  │  Human Feedback ──► MemAlign ──► Aligned    │
                  │       +                          Judge      │
                  │  Initial Judge                     │        │
                  └─────────────────────────────────────┼───────┘
                                                        │
                                                        ▼
                  ┌─────────────────────────────────────────────┐
                  │        Notebook 2: Prompt Optimization       │
                  │                                             │
                  │  ┌──────────┐    ┌──────────┐              │
                  │  │ Prompt 1 │───►│ Prompt 2 │──► Output    │
                  │  │ Analyzer │    │  Router  │              │
                  │  └──────────┘    └──────────┘              │
                  │       │               │                     │
                  │       └───── GEPA ────┘                     │
                  │            Optimizer                        │
                  │               │                             │
                  │     Aligned Judge + Exact Match             │
                  └─────────────────────────────────────────────┘
```

## Key Technologies

- **[MemAlign](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/memalign/)** — Aligns LLM judges with human feedback using a dual-memory system (semantic guidelines + episodic examples). Fast (~40s) and cost-effective.
- **[GEPA](https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/)** — Genetic-Pareto prompt optimizer that iteratively refines prompts through LLM-driven reflection and automated evaluation.
- **[MLflow Prompt Registry](https://mlflow.org/docs/latest/genai/prompt-registry/)** — Version-controlled prompt management integrated with Unity Catalog.
- **[MLflow GenAI Evaluate](https://mlflow.org/docs/latest/genai/eval-monitor/)** — Evaluate LLM applications with custom and built-in scorers.

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- Access to Foundation Model API endpoints (`databricks-gpt-5-4-mini`, `databricks-claude-haiku-4-5`, `databricks-gte-large-en`)
- MLflow 3.5.0+

## Deployment

This project uses [Databricks Asset Bundles](https://docs.databricks.com/dev-tools/bundles/) for deployment.

```bash
# Deploy to your workspace
databricks bundle deploy

# Run the optimization notebook with custom parameters
databricks bundle run optimize_prompts --params max_metric_calls=100
```

### Configuration

Edit `databricks.yml` to set your workspace profile:

```yaml
targets:
  dev:
    workspace:
      profile: your_profile
```

Edit `src/notebooks/02_optimize_prompts.py` to set your Unity Catalog location:

```python
UC_CATALOG = "your_catalog"
UC_SCHEMA = "your_schema"
```

## Workshop Flow

1. **Run Notebook 00** — Read the overview to understand the end-to-end workflow
2. **Run Notebook 01** — Create and align the judge, copy the `experiment_id` from the output
3. **Run Notebook 02** — Paste the `experiment_id`, run baseline evaluation, optimize prompts, compare results
