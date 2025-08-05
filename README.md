# APEX-ML Project Documentation

## Overview

APEX-ML is an advanced automated machine learning system that utilizes a multi-agent architecture to handle the complete machine learning pipeline. The system employs specialized agents to perform different aspects of ML tasks including data preprocessing, feature engineering, model selection, hyperparameter optimization, ensemble methods, and result interpretation.

## Project Structure

## Project Structure

```
one-above-all/
├── agents/                 # Agent implementations
│   ├── initialization/     # Model search and evaluation
│   ├── refinement/        # Ablation studies and improvements
│   ├── ensemble/          # Solution combination strategies
│   ├── feature_engineering/ # Advanced feature creation
│   ├── hyperopt/          # Bayesian hyperparameter optimization
│   ├── error_analysis/    # Error pattern analysis
│   ├── cv_strategy/       # Advanced cross-validation
│   ├── explainability/    # Model interpretability
│   └── submission/        # Final submission generation
├── config/                # Configuration files
├── utils/                 # Helper utilities
│   ├── openrouter_client.py  # LLM interactions
│   ├── web_search.py         # DuckDuckGo search
│   └── code_executor.py      # Safe code execution
├── tasks/                 # Competition tasks
├── workspace/             # Working directory
└── main.py               # Main orchestrator
```

## Key Features

### Multi-Agent Architecture
The system employs a sophisticated multi-agent approach where each agent specializes in a specific aspect of the machine learning pipeline:

1. **Automated Data Analysis** - Initial data exploration and understanding
2. **Intelligent Feature Engineering** - Automated creation of relevant features
3. **Model Selection** - Automated selection of appropriate algorithms
4. **Hyperparameter Optimization** - Systematic parameter tuning using Optuna
5. **Ensemble Methods** - Combination of multiple models for improved performance
6. **Cross-Validation** - Robust validation strategies
7. **Error Analysis** - Deep dive into model failures and improvements
8. **Model Explainability** - SHAP-based model interpretation
9. **Automated Refinement** - Iterative model improvement

### Technology Stack

**Core ML Libraries**
- **scikit-learn** - Primary machine learning framework
- **XGBoost** - Gradient boosting framework
- **LightGBM** - Efficient gradient boosting
- **CatBoost** - Categorical feature handling
- **PyTorch** - Deep learning capabilities

**Data Processing**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing

**Optimization and Tuning**
- **Optuna** - Hyperparameter optimization
- **joblib** - Parallel processing

**Model Interpretation**
- **SHAP** - Model explainability
- **ELI5** - Model interpretation

**Visualization**
- **matplotlib** - Basic plotting
- **seaborn** - Statistical visualizations

**AI Integration**
- **OpenAI** - LLM integration for intelligent decision making
- **DuckDuckGo Search** - Web search for additional context

**Utilities**
- **requests** - HTTP client
- **aiohttp** - Asynchronous HTTP
- **BeautifulSoup4** - Web scraping
- **python-dotenv** - Environment variable management
- **tqdm** - Progress bars

## Architecture Overview

## Pipeline Stages

### Core Pipeline

**1. Initialization**
- Summarizes task for effective search
- Searches web for relevant ML models
- Evaluates each model on the dataset
- Merges best-performing solutions

**2. Refinement**
- Performs ablation studies to identify critical components
- Extracts code blocks with highest impact
- Tries multiple improvement strategies
- Selects best improvements based on validation scores

**3. Ensemble**
- Proposes novel ensemble strategies
- Implements techniques like voting, stacking, blending
- Iteratively refines ensemble approach
- Selects best ensemble configuration

**4. Submission**
- Selects overall best solution
- Adds submission generation code
- Creates competition-ready output file

### Enhanced Pipeline (Optional Agents)

**5. Feature Engineering**
- Analyzes current feature usage
- Generates polynomial and interaction features
- Creates domain-specific transformations
- Tests multiple feature strategies

**6. Hyperparameter Optimization**
- Identifies tunable parameters
- Uses Optuna for Bayesian optimization
- Tests parameter combinations efficiently
- Applies best parameters to final model

**7. Error Analysis**
- Analyzes prediction residuals
- Identifies patterns in errors
- Suggests targeted improvements
- Implements error-based corrections

**8. Cross-Validation Strategy**
- Selects appropriate CV method
- Implements advanced splitting strategies
- Provides robust performance estimates
- Prevents overfitting

**9. Model Explainability**
- Calculates feature importance
- Generates SHAP values
- Creates interpretability visualizations
- Improves model based on insights

## Configuration

Edit `config/config.py` to customize system behavior:

### Core Settings
- `model_name`: LLM model to use (via OpenRouter)
- `num_solutions`: Number of parallel solutions (default: 2)
- `num_model_candidates`: Models to search for (default: 2)
- `outer_loop_rounds`: Refinement iterations (default: 2)
- `inner_loop_rounds`: Strategies per refinement (default: 2)
- `ensemble_loop_rounds`: Ensemble iterations (default: 2)

### Enhanced Agents (set to False to disable)
- `use_feature_engineering`: Advanced feature creation (default: True)
- `use_hyperopt`: Bayesian hyperparameter optimization (default: True)
- `use_error_analysis`: Error pattern analysis (default: True)
- `use_cv_strategy`: Advanced cross-validation (default: True)
- `use_explainability`: Model interpretability (default: True)

### Agent-Specific Settings
- `hyperopt_trials`: Number of optimization trials (default: 50)
- `feature_engineering_strategies`: Number of FE strategies to try (default: 3)

## Installation and Setup

### Quick Start

**Prerequisites**
- Python 3.8+
- OpenRouter API key
- CUDA-capable GPU (recommended)

**Installation**

1. Clone the repository:
```bash
git clone https://github.com/vedantparmar12/APEX-ML.git
cd one-above-all
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenRouter API key:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

**Running the System**

Basic usage:
```bash
python main.py --task california-housing-prices
```

With custom parameters:
```bash
python main.py \
    --task your-task-name \
    --model anthropic/claude-3.5-sonnet \
    --num-solutions 3 \
    --api-key your-api-key
```

## Task Format

Place your tasks in the `tasks/` directory with this structure:

```
tasks/
└── your-task-name/
    ├── task_description.txt  # Task details and requirements
    ├── train.csv            # Training data
    └── test.csv             # Test data
```

Example `task_description.txt`:
```
# Task
Predict the target variable.

# Metric
root_mean_squared_error

# Submission Format
target 123.45 678.90 ...

# Dataset
Description of features and target...
```

## Monitoring Progress

The system provides detailed progress updates:

```
[STAGE 1] INITIALIZATION - Searching and evaluating models...
[Agent 1] Searching for model candidates...
[Agent 1] Evaluating model 1: XGBoost
[Agent 1] Evaluating model 2: LightGBM
[Agent 1] Merging solutions...

[STAGE 2] REFINEMENT - Running ablation studies...
[Refinement 1] Running ablation study...
[Refinement 1] Step 1 improved score to 0.123

[STAGE 3] ENSEMBLE - Creating ensemble solutions...
[Ensemble] Implementing plan: Weighted average based on validation scores...

[STAGE 4] SUBMISSION - Creating final submission file...
[Submission] Best solution selected with score: 0.115
```

## Project Metrics

- **Total Files**: 40
- **Total Lines of Code**: 6,865
- **Primary Language**: Python (33 files)
- **Documentation Files**: 2 Markdown files
- **Configuration Files**: 2 text files
- **Data Files**: 2 CSV files

### Largest Components
- Training data: `tasks/california-housing-prices/train.csv` (2,401 lines)
- Initialization agent: `agents/initialization/agent.py` (13,615 bytes)
- Refinement agent: `agents/refinement/agent.py` (12,621 bytes)
- Code executor utility: `utils/code_executor.py` (10,016 bytes)

## Agent Details

### Initialization Agent
Responsible for data loading, initial preprocessing, and problem setup. This agent analyzes the dataset characteristics and determines the appropriate ML approach.

### Feature Engineering Agent
Creates new features, handles categorical encoding, and performs feature selection. Uses domain knowledge and statistical methods to improve model input quality.

### Hyperparameter Optimization Agent
Utilizes Optuna for systematic hyperparameter tuning across different algorithms. Implements efficient search strategies to find optimal parameters.

### Ensemble Agent
Combines multiple models using various ensemble techniques such as voting, stacking, and blending to improve prediction accuracy.

### Refinement Agent
Analyzes model performance, identifies weaknesses, and suggests improvements. Implements iterative refinement strategies.

### Explainability Agent
Provides model interpretations using SHAP values and other explainability techniques to understand model decisions.

## Advanced Usage

### Custom Web Search
Modify `utils/web_search.py` to use different search engines or add specialized sources.

### Custom Models
Add preferred models to search queries by modifying prompts in agent files.

### Debugging
Enable verbose logging:
```python
CONFIG.verbose = True
CONFIG.save_intermediate_results = True
```

## Important Notes

- Ensure you have sufficient OpenRouter credits
- GPU recommended for faster model training
- Results may vary based on LLM model used
- Always verify submission format matches competition requirements

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing documentation
- Review agent logs in workspace directory

Remember: This system is designed to achieve competitive performance, but success also depends on understanding your specific competition requirements and data characteristics.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.
