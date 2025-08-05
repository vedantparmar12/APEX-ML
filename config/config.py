"""Configuration for One-Above-All ML Engineering System"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Main configuration for the ML Engineering system"""
    
    # OpenRouter API Configuration
    openrouter_api_key: str = os.environ.get("OPENROUTER_API_KEY", "")
    model_name: str = "anthropic/claude-3.5-sonnet"  # Can use other models via OpenRouter
    
    # Task Configuration
    data_dir: str = "./tasks/"
    workspace_dir: str = "./workspace/"
    task_name: str = "california-housing-prices"
    task_type: str = "Tabular Regression"
    lower_is_better: bool = True  # For RMSE, lower is better
    
    # Agent Configuration
    num_solutions: int = 2  # Number of parallel solutions to generate
    num_model_candidates: int = 2  # Number of models to search for
    max_retry: int = 10  # Maximum retries for failed operations
    max_debug_rounds: int = 5  # Maximum debugging iterations
    
    # Refinement Configuration
    inner_loop_rounds: int = 2  # Number of refinement strategies per code block
    outer_loop_rounds: int = 2  # Number of code blocks to refine
    num_top_plans: int = 3  # Top plans to consider in refinement
    
    # Ensemble Configuration
    ensemble_loop_rounds: int = 2  # Number of ensemble iterations
    
    # Execution Configuration
    exec_timeout: int = 600  # Timeout for code execution in seconds
    use_gpu: bool = True  # Enable GPU usage if available
    random_seed: int = 42  # For reproducibility
    
    # Search Configuration
    search_engine: str = "duckduckgo"  # Alternative to Google Search
    max_search_results: int = 5
    
    # Logging Configuration
    verbose: bool = True
    save_intermediate_results: bool = True
    
    # Advanced Agents Configuration
    use_feature_engineering: bool = True
    use_hyperopt: bool = True
    use_error_analysis: bool = True
    use_cv_strategy: bool = True
    use_explainability: bool = True
    
    # Agent-specific settings
    hyperopt_trials: int = 50  # Number of hyperopt trials
    feature_engineering_strategies: int = 3  # Number of FE strategies to try
    
    def validate(self):
        """Validate configuration"""
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable must be set")
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            
        if not os.path.exists(self.workspace_dir):
            os.makedirs(self.workspace_dir, exist_ok=True)


# Global configuration instance
CONFIG = Config()