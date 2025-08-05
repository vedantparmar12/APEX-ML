"""Prompts for the Ensemble Agent"""

from typing import List, Dict
from config.config import CONFIG


class EnsemblePrompts:
    """Collection of prompts for ensemble phase"""
    
    def get_initial_ensemble_prompt(self, solution_codes: List[str]) -> str:
        """Get prompt for initial ensemble plan"""
        
        solutions_text = ""
        for i, code in enumerate(solution_codes):
            solutions_text += f"# Python Solution {i+1}\n```python\n{code}\n```\n\n"
        
        return f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- We will now provide {len(solution_codes)} Python Solutions used for the competition.
- Your task is to propose a plan to ensemble the {len(solution_codes)} solutions to achieve the best performance.

{solutions_text}

# Your task
- Suggest a plan to ensemble the {len(solution_codes)} solutions. You should concentrate on how to merge, not the other parts like hyperparameters.
- The suggested plan should be novel, effective, and easy to implement.
- All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.
- Consider various ensemble techniques:
  - Simple averaging or voting
  - Weighted averaging based on validation performance
  - Stacking with a meta-learner
  - Blending with different fold strategies
  - Rank averaging for robust predictions

# Response format
- Your response should be an outline/sketch of your proposed solution in natural language.
- There should be no additional headings or text in your response.
- Plan should not modify the original solutions too much since execution errors can occur.
- Focus on combining predictions rather than retraining models."""
    
    def get_implement_ensemble_prompt(self, solution_codes: List[str], plan: str) -> str:
        """Get prompt for implementing ensemble"""
        
        solutions_text = ""
        for i, code in enumerate(solution_codes):
            solutions_text += f"# Python Solution {i+1}\n```python\n{code}\n```\n\n"
        
        return f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to ensemble {len(solution_codes)} Python Solutions for better performance based on the ensemble plan.
- We will now provide the Python Solutions and the ensemble plan.

{solutions_text}

# Ensemble Plan
{plan}

# Your task
- Implement the ensemble plan with the provided solutions.
- Unless mentioned in the ensemble plan, do not modify the original Python Solutions too much.
- All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.
- The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.
- Make sure to properly handle the predictions from each model and combine them according to the plan.

# Response format required
- Your response should be a single markdown code block (wrapped in ```) which is the ensemble of {len(solution_codes)} Python Solutions.
- There should be no additional headings or text in your response.
- Do not modify original Python Solutions especially the preprocessing and feature engineering parts to avoid errors.
- Do not subsample or introduce dummy variables. You have to provide full new Python Solution using the {len(solution_codes)} provided solutions.
- Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {{final_validation_score}}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Do not modify the original codes too much and implement the plan since new errors can occur."""
    
    def get_refined_ensemble_prompt(self, solution_codes: List[str], previous_attempts: List[Dict]) -> str:
        """Get prompt for refined ensemble plan"""
        
        solutions_text = ""
        for i, code in enumerate(solution_codes):
            solutions_text += f"# Python Solution {i+1}\n```python\n{code[:500]}...\n```\n\n"
        
        # Sort previous attempts by score
        sorted_attempts = sorted(
            previous_attempts,
            key=lambda x: x["score"],
            reverse=not CONFIG.lower_is_better
        )
        
        # Show top attempts
        attempts_text = "# Ensemble plans you have tried\n\n"
        for i, attempt in enumerate(sorted_attempts[:CONFIG.num_top_plans]):
            attempts_text += f"## Plan {i+1}: {attempt['plan']}\n"
            attempts_text += f"## Score: {attempt['score']:.5f}\n\n"
        
        criteria = "lower" if CONFIG.lower_is_better else "higher"
        
        return f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you have to ensemble {len(solution_codes)} Python Solutions for better performance.
- We will provide the Python Solutions and the ensemble plans you have tried.

{solutions_text}

{attempts_text}

# Your task
- Suggest a better plan to ensemble the {len(solution_codes)} solutions. You should concentrate on how to merge, not the other parts like hyperparameters.
- The suggested plan must be easy to implement, novel, and effective.
- The suggested plan should be different from the previous plans you have tried and should receive a {criteria} score.
- Consider advanced ensemble techniques you haven't tried:
  - Different weighting schemes (e.g., based on model confidence)
  - Multi-level stacking with different meta-learners
  - Ensemble pruning to remove redundant models
  - Dynamic weighting based on input characteristics
  - Bayesian model averaging

# Response format
- Your response should be an outline/sketch of your proposed solution in natural language.
- There should be no additional headings or text in your response.
- Plan should not modify the original solutions too much since execution errors can occur."""