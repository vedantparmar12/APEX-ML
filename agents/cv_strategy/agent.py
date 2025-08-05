"""Cross-Validation Strategy Agent - Implements robust validation strategies"""

import os
from typing import Dict, Any

from utils.openrouter_client import llm_client
from utils.code_executor import code_executor
from config.config import CONFIG


class CVStrategyAgent:
    """Agent for implementing advanced cross-validation strategies"""
    
    def __init__(self, task_id: int = 1):
        self.task_id = task_id
        self.workspace_dir = os.path.join(CONFIG.workspace_dir, CONFIG.task_name, str(task_id))
    
    def run(self, code: str, task_description: str) -> Dict[str, Any]:
        """Implement advanced CV strategy"""
        
        results = {
            "task_id": self.task_id,
            "cv_strategies_tested": [],
            "best_strategy": None,
            "best_code": None,
            "cv_scores": {}
        }
        
        # Define CV strategies to test
        strategies = [
            {
                "name": "Stratified K-Fold",
                "description": "Maintains class distribution in each fold",
                "suitable_for": "classification with imbalanced classes"
            },
            {
                "name": "Time Series Split",
                "description": "Respects temporal order for time-based data",
                "suitable_for": "time series or ordered data"
            },
            {
                "name": "Group K-Fold",
                "description": "Ensures groups don't overlap between train/test",
                "suitable_for": "grouped or clustered data"
            },
            {
                "name": "Repeated Stratified K-Fold",
                "description": "Multiple rounds of stratified CV for stability",
                "suitable_for": "small datasets needing robust estimates"
            }
        ]
        
        # Analyze which strategy fits best
        print(f"[CV Strategy {self.task_id}] Analyzing data characteristics...")
        best_strategy = self._select_strategy(code, task_description, strategies)
        
        # Implement selected strategy
        print(f"[CV Strategy {self.task_id}] Implementing {best_strategy['name']}...")
        cv_code = self._implement_cv_strategy(code, best_strategy)
        
        # Test implementation
        result = code_executor.execute_code(
            cv_code,
            filename="cv_strategy.py",
            working_dir=self.workspace_dir
        )
        
        if result["success"]:
            results["best_strategy"] = best_strategy
            results["best_code"] = cv_code
            results["cv_scores"] = self._extract_cv_scores(result["stdout"])
        
        return results
    
    def _select_strategy(self, code: str, task_description: str, strategies: list) -> Dict:
        """Select best CV strategy based on data characteristics"""
        
        strategies_text = "\n".join([
            f"- {s['name']}: {s['description']} (for {s['suitable_for']})"
            for s in strategies
        ])
        
        prompt = f"""Analyze this ML task and code to select the best cross-validation strategy:

Task Description:
{task_description}

Current Code (first 1000 chars):
```python
{code[:1000]}...
```

Available strategies:
{strategies_text}

Which strategy is most appropriate and why? Consider:
1. Data characteristics (size, distribution, temporal aspects)
2. Task type (classification/regression)
3. Potential data leakage risks

Return your choice as: {{"name": "strategy_name", "reason": "explanation"}}"""
        
        choice = llm_client.get_structured_output(prompt, temperature=0.3)
        
        # Find matching strategy
        for s in strategies:
            if s["name"].lower() in choice.get("name", "").lower():
                s["reason"] = choice.get("reason", "")
                return s
        
        return strategies[0]  # Default to first
    
    def _implement_cv_strategy(self, code: str, strategy: Dict) -> str:
        """Implement the selected CV strategy"""
        
        prompt = f"""Modify this code to use {strategy['name']} cross-validation:
```python
{code}
```

Requirements:
1. Replace current train/val split with {strategy['name']}
2. Use appropriate sklearn CV splitter
3. Average scores across all folds
4. Print fold scores and final average
5. Ensure consistent random state
6. Print 'Final Validation Performance: {{average_score}}'

Reason for this strategy: {strategy.get('reason', '')}

Return complete modified code."""
        
        cv_code = llm_client.get_completion(prompt, temperature=0.3)
        return cv_code.replace("```python", "").replace("```", "").strip()
    
    def _extract_cv_scores(self, output: str) -> Dict:
        """Extract CV scores from output"""
        
        prompt = f"""Extract cross-validation scores from this output:
{output[:1000]}

Return as: {{"fold_scores": [float], "mean": float, "std": float}}"""
        
        return llm_client.get_structured_output(prompt, temperature=0.1)