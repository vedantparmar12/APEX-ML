"""Hyperparameter Optimization Agent - Fine-tunes models for optimal performance"""

import os
import json
from typing import Dict, Any, List

from utils.openrouter_client import llm_client
from utils.code_executor import code_executor
from config.config import CONFIG
from agents.hyperopt.prompts import HyperoptPrompts


class HyperoptAgent:
    """Agent for intelligent hyperparameter optimization"""
    
    def __init__(self, task_id: int = 1):
        self.task_id = task_id
        self.prompts = HyperoptPrompts()
        self.workspace_dir = os.path.join(CONFIG.workspace_dir, CONFIG.task_name, str(task_id))
    
    def run(self, code: str, current_score: float) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        
        results = {
            "task_id": self.task_id,
            "original_score": current_score,
            "optimization_strategy": None,
            "best_params": None,
            "best_score": current_score,
            "optimized_code": None
        }
        
        # Step 1: Analyze current hyperparameters
        print(f"[Hyperopt {self.task_id}] Analyzing current hyperparameters...")
        param_analysis = self._analyze_parameters(code)
        
        # Step 2: Generate optimization strategy
        print(f"[Hyperopt {self.task_id}] Generating optimization strategy...")
        strategy = self._generate_strategy(code, param_analysis)
        results["optimization_strategy"] = strategy
        
        # Step 3: Implement Bayesian optimization
        print(f"[Hyperopt {self.task_id}] Implementing optimization with Optuna...")
        optimized_code = self._implement_optimization(code, strategy)
        
        # Step 4: Run optimization
        print(f"[Hyperopt {self.task_id}] Running hyperparameter search...")
        result = code_executor.execute_code(
            optimized_code,
            filename="hyperopt_search.py",
            working_dir=self.workspace_dir,
            timeout=900  # 15 minutes for optimization
        )
        
        if result["success"]:
            # Extract best parameters from output
            best_params = self._extract_best_params(result["stdout"])
            results["best_params"] = best_params
            
            # Generate final code with best parameters
            final_code = self._apply_best_params(code, best_params)
            
            # Validate final performance
            final_result = code_executor.execute_code(
                final_code,
                filename="hyperopt_final.py",
                working_dir=self.workspace_dir
            )
            
            if final_result["success"] and final_result.get("score"):
                results["best_score"] = final_result["score"]
                results["optimized_code"] = final_code
        
        return results
    
    def _analyze_parameters(self, code: str) -> str:
        """Analyze current hyperparameters"""
        
        prompt = self.prompts.get_param_analysis_prompt(code)
        return llm_client.get_completion(prompt, temperature=0.3)
    
    def _generate_strategy(self, code: str, analysis: str) -> Dict:
        """Generate optimization strategy"""
        
        prompt = self.prompts.get_strategy_prompt(code, analysis)
        return llm_client.get_structured_output(prompt, temperature=0.5)
    
    def _implement_optimization(self, code: str, strategy: Dict) -> str:
        """Add Optuna optimization to code"""
        
        prompt = self.prompts.get_optimization_prompt(code, strategy)
        opt_code = llm_client.get_completion(prompt, temperature=0.3)
        return opt_code.replace("```python", "").replace("```", "").strip()
    
    def _extract_best_params(self, output: str) -> Dict:
        """Extract best parameters from optimization output"""
        
        prompt = self.prompts.get_extraction_prompt(output)
        return llm_client.get_structured_output(prompt, temperature=0.1)
    
    def _apply_best_params(self, code: str, params: Dict) -> str:
        """Apply best parameters to original code"""
        
        prompt = self.prompts.get_application_prompt(code, params)
        final_code = llm_client.get_completion(prompt, temperature=0.1)
        return final_code.replace("```python", "").replace("```", "").strip()