"""Feature Engineering Agent - Creates advanced features for better model performance"""

import os
import json
from typing import Dict, List, Any, Optional
import numpy as np

from utils.openrouter_client import llm_client
from utils.code_executor import code_executor
from config.config import CONFIG
from agents.feature_engineering.prompts import FeatureEngineeringPrompts


class FeatureEngineeringAgent:
    """Agent responsible for advanced feature engineering"""
    
    def __init__(self, task_id: int = 1):
        self.task_id = task_id
        self.prompts = FeatureEngineeringPrompts()
        self.workspace_dir = os.path.join(CONFIG.workspace_dir, CONFIG.task_name, str(task_id))
    
    def run(self, code: str, score: float, task_description: str) -> Dict[str, Any]:
        """Generate advanced features"""
        
        results = {
            "task_id": self.task_id,
            "original_score": score,
            "feature_strategies": [],
            "best_features_code": None,
            "best_score": score
        }
        
        # Step 1: Analyze current features
        print(f"[Feature Engineering {self.task_id}] Analyzing current features...")
        feature_analysis = self._analyze_features(code, task_description)
        
        # Step 2: Generate feature engineering strategies
        print(f"[Feature Engineering {self.task_id}] Generating feature strategies...")
        strategies = self._generate_strategies(code, feature_analysis, task_description)
        
        # Step 3: Implement and test each strategy
        for i, strategy in enumerate(strategies):
            print(f"[Feature Engineering {self.task_id}] Testing strategy {i+1}: {strategy['name']}")
            
            # Generate code with new features
            enhanced_code = self._implement_strategy(code, strategy)
            
            # Evaluate
            result = code_executor.execute_code(
                enhanced_code,
                filename=f"features_strategy_{i}.py",
                working_dir=self.workspace_dir
            )
            
            strategy_result = {
                "strategy": strategy,
                "success": result["success"],
                "score": result.get("score"),
                "code": enhanced_code if result["success"] else None
            }
            
            results["feature_strategies"].append(strategy_result)
            
            # Update best if improved
            if result["success"] and result.get("score") is not None:
                if CONFIG.lower_is_better:
                    improved = result["score"] < results["best_score"]
                else:
                    improved = result["score"] > results["best_score"]
                
                if improved:
                    results["best_features_code"] = enhanced_code
                    results["best_score"] = result["score"]
        
        return results
    
    def _analyze_features(self, code: str, task_description: str) -> str:
        """Analyze current feature usage"""
        
        prompt = self.prompts.get_feature_analysis_prompt(code, task_description)
        return llm_client.get_completion(prompt, temperature=0.3)
    
    def _generate_strategies(self, code: str, analysis: str, task_description: str) -> List[Dict]:
        """Generate feature engineering strategies"""
        
        prompt = self.prompts.get_strategy_generation_prompt(
            code, analysis, task_description
        )
        
        strategies = llm_client.get_structured_output(prompt, temperature=0.7)
        
        if isinstance(strategies, list):
            return strategies[:3]  # Top 3 strategies
        else:
            return strategies.get("strategies", [])[:3]
    
    def _implement_strategy(self, code: str, strategy: Dict) -> str:
        """Implement a feature engineering strategy"""
        
        prompt = self.prompts.get_implementation_prompt(
            code,
            strategy.get("name", ""),
            strategy.get("description", ""),
            strategy.get("features", [])
        )
        
        enhanced_code = llm_client.get_completion(prompt, temperature=0.3)
        return enhanced_code.replace("```python", "").replace("```", "").strip()