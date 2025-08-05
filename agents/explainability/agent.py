"""Model Explainability Agent - Adds interpretability to ML models"""

import os
from typing import Dict, Any

from utils.openrouter_client import llm_client
from utils.code_executor import code_executor
from config.config import CONFIG


class ExplainabilityAgent:
    """Agent for adding model interpretability"""
    
    def __init__(self, task_id: int = 1):
        self.task_id = task_id
        self.workspace_dir = os.path.join(CONFIG.workspace_dir, CONFIG.task_name, str(task_id))
    
    def run(self, code: str) -> Dict[str, Any]:
        """Add explainability and extract insights"""
        
        results = {
            "task_id": self.task_id,
            "feature_importance": None,
            "insights": [],
            "improved_code": None
        }
        
        # Step 1: Add SHAP/feature importance analysis
        print(f"[Explainability {self.task_id}] Adding interpretability analysis...")
        explain_code = self._add_explainability(code)
        
        # Step 2: Run analysis
        result = code_executor.execute_code(
            explain_code,
            filename="explainability.py",
            working_dir=self.workspace_dir
        )
        
        if result["success"]:
            # Step 3: Extract insights
            insights = self._extract_insights(result["stdout"])
            results["insights"] = insights
            
            # Step 4: Improve model based on insights
            improved_code = self._improve_based_on_insights(code, insights)
            results["improved_code"] = improved_code
        
        return results
    
    def _add_explainability(self, code: str) -> str:
        """Add SHAP or feature importance analysis"""
        
        prompt = f"""Add model explainability to this code:
```python
{code}
```

Add:
1. Feature importance calculation (if tree-based model)
2. SHAP values (if possible)
3. Partial dependence plots for top features
4. Analysis of feature interactions
5. Print top 10 most important features

Use try/except for SHAP in case it's not compatible.
Return complete code with explainability."""
        
        return llm_client.get_completion(prompt, temperature=0.3)
    
    def _extract_insights(self, output: str) -> list:
        """Extract insights from explainability output"""
        
        prompt = f"""From this model explainability output:
{output[:2000]}

Extract key insights:
1. Which features are most important?
2. Are there surprising unimportant features?
3. Do feature interactions matter?
4. Are there features that could be removed?
5. Suggestions for feature engineering based on importance

Return as list of actionable insights."""
        
        response = llm_client.get_structured_output(prompt, temperature=0.5)
        return response if isinstance(response, list) else response.get("insights", [])
    
    def _improve_based_on_insights(self, code: str, insights: list) -> str:
        """Improve model based on explainability insights"""
        
        insights_text = "\n".join([f"- {insight}" for insight in insights])
        
        prompt = f"""Based on these model insights:
{insights_text}

Improve this code:
```python
{code[:2000]}...
```

Improvements to make:
1. Remove unimportant features
2. Add polynomial features for important ones
3. Create interactions between top features
4. Adjust model focus on important features

Return complete improved code."""
        
        improved = llm_client.get_completion(prompt, temperature=0.3)
        return improved.replace("```python", "").replace("```", "").strip()