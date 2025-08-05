"""Error Analysis Agent - Analyzes prediction errors to improve model"""

import os
import json
from typing import Dict, Any

from utils.openrouter_client import llm_client
from utils.code_executor import code_executor
from config.config import CONFIG


class ErrorAnalysisAgent:
    """Agent for analyzing and learning from prediction errors"""
    
    def __init__(self, task_id: int = 1):
        self.task_id = task_id
        self.workspace_dir = os.path.join(CONFIG.workspace_dir, CONFIG.task_name, str(task_id))
    
    def run(self, code: str, task_description: str) -> Dict[str, Any]:
        """Analyze errors and improve model"""
        
        results = {
            "task_id": self.task_id,
            "error_patterns": [],
            "improvement_strategies": [],
            "corrected_code": None,
            "improvement": 0
        }
        
        # Step 1: Add error analysis code
        print(f"[Error Analysis {self.task_id}] Adding error analysis...")
        analysis_code = self._add_error_analysis(code)
        
        # Step 2: Run analysis
        result = code_executor.execute_code(
            analysis_code,
            filename="error_analysis.py",
            working_dir=self.workspace_dir
        )
        
        if result["success"]:
            # Step 3: Identify error patterns
            patterns = self._identify_patterns(result["stdout"], task_description)
            results["error_patterns"] = patterns
            
            # Step 4: Generate improvement strategies
            strategies = self._generate_improvements(code, patterns)
            results["improvement_strategies"] = strategies
            
            # Step 5: Implement corrections
            corrected_code = self._implement_corrections(code, strategies)
            results["corrected_code"] = corrected_code
        
        return results
    
    def _add_error_analysis(self, code: str) -> str:
        """Add error analysis to code"""
        
        prompt = f"""Add comprehensive error analysis to this code:
```python
{code}
```

Add code to:
1. Calculate residuals (actual - predicted)
2. Analyze error distribution
3. Find samples with largest errors
4. Identify feature patterns in high-error samples
5. Print error statistics and patterns

Return complete code with error analysis."""
        
        return llm_client.get_completion(prompt, temperature=0.3)
    
    def _identify_patterns(self, output: str, task_description: str) -> List[Dict]:
        """Identify error patterns"""
        
        prompt = f"""Given this error analysis output:
{output[:2000]}...

And task description:
{task_description}

Identify:
1. What types of samples have high errors?
2. Are there specific feature ranges with poor predictions?
3. Is the model biased (over/under predicting)?
4. Are there outliers affecting performance?

Return findings as JSON list."""
        
        return llm_client.get_structured_output(prompt, temperature=0.5)
    
    def _generate_improvements(self, code: str, patterns: List[Dict]) -> List[str]:
        """Generate improvement strategies based on error patterns"""
        
        prompt = f"""Based on these error patterns:
{json.dumps(patterns, indent=2)}

Suggest 3 specific improvements:
1. Model architecture changes
2. Feature engineering to address errors
3. Special handling for problematic samples

Return as list of strategies."""
        
        response = llm_client.get_structured_output(prompt, temperature=0.6)
        return response if isinstance(response, list) else response.get("strategies", [])
    
    def _implement_corrections(self, code: str, strategies: List[str]) -> str:
        """Implement error corrections"""
        
        strategies_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(strategies)])
        
        prompt = f"""Implement these improvements to address prediction errors:
{strategies_text}

Current code:
```python
{code}
```

Return improved code that addresses the identified error patterns."""
        
        improved = llm_client.get_completion(prompt, temperature=0.3)
        return improved.replace("```python", "").replace("```", "").strip()