"""Prompts for Hyperparameter Optimization Agent"""

import json


class HyperoptPrompts:
    """Prompts for hyperparameter optimization"""
    
    def get_param_analysis_prompt(self, code: str) -> str:
        """Analyze current hyperparameters"""
        
        return f"""Analyze the hyperparameters in this ML code:
```python
{code[:2000]}...
```

Identify:
1. What model(s) are being used?
2. What hyperparameters are currently set?
3. Which parameters are hardcoded vs tuned?
4. What parameter ranges would be reasonable to search?

Be specific about parameter names and current values."""

    def get_strategy_prompt(self, code: str, analysis: str) -> str:
        """Generate optimization strategy"""
        
        return f"""Based on this hyperparameter analysis:
{analysis}

Create an optimization strategy for these models.

For each model, specify:
1. Parameters to optimize
2. Search ranges (min, max, type)
3. Whether to use log scale
4. Priority (high/medium/low)

Return as JSON:
{{
    "models": [
        {{
            "name": "model_name",
            "parameters": [
                {{
                    "name": "param_name",
                    "type": "int|float|categorical",
                    "range": [min, max] or ["option1", "option2"],
                    "log_scale": true|false,
                    "priority": "high|medium|low"
                }}
            ]
        }}
    ],
    "n_trials": 50
}}"""

    def get_optimization_prompt(self, code: str, strategy: dict) -> str:
        """Add Optuna optimization"""
        
        params_text = json.dumps(strategy, indent=2)
        
        return f"""Add Optuna hyperparameter optimization to this code:
```python
{code}
```

Optimization strategy:
{params_text}

Requirements:
1. Import optuna
2. Create objective function that:
   - Suggests parameters according to strategy
   - Trains model with suggested params
   - Returns validation score to minimize/maximize
3. Run optimization for specified trials
4. Print best parameters and score
5. Train final model with best params
6. Print 'Final Validation Performance: {{score}}'

Return complete code with optimization."""

    def get_extraction_prompt(self, output: str) -> str:
        """Extract best parameters"""
        
        return f"""Extract the best hyperparameters from this Optuna output:
{output[:1500]}

Return as JSON dict with parameter names and values."""

    def get_application_prompt(self, code: str, params: dict) -> str:
        """Apply best parameters"""
        
        params_text = json.dumps(params, indent=2)
        
        return f"""Apply these optimized hyperparameters to the code:
{params_text}

Original code:
```python
{code}
```

Replace all hyperparameters with the optimized values.
Return complete code with optimized parameters."""