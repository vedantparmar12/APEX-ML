"""Prompts for Feature Engineering Agent"""

from typing import List


class FeatureEngineeringPrompts:
    """Prompts for feature engineering"""
    
    def get_feature_analysis_prompt(self, code: str, task_description: str) -> str:
        """Analyze current features"""
        
        return f"""# Task Description
{task_description}

# Current Solution
```python
{code}
```

# Your Task
Analyze the current feature engineering in this code:
1. What features are currently being used?
2. What preprocessing is applied?
3. What feature transformations are missing?
4. What domain-specific features could be created?
5. What interaction features might be valuable?

Be specific and concise."""

    def get_strategy_generation_prompt(self, code: str, analysis: str, task_description: str) -> str:
        """Generate feature engineering strategies"""
        
        return f"""# Task Description
{task_description}

# Feature Analysis
{analysis}

# Current Code (first 500 lines)
```python
{code[:2000]}...
```

# Your Task
Propose 3 advanced feature engineering strategies that could improve model performance.

For each strategy, include:
1. Strategy name
2. Description (2-3 sentences)
3. Specific features to create (list 3-5 features)

Focus on:
- Polynomial features for important variables
- Interaction terms between correlated features  
- Domain-specific transformations
- Temporal features if applicable
- Categorical encoding improvements
- Feature scaling/normalization
- Dimensionality reduction

Return as JSON:
[{{"name": str, "description": str, "features": [str]}}]"""

    def get_implementation_prompt(self, code: str, strategy_name: str, description: str, features: List[str]) -> str:
        """Implement feature engineering strategy"""
        
        features_list = "\n".join([f"- {f}" for f in features])
        
        return f"""# Current Solution
```python
{code}
```

# Feature Engineering Strategy
Name: {strategy_name}
Description: {description}

Features to implement:
{features_list}

# Your Task
Modify the code to implement these feature engineering improvements:
1. Add the new features in the preprocessing section
2. Ensure features are created for both train and test data
3. Handle any missing values appropriately
4. Scale features if needed
5. Update the model to use new features

# Requirements
- Return the complete modified code
- Maintain all existing functionality
- Print 'Final Validation Performance: {{score}}'
- Single code block only
- No additional text"""