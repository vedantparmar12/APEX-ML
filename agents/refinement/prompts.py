"""Prompts for the Refinement Agent - Hybrid approach with exact critical prompts"""

from typing import List, Dict


class RefinementPrompts:
    """Collection of prompts for refinement phase"""
    
    def get_ablation_prompt(self, code: str, previous_ablations: List[str]) -> str:
        """Get prompt for ablation study - EXACT from ADK"""
        
        if previous_ablations:
            prev_ablations_str = ""
            for i, ablation in enumerate(previous_ablations):
                prev_ablations_str += f"## Previous ablation study result {i+1}\n{ablation}\n\n"
            
            return f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to perform an ablation study on the current Python solution to know which parts of the code contribute the most to the overall performance.
- We will now provide a current Python solution.
- We will also provide the summaries of previous ablation studies.

# Python solution
```python
{code}
```

{prev_ablations_str}

# Instructions
- You need you to generate a simple Python code that performs an ablation study on the train.py script.
- The generated code should create variations by modifying or disabling parts (2-3 parts) of the training process.
- Your ablation study should concentrate on the other parts that have not been previously considered.
- For each ablation, print out how the modification affects the model's performance.

# Response format
- There should be no additional headings or text in your response.
- The Python code for the ablation study should not load test data. It should only focus on training and evaluating the model on the validation set.
- The code should include a printing statement that shows the performance of each ablation.
- The code should consequently print out what part of the code contributes the most to the overall performance."""
        
        else:
            return f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to perform an ablation study on the current Python solution to know which parts of the code contribute the most to the overall performance.
- We will now provide a current Python solution.

# Python solution
```python
{code}
```
# Instructions
- You need to generate a simple Python code that performs an ablation study on the above Python solution script.
- The generated code should create variations by modifying or disabling parts (1-2 simple parts) of the training process.
- For each ablation, print out how the modification affects the model's performance.

# Response format
- There should be no additional headings or text in your response.
- The Python code for the ablation study should not load test data. It should only focus on training and evaluating the model on the validation set.
- The code should include a printing statement that shows the performance of each ablation.
- The code should consequently print out which part of the code contributes the most to the overall performance."""
    
    def get_ablation_summary_prompt(self, ablation_code: str, output: str) -> str:
        """Get prompt for summarizing ablation results - EXACT from ADK"""
        
        return f"""# Your code for ablation study was:
```python
{ablation_code}
```

# Ablation study results after running the above code:
{output}

# Your task
- Summarize the result of ablation study based on the code and printed output."""
    
    def get_extract_block_prompt(self, code: str, ablation_summary: str, previous_refinements: List[Dict]) -> str:
        """Get prompt for extracting code block and planning improvement - EXACT from ADK with optimization"""
        
        if previous_refinements:
            prev_blocks_str = "\n".join([f"# Previous code block {i+1}\n{ref['code_block']}\n" for i, ref in enumerate(previous_refinements)])
            
            return f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to extract a code block from the current Python solution and improve the extracted block for better performance.
- Your suggestion should be based on the ablation study results of the current Python solution.
- We will now provide the current Python solution and the ablation study results.
- We also provide code blocks which you have tried to improve previously.

# Python solution
```python
{code}
```

# Ablation study results
{ablation_summary}

{prev_blocks_str}

# Your task
- Given the ablation study results, suggest an effective next plan to improve the above Python script.
- The plan should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences).
- Please avoid plan which can make the solution's running time too long (e.g., searching hyperparameters in a very large search space).
- Try to improve the other part which was not considered before.
- Also extract the code block from the above Python script that need to be improved according to the proposed plan. You should try to extract the code block which was not improved before.

# Response format
- Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences) and a single markdown code block which is the code block that need to be improved.
- The code block can be long but should be exactly extracted from the Python script provided above.

Use this JSON schema:

Refine_Plan = {{"code_block": str, "plan": str}}
Return: list[Refine_Plan]"""
        else:
            return f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to extract a code block from the current Python solution and improve the extracted block for better performance.
- Your suggestion should be based on the ablation study results of the current Python solution.
- We will now provide the current Python solution and the ablation study results.

# Python solution
```python
{code}
```

# Ablation study results
{ablation_summary}

# Your task
- Given the ablation study results, suggest an effective next plan to improve the above Python script.
- The plan should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences).
- Please avoid plan which can make the solution's running time too long (e.g., searching hyperparameters in a very large search space).
- Also extract the code block from the above Python script that need to be improved according to the proposed plan.

# Response format
- Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences) and a single markdown code block which is the code block that need to be improved.
- The code block can be long but should be exactly extracted from the Python script provided above.

Use this JSON schema:

Refine_Plan = {{"code_block": str, "plan": str}}
Return: list[Refine_Plan]"""
    
    def get_implement_plan_prompt(self, full_code: str, code_block: str, plan: str) -> str:
        """Get prompt for implementing improvement plan - Optimized for full code replacement"""
        
        return f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to improve the Python solution by refining a specific code block.
- We will provide the full Python solution, the code block to improve, and the improvement plan.

# Current Python solution
```python
{full_code}
```

# Code block to improve
```python
{code_block}
```

# Improvement plan
{plan}

# Your task
- Implement the improvement plan by modifying the full Python solution above.
- Replace the identified code block with your improved version according to the plan.
- But do not remove subsampling if exists.
- Ensure the improved code integrates seamlessly with the rest of the solution.
- Make sure all necessary imports are included.

# Response format
- Your response should be a single markdown code block (wrapped in ```) which is the complete improved Python solution.
- There should be no additional headings or text in your response.
- Remember to print 'Final Validation Performance: {{final_validation_score}}' so we can parse performance.
- Do not use exit() function in the Python code."""
    
    def get_plan_refinement_prompt(self, code_block: str, previous_attempts_summary: str) -> str:
        """Get prompt for refining improvement plan - Optimized"""
        
        return f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you have to improve the code block for better performance.
- We will provide the code block you are improving and the improvement plans you have tried.

# Code block
```python
{code_block}
```

# Improvement plans you have tried

{previous_attempts_summary}

# Your task
- Suggest a better plan to improve the above code block.
- The suggested plan must be novel and effective.
- Please avoid plans which can make the solution's running time too long (e.g., searching hyperparameters in a very large search space).
- The suggested plan should be differ from the previous plans you have tried and should receive a higher score.

# Response format
- Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences).
- There should be no additional headings or text in your response."""