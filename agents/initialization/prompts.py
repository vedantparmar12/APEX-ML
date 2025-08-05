"""Prompts for the Initialization Agent - Hybrid approach with exact critical prompts"""


class InitializationPrompts:
    """Collection of prompts for initialization phase"""
    
    def get_summarization_prompt(self, task_description: str, task_type: str) -> str:
        """Get prompt for task summarization - EXACT from ADK"""
        
        return f"""# Task description
{task_description}

# Your task
- Summarize this task description.
- Your summary will be used for searching recent effective models for {task_type}.

# Requirement
- We will directly use your response, so be simple and concise."""

    def get_model_retrieval_prompt(self, task_summary: str, num_candidates: int) -> str:
        """Get prompt for model retrieval - EXACT from ADK"""
        
        return f"""# Competition
{task_summary}

# Your task
- List {num_candidates} recent effective models and their example codes to win the above competition.

# Requirement
- The example code should be concise and simple.
- You must provide an example code, i.e., do not just mention GitHubs or papers.

Use this JSON schema:
Model = {{"model_name": str, "example_code": str}}
Return: list[Model]"""

    def get_model_eval_prompt(self, task_description: str, model_name: str, example_code: str) -> str:
        """Get prompt for model evaluation - EXACT critical parts from ADK"""
        
        return f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- We will now provide a task description and a model description.
- You need to implement your Python solution using the provided model.

# Task description
{task_description}

# Model description
## Model name
{model_name}

## Example Python code
{example_code}

# Your task
- Implement the solution in Python.
- You must use the model as described in the model description.
- This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.
- Propose an evaluation metric that is reasonable for this task.
- All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.
- Do not include other models that are not directly related to the model described.
- Use PyTorch rather than TensorFlow. Use CUDA if you need. All the necessary libraries are installed.
- The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.
- Only use the provided train data in the `./input` directory.

# Required
- There should be no additional headings or text in your response.
- Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {{final_validation_score}}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the Python code.
- Do not use try: and except: or if else to ignore unintended behavior."""

    def get_code_integration_prompt(self, base_code: str, reference_code: str) -> str:
        """Get prompt for code integration - EXACT from ADK"""
        
        return f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- We will now provide a base solution and an additional reference solution.
- You need to implement your Python solution by integrating reference solution to the base solution.

# Base solution
```python
{base_code}
```

# Reference solution
```python
{reference_code}
```

# Your task
- Implement the solution in Python.
- You have to integrate the reference solution to the base solution.
- Your code base should be the base solution.
- Try to train additional model of the reference solution.
- When integrating, try to keep code with similar functionality in the same place (e.g., all preprocessing should be done and then all training).
- When integrating, ensemble the models.
- The solution design should be relatively simple.
- The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.
- Only use the provided train data in the `./input` directory.

# Required
- There should be no additional headings or text in your response.
- Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {{final_validation_score}}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the Python code.
- Do not use try: and except: or if else to ignore unintended behavior."""

    def get_bug_summary_prompt(self, error: str) -> str:
        """Get prompt for bug summarization - Optimized"""
        
        return f"""# Error report
{error}

# Your task
- Remove all unnecessary parts of the above error report.
- We are now running the Python file. Do not remove where the error occurred."""

    def get_bug_fix_prompt(self, task_description: str, code: str, bug: str) -> str:
        """Get prompt for bug fixing - EXACT from ADK"""
        
        return f"""# Task description
{task_description}

# Code with an error:
{code}

# Error:
{bug}

# Your task
- Please revise the code to fix the error.
- Do not remove subsampling if exists.
- Provide the improved, self-contained Python script again.
- There should be no additional headings or text in your response.
- All the provided input data is stored in "./input" directory.
- Remember to print a line in the code with 'Final Validation Performance: {{final_validation_score}}' so we can parse performance.
- The code should be a single-file python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the refined Python code."""

    def get_data_usage_check_prompt(self, code: str, task_description: str) -> str:
        """Get prompt for checking data usage"""
        
        return f"""I have provided Python code for a machine learning task (attached below):
# Solution Code
```python
{code}
```

# Task description
{task_description}

# Your task
If the above solution code does not use the information provided, try to incorporate all. Do not bypass using try-except.
DO NOT USE TRY and EXCEPT; just occur error so we can debug it!
See the task description carefully, to know how to extract unused information effectively.
When improving the solution code by incorporating unused information, DO NOT FORGET to print out 'Final Validation Performance: {{final_validation_score}}' as in original solution code.

Response format:
Option 1: If the code did not use all the provided information, your response should be a single markdown code block (wrapped in ```) which is the improved code block. There should be no additional headings or text in your response.
Option 2: If the code used all the provided information, simply state that "All the provided information is used."
"""