"""Prompts for the Submission Agent"""


class SubmissionPrompts:
    """Collection of prompts for submission phase"""
    
    def get_submission_prompt(self, task_description: str, code: str) -> str:
        """Get prompt for adding submission code"""
        
        return f"""# Task description
{task_description}

# Current solution code
```python
{code}
```

# Your task
- Add code to the above solution to create a submission file for the competition.
- The submission file should be saved as 'submission.csv' in the current directory.
- Make sure the submission format matches exactly what is specified in the task description.
- The code should:
  1. Load the test data from './input/test.csv'
  2. Apply the same preprocessing as used for training
  3. Make predictions using the trained model(s)
  4. Create a submission file with the correct format
  5. Save it as 'submission.csv'

# Important requirements
- Do not modify the existing training code, only add the submission generation part.
- Ensure all necessary imports are included.
- Handle any potential errors gracefully.
- The submission file must have the exact column names and format as specified.
- Print a confirmation message when the submission file is created successfully.

# Response format
- Your response should be a single markdown code block (wrapped in ```) containing the complete code.
- The code should include both the original solution and the added submission generation code.
- There should be no additional headings or text in your response.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- At the end, print: "Submission file saved successfully to submission.csv" """