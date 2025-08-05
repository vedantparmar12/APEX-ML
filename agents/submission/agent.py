"""Submission Agent - Prepares final submission file"""

import os
import json
from typing import Dict, List, Any, Optional

from utils.openrouter_client import llm_client
from utils.code_executor import code_executor
from config.config import CONFIG
from agents.submission.prompts import SubmissionPrompts


class SubmissionAgent:
    """Agent responsible for creating the final submission"""
    
    def __init__(self):
        self.prompts = SubmissionPrompts()
        self.workspace_dir = os.path.join(CONFIG.workspace_dir, CONFIG.task_name, "submission")
        os.makedirs(self.workspace_dir, exist_ok=True)
    
    def run(
        self,
        task_description: str,
        refined_results: List[Dict],
        ensemble_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create final submission"""
        
        print("[Submission] Selecting best solution...")
        
        # Select best solution
        best_solution = self._select_best_solution(refined_results, ensemble_result)
        
        if not best_solution:
            return {
                "success": False,
                "message": "No valid solution found"
            }
        
        print(f"[Submission] Best solution selected with score: {best_solution['score']}")
        
        # Add submission code
        print("[Submission] Adding submission generation code...")
        final_code = self._add_submission_code(
            task_description,
            best_solution["code"]
        )
        
        # Execute and verify
        print("[Submission] Generating submission file...")
        result = self._execute_submission_code(final_code)
        
        submission_result = {
            "success": result["success"],
            "best_score": best_solution["score"],
            "source": best_solution["source"],
            "final_code": final_code if result["success"] else None,
            "submission_file": None,
            "error": result.get("stderr") if not result["success"] else None
        }
        
        if result["success"]:
            # Check for submission file
            submission_files = [
                f for f in result.get("generated_files", [])
                if "submission" in f.lower() or "predictions" in f.lower()
            ]
            
            if submission_files:
                submission_result["submission_file"] = submission_files[0]
                print(f"[Submission] Submission file created: {submission_files[0]}")
            else:
                # Look for common submission file names
                common_names = ["submission.csv", "predictions.csv", "output.csv"]
                for name in common_names:
                    filepath = os.path.join(self.workspace_dir, name)
                    if os.path.exists(filepath):
                        submission_result["submission_file"] = filepath
                        break
        
        # Save results
        self._save_results(submission_result)
        
        return submission_result
    
    def _select_best_solution(
        self,
        refined_results: List[Dict],
        ensemble_result: Optional[Dict]
    ) -> Optional[Dict]:
        """Select the best solution from all available options"""
        
        candidates = []
        
        # Add refined solutions
        for result in refined_results:
            if result.get("final_code") and result.get("final_score") is not None:
                candidates.append({
                    "code": result["final_code"],
                    "score": result["final_score"],
                    "source": f"refined_agent_{result['task_id']}"
                })
        
        # Add ensemble solution if available
        if ensemble_result and ensemble_result.get("best_ensemble_code"):
            if ensemble_result.get("best_ensemble_score") is not None:
                candidates.append({
                    "code": ensemble_result["best_ensemble_code"],
                    "score": ensemble_result["best_ensemble_score"],
                    "source": "ensemble"
                })
        
        if not candidates:
            return None
        
        # Sort by score
        candidates.sort(
            key=lambda x: x["score"],
            reverse=not CONFIG.lower_is_better
        )
        
        return candidates[0]
    
    def _add_submission_code(self, task_description: str, code: str) -> str:
        """Add submission generation code to the solution"""
        
        prompt = self.prompts.get_submission_prompt(task_description, code)
        
        final_code = llm_client.get_completion(prompt, temperature=0.3)
        final_code = self._clean_code(final_code)
        
        return final_code
    
    def _execute_submission_code(self, code: str) -> Dict[str, Any]:
        """Execute the submission code"""
        
        # Save code
        code_file = os.path.join(self.workspace_dir, "final_solution.py")
        with open(code_file, "w") as f:
            f.write(code)
        
        # Execute
        result = code_executor.execute_code(
            code,
            filename="submission_generation.py",
            working_dir=self.workspace_dir
        )
        
        # Try debugging if failed
        if not result["success"] and CONFIG.max_debug_rounds > 0:
            print("[Submission] Debugging submission code...")
            code, result = code_executor.debug_code(
                code,
                result["stderr"],
                CONFIG.max_debug_rounds
            )
            
            # Save debugged code
            with open(code_file, "w") as f:
                f.write(code)
        
        return result
    
    def _clean_code(self, code: str) -> str:
        """Clean code from markdown blocks"""
        
        code = code.replace("```python", "").replace("```", "")
        return code.strip()
    
    def _save_results(self, results: Dict):
        """Save submission results"""
        
        # Save final code if successful
        if results.get("final_code"):
            final_file = os.path.join(self.workspace_dir, "final_solution.py")
            with open(final_file, "w") as f:
                f.write(results["final_code"])
        
        # Save results summary (without code to keep file small)
        save_results = results.copy()
        save_results.pop("final_code", None)
        
        results_file = os.path.join(self.workspace_dir, "submission_results.json")
        with open(results_file, "w") as f:
            json.dump(save_results, f, indent=2)


def run_submission_agent(
    task_description: str,
    refined_results: List[Dict],
    ensemble_result: Optional[Dict] = None
) -> Dict[str, Any]:
    """Run submission agent"""
    
    agent = SubmissionAgent()
    return agent.run(task_description, refined_results, ensemble_result)