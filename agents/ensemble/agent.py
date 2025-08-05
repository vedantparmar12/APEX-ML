"""Ensemble Agent - Combines multiple solutions for better performance"""

import os
import json
from typing import Dict, List, Any, Optional
import numpy as np

from utils.openrouter_client import llm_client
from utils.code_executor import code_executor
from config.config import CONFIG
from agents.ensemble.prompts import EnsemblePrompts


class EnsembleAgent:
    """Agent responsible for creating ensemble solutions"""
    
    def __init__(self):
        self.prompts = EnsemblePrompts()
        self.workspace_dir = os.path.join(CONFIG.workspace_dir, CONFIG.task_name, "ensemble")
        os.makedirs(self.workspace_dir, exist_ok=True)
        self.ensemble_plans = []
    
    def run(self, refined_solutions: List[Dict]) -> Dict[str, Any]:
        """Execute ensemble pipeline"""
        
        # Filter valid solutions
        valid_solutions = [
            sol for sol in refined_solutions
            if sol.get("final_code") and sol.get("final_score") is not None
        ]
        
        if len(valid_solutions) < 2:
            print("[Ensemble] Not enough valid solutions for ensemble")
            return {
                "success": False,
                "message": "Insufficient solutions for ensemble"
            }
        
        results = {
            "num_solutions": len(valid_solutions),
            "input_scores": [sol["final_score"] for sol in valid_solutions],
            "ensemble_iterations": [],
            "best_ensemble_code": None,
            "best_ensemble_score": None
        }
        
        # Prepare solution codes
        solution_codes = [sol["final_code"] for sol in valid_solutions]
        
        # Initial ensemble plan
        print("[Ensemble] Generating initial ensemble plan...")
        initial_plan = self._generate_initial_plan(solution_codes)
        self.ensemble_plans.append(initial_plan)
        
        # Implement and evaluate initial ensemble
        ensemble_result = self._implement_and_evaluate_plan(
            solution_codes,
            initial_plan,
            iteration=0
        )
        
        results["ensemble_iterations"].append(ensemble_result)
        
        best_code = ensemble_result.get("code")
        best_score = ensemble_result.get("score")
        
        # Iterative refinement of ensemble
        for iteration in range(1, CONFIG.ensemble_loop_rounds + 1):
            print(f"\n[Ensemble] Iteration {iteration}/{CONFIG.ensemble_loop_rounds}")
            
            # Generate refined plan based on previous results
            refined_plan = self._generate_refined_plan(
                solution_codes,
                self.ensemble_plans,
                results["ensemble_iterations"]
            )
            
            if not refined_plan:
                print("[Ensemble] No new plan generated")
                break
            
            self.ensemble_plans.append(refined_plan)
            
            # Implement and evaluate
            ensemble_result = self._implement_and_evaluate_plan(
                solution_codes,
                refined_plan,
                iteration=iteration
            )
            
            results["ensemble_iterations"].append(ensemble_result)
            
            # Update best if improved
            if ensemble_result["success"] and ensemble_result.get("score") is not None:
                if best_score is None:
                    best_code = ensemble_result["code"]
                    best_score = ensemble_result["score"]
                else:
                    if CONFIG.lower_is_better:
                        improved = ensemble_result["score"] < best_score
                    else:
                        improved = ensemble_result["score"] > best_score
                    
                    if improved:
                        best_code = ensemble_result["code"]
                        best_score = ensemble_result["score"]
                        print(f"[Ensemble] Improved score to {best_score}")
        
        results["best_ensemble_code"] = best_code
        results["best_ensemble_score"] = best_score
        
        # Compare with individual solutions
        if best_score is not None:
            individual_best = min(results["input_scores"]) if CONFIG.lower_is_better else max(results["input_scores"])
            
            if CONFIG.lower_is_better:
                ensemble_improved = best_score < individual_best
            else:
                ensemble_improved = best_score > individual_best
            
            results["ensemble_improved"] = ensemble_improved
            results["improvement_percentage"] = abs((best_score - individual_best) / individual_best * 100)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _generate_initial_plan(self, solution_codes: List[str]) -> str:
        """Generate initial ensemble plan"""
        
        prompt = self.prompts.get_initial_ensemble_prompt(solution_codes)
        plan = llm_client.get_completion(prompt, temperature=0.7)
        
        return plan
    
    def _generate_refined_plan(
        self,
        solution_codes: List[str],
        previous_plans: List[str],
        previous_results: List[Dict]
    ) -> Optional[str]:
        """Generate refined ensemble plan"""
        
        # Prepare summary of previous attempts
        plans_and_scores = []
        for plan, result in zip(previous_plans, previous_results):
            if result["success"] and result.get("score") is not None:
                plans_and_scores.append({
                    "plan": plan,
                    "score": result["score"]
                })
        
        if not plans_and_scores:
            return None
        
        prompt = self.prompts.get_refined_ensemble_prompt(
            solution_codes,
            plans_and_scores
        )
        
        refined_plan = llm_client.get_completion(prompt, temperature=0.8)
        
        # Check if plan is substantially different
        if any(refined_plan.lower() in plan.lower() for plan in previous_plans):
            return None
        
        return refined_plan
    
    def _implement_and_evaluate_plan(
        self,
        solution_codes: List[str],
        plan: str,
        iteration: int
    ) -> Dict[str, Any]:
        """Implement ensemble plan and evaluate"""
        
        print(f"[Ensemble] Implementing plan: {plan[:100]}...")
        
        # Generate ensemble code
        prompt = self.prompts.get_implement_ensemble_prompt(
            solution_codes,
            plan
        )
        
        ensemble_code = llm_client.get_completion(prompt, temperature=0.3)
        ensemble_code = self._clean_code(ensemble_code)
        
        # Save ensemble code
        code_file = os.path.join(self.workspace_dir, f"ensemble_{iteration}.py")
        with open(code_file, "w") as f:
            f.write(ensemble_code)
        
        # Execute and evaluate
        result = code_executor.execute_code(
            ensemble_code,
            filename=f"ensemble_run_{iteration}.py",
            working_dir=self.workspace_dir
        )
        
        # Try debugging if failed
        if not result["success"] and CONFIG.max_debug_rounds > 0:
            print(f"[Ensemble] Debugging ensemble {iteration}...")
            ensemble_code, result = code_executor.debug_code(
                ensemble_code,
                result["stderr"],
                CONFIG.max_debug_rounds
            )
            
            # Save debugged code
            with open(code_file, "w") as f:
                f.write(ensemble_code)
        
        return {
            "iteration": iteration,
            "plan": plan,
            "code": ensemble_code,
            "success": result["success"],
            "score": result.get("score"),
            "execution_time": result.get("execution_time"),
            "error": result.get("stderr") if not result["success"] else None
        }
    
    def _clean_code(self, code: str) -> str:
        """Clean code from markdown blocks"""
        
        code = code.replace("```python", "").replace("```", "")
        return code.strip()
    
    def _save_results(self, results: Dict):
        """Save ensemble results"""
        
        # Save best ensemble code
        if results.get("best_ensemble_code"):
            best_file = os.path.join(self.workspace_dir, "best_ensemble.py")
            with open(best_file, "w") as f:
                f.write(results["best_ensemble_code"])
        
        # Save results summary
        results_file = os.path.join(self.workspace_dir, "ensemble_results.json")
        
        # Remove code from results to keep file size manageable
        save_results = results.copy()
        save_results.pop("best_ensemble_code", None)
        for iteration in save_results.get("ensemble_iterations", []):
            iteration.pop("code", None)
        
        with open(results_file, "w") as f:
            json.dump(save_results, f, indent=2)


def run_ensemble_agent(refined_results: List[Dict]) -> Dict[str, Any]:
    """Run ensemble agent on refined solutions"""
    
    agent = EnsembleAgent()
    return agent.run(refined_results)