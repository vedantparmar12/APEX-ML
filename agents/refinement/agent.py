"""Refinement Agent - Performs ablation studies and targeted improvements"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.openrouter_client import llm_client
from utils.code_executor import code_executor
from config.config import CONFIG
from agents.refinement.prompts import RefinementPrompts


class RefinementAgent:
    """Agent responsible for refining ML solutions through ablation and targeted improvements"""
    
    def __init__(self, task_id: int = 1):
        self.task_id = task_id
        self.prompts = RefinementPrompts()
        self.workspace_dir = os.path.join(CONFIG.workspace_dir, CONFIG.task_name, str(task_id))
        self.refinement_history = []
        self.ablation_history = []
    
    def run(self, initial_code: str, initial_score: float) -> Dict[str, Any]:
        """Execute refinement pipeline"""
        
        results = {
            "task_id": self.task_id,
            "initial_score": initial_score,
            "refinement_steps": [],
            "final_code": initial_code,
            "final_score": initial_score
        }
        
        current_code = initial_code
        current_score = initial_score
        
        # Outer loop: Different code blocks to refine
        for step in range(CONFIG.outer_loop_rounds):
            print(f"\n[Refinement {self.task_id}] Starting refinement step {step + 1}/{CONFIG.outer_loop_rounds}")
            
            # Step 1: Run ablation study
            ablation_results = self._run_ablation_study(current_code, step)
            
            # Step 2: Extract code block and plan
            code_block, improvement_plan = self._extract_block_and_plan(
                current_code,
                ablation_results,
                step
            )
            
            if not code_block or not improvement_plan:
                print(f"[Refinement {self.task_id}] No improvement opportunities found")
                break
            
            # Step 3: Inner loop - try different improvement strategies
            best_improvement = self._run_inner_loop(
                current_code,
                current_score,
                code_block,
                improvement_plan,
                step
            )
            
            # Update if improved
            if best_improvement["improved"]:
                current_code = best_improvement["code"]
                current_score = best_improvement["score"]
                print(f"[Refinement {self.task_id}] Step {step + 1} improved score to {current_score}")
            else:
                print(f"[Refinement {self.task_id}] Step {step + 1} no improvement found")
            
            # Record step results
            results["refinement_steps"].append({
                "step": step + 1,
                "ablation_results": ablation_results,
                "code_block": code_block,
                "improvement_plan": improvement_plan,
                "improved": best_improvement["improved"],
                "new_score": current_score if best_improvement["improved"] else None
            })
        
        results["final_code"] = current_code
        results["final_score"] = current_score
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _run_ablation_study(self, code: str, step: int) -> Dict[str, Any]:
        """Run ablation study to identify important code components"""
        
        print(f"[Refinement {self.task_id}] Running ablation study...")
        
        # Generate ablation code
        prompt = self.prompts.get_ablation_prompt(code, self.ablation_history)
        ablation_code = llm_client.get_completion(prompt, temperature=0.5)
        ablation_code = self._clean_code(ablation_code)
        
        # Save ablation code
        ablation_file = os.path.join(self.workspace_dir, f"ablation_{step}.py")
        with open(ablation_file, "w") as f:
            f.write(ablation_code)
        
        # Execute ablation study
        result = code_executor.run_ablation_study(
            code,
            ablation_code,
            self.workspace_dir
        )
        
        if not result["success"]:
            print(f"[Refinement {self.task_id}] Ablation study failed: {result['stderr']}")
            return {}
        
        # Summarize ablation results
        prompt = self.prompts.get_ablation_summary_prompt(
            ablation_code,
            result["stdout"]
        )
        
        summary = llm_client.get_completion(prompt, temperature=0.3)
        
        ablation_results = {
            "code": ablation_code,
            "output": result["stdout"],
            "summary": summary,
            "ablation_scores": result.get("ablation_results", {})
        }
        
        self.ablation_history.append(summary)
        
        return ablation_results
    
    def _extract_block_and_plan(
        self,
        code: str,
        ablation_results: Dict,
        step: int
    ) -> Tuple[str, str]:
        """Extract code block to improve and generate improvement plan"""
        
        if not ablation_results:
            return None, None
        
        prompt = self.prompts.get_extract_block_prompt(
            code,
            ablation_results["summary"],
            self.refinement_history
        )
        
        response = llm_client.get_structured_output(prompt, temperature=0.5)
        
        if isinstance(response, list) and len(response) > 0:
            plan_data = response[0]
        else:
            plan_data = response
        
        code_block = plan_data.get("code_block", "")
        plan = plan_data.get("plan", "")
        
        return code_block, plan
    
    def _run_inner_loop(
        self,
        base_code: str,
        base_score: float,
        code_block: str,
        initial_plan: str,
        step: int
    ) -> Dict[str, Any]:
        """Try multiple improvement strategies for the code block"""
        
        improvements = []
        plans_tried = [initial_plan]
        
        # Try initial plan
        improved_code = self._implement_plan(base_code, code_block, initial_plan)
        result = self._evaluate_code(improved_code, f"train{step}_improve0.py")
        
        improvements.append({
            "plan": initial_plan,
            "code": improved_code,
            "score": result.get("score"),
            "success": result["success"]
        })
        
        # Try additional plans
        for inner_iter in range(1, CONFIG.inner_loop_rounds):
            # Generate new plan based on previous results
            new_plan = self._generate_refined_plan(
                code_block,
                plans_tried,
                improvements
            )
            
            if not new_plan:
                break
            
            plans_tried.append(new_plan)
            
            # Implement and evaluate
            improved_code = self._implement_plan(base_code, code_block, new_plan)
            result = self._evaluate_code(improved_code, f"train{step}_improve{inner_iter}.py")
            
            improvements.append({
                "plan": new_plan,
                "code": improved_code,
                "score": result.get("score"),
                "success": result["success"]
            })
        
        # Find best improvement
        valid_improvements = [
            imp for imp in improvements
            if imp["success"] and imp["score"] is not None
        ]
        
        if not valid_improvements:
            return {"improved": False}
        
        # Sort by score
        valid_improvements.sort(
            key=lambda x: x["score"],
            reverse=not CONFIG.lower_is_better
        )
        
        best = valid_improvements[0]
        
        # Check if actually improved
        if CONFIG.lower_is_better:
            improved = best["score"] < base_score
        else:
            improved = best["score"] > base_score
        
        if improved:
            self.refinement_history.append({
                "code_block": code_block,
                "plan": best["plan"]
            })
            
            return {
                "improved": True,
                "code": best["code"],
                "score": best["score"],
                "plan": best["plan"]
            }
        
        return {"improved": False}
    
    def _implement_plan(self, base_code: str, code_block: str, plan: str) -> str:
        """Implement improvement plan"""
        
        prompt = self.prompts.get_implement_plan_prompt(
            base_code,
            code_block,
            plan
        )
        
        improved_code = llm_client.get_completion(prompt, temperature=0.3)
        return self._clean_code(improved_code)
    
    def _generate_refined_plan(
        self,
        code_block: str,
        plans_tried: List[str],
        improvements: List[Dict]
    ) -> Optional[str]:
        """Generate a refined improvement plan"""
        
        # Prepare summary of previous attempts
        plan_summary = self._summarize_plan_results(plans_tried, improvements)
        
        prompt = self.prompts.get_plan_refinement_prompt(
            code_block,
            plan_summary
        )
        
        new_plan = llm_client.get_completion(prompt, temperature=0.7)
        
        # Check if plan is substantially different
        if any(new_plan.lower() in tried.lower() for tried in plans_tried):
            return None
        
        return new_plan
    
    def _summarize_plan_results(
        self,
        plans: List[str],
        improvements: List[Dict]
    ) -> str:
        """Summarize results of previous improvement attempts"""
        
        summary = ""
        
        for i, (plan, imp) in enumerate(zip(plans, improvements)):
            if imp["success"] and imp["score"] is not None:
                summary += f"Plan {i+1}: {plan}\n"
                summary += f"Score improvement: {imp['score']:.5f}\n\n"
            else:
                summary += f"Plan {i+1}: {plan}\n"
                summary += f"Failed to execute\n\n"
        
        return summary
    
    def _evaluate_code(self, code: str, filename: str) -> Dict[str, Any]:
        """Evaluate code and handle debugging if needed"""
        
        result = code_executor.execute_code(
            code,
            filename=filename,
            working_dir=self.workspace_dir
        )
        
        # Try debugging if failed
        if not result["success"] and CONFIG.max_debug_rounds > 0:
            code, result = code_executor.debug_code(
                code,
                result["stderr"],
                CONFIG.max_debug_rounds
            )
        
        return result
    
    def _clean_code(self, code: str) -> str:
        """Clean code from markdown blocks"""
        
        code = code.replace("```python", "").replace("```", "")
        return code.strip()
    
    def _save_results(self, results: Dict):
        """Save refinement results"""
        
        # Save final refined code
        final_file = os.path.join(
            self.workspace_dir,
            f"train{CONFIG.outer_loop_rounds}.py"
        )
        with open(final_file, "w") as f:
            f.write(results["final_code"])
        
        # Save results summary
        results_file = os.path.join(self.workspace_dir, "refinement_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)


def run_refinement_agents(initialization_results: List[Dict]) -> List[Dict]:
    """Run refinement for each initialization result"""
    
    refinement_results = []
    
    for init_result in initialization_results:
        if init_result["best_solution"] and init_result["best_score"] is not None:
            agent = RefinementAgent(task_id=init_result["task_id"])
            
            result = agent.run(
                init_result["best_solution"],
                init_result["best_score"]
            )
            
            refinement_results.append(result)
    
    return refinement_results