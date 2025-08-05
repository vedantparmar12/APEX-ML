"""Initialization Agent - Searches for and evaluates initial ML models"""

import os
import json
import shutil
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from utils.openrouter_client import llm_client
from utils.web_search import web_searcher
from utils.code_executor import code_executor
from config.config import CONFIG
from agents.initialization.prompts import InitializationPrompts


class InitializationAgent:
    """Agent responsible for finding and evaluating initial ML solutions"""
    
    def __init__(self, task_id: int = 1):
        self.task_id = task_id
        self.prompts = InitializationPrompts()
        self.workspace_dir = os.path.join(CONFIG.workspace_dir, CONFIG.task_name, str(task_id))
        self._setup_workspace()
    
    def _setup_workspace(self):
        """Create workspace directories"""
        os.makedirs(self.workspace_dir, exist_ok=True)
        os.makedirs(os.path.join(self.workspace_dir, "input"), exist_ok=True)
        os.makedirs(os.path.join(self.workspace_dir, "model_candidates"), exist_ok=True)
        
        # Copy task data to workspace
        task_dir = os.path.join(CONFIG.data_dir, CONFIG.task_name)
        if os.path.exists(task_dir):
            for file in os.listdir(task_dir):
                if not file.startswith("answer"):
                    src = os.path.join(task_dir, file)
                    dst = os.path.join(self.workspace_dir, "input", file)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
    
    def run(self, task_description: str) -> Dict[str, Any]:
        """Execute initialization pipeline"""
        
        results = {
            "task_id": self.task_id,
            "task_summary": "",
            "model_candidates": [],
            "evaluation_results": [],
            "best_solution": None,
            "best_score": None
        }
        
        try:
            # Step 1: Summarize task for search
            print(f"[Agent {self.task_id}] Summarizing task...")
            task_summary = self._summarize_task(task_description)
            results["task_summary"] = task_summary
            
            # Step 2: Search for model candidates
            print(f"[Agent {self.task_id}] Searching for model candidates...")
            model_candidates = self._search_models(task_summary)
            results["model_candidates"] = model_candidates
            
            # Step 3: Generate and evaluate solutions
            print(f"[Agent {self.task_id}] Evaluating {len(model_candidates)} model candidates...")
            evaluation_results = self._evaluate_models(model_candidates, task_description)
            results["evaluation_results"] = evaluation_results
            
            # Step 4: Merge best solutions
            print(f"[Agent {self.task_id}] Merging best solutions...")
            best_solution = self._merge_solutions(evaluation_results, task_description)
            results["best_solution"] = best_solution["code"]
            results["best_score"] = best_solution["score"]
            
            # Save results
            self._save_results(results)
            
        except Exception as e:
            print(f"[Agent {self.task_id}] Error: {str(e)}")
            traceback.print_exc()
        
        return results
    
    def _summarize_task(self, task_description: str) -> str:
        """Create concise task summary for search"""
        
        prompt = self.prompts.get_summarization_prompt(task_description, CONFIG.task_type)
        summary = llm_client.get_completion(prompt, temperature=0.3)
        
        # Save summary
        with open(os.path.join(self.workspace_dir, "task_summary.txt"), "w") as f:
            f.write(summary)
        
        return summary
    
    def _search_models(self, task_summary: str) -> List[Dict[str, Any]]:
        """Search for suitable ML models"""
        
        # Search for models
        search_results = web_searcher.search_ml_models(
            task_summary,
            CONFIG.task_type,
            num_results=CONFIG.max_search_results
        )
        
        # Get model candidates from LLM
        prompt = self.prompts.get_model_retrieval_prompt(
            task_summary,
            CONFIG.num_model_candidates
        )
        
        model_json = llm_client.get_structured_output(prompt, temperature=0.7)
        
        if isinstance(model_json, list):
            models = model_json
        else:
            models = model_json.get("models", [])
        
        # Enhance with search results
        for i, model in enumerate(models[:CONFIG.num_model_candidates]):
            if i < len(search_results):
                model["search_url"] = search_results[i].get("url", "")
                model["search_snippet"] = search_results[i].get("snippet", "")
            
            # Save model info
            model_file = os.path.join(self.workspace_dir, "model_candidates", f"model_{i+1}.json")
            with open(model_file, "w") as f:
                json.dump(model, f, indent=2)
        
        return models[:CONFIG.num_model_candidates]
    
    def _evaluate_models(self, model_candidates: List[Dict], task_description: str) -> List[Dict]:
        """Evaluate each model candidate"""
        
        evaluation_results = []
        
        with ThreadPoolExecutor(max_workers=min(len(model_candidates), 4)) as executor:
            futures = {}
            
            for i, model in enumerate(model_candidates):
                future = executor.submit(
                    self._evaluate_single_model,
                    model,
                    task_description,
                    i + 1
                )
                futures[future] = i
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    evaluation_results.append(result)
                except Exception as e:
                    print(f"[Agent {self.task_id}] Model {idx+1} evaluation failed: {str(e)}")
                    evaluation_results.append({
                        "model_id": idx + 1,
                        "success": False,
                        "error": str(e)
                    })
        
        return evaluation_results
    
    def _evaluate_single_model(self, model: Dict, task_description: str, model_id: int) -> Dict:
        """Evaluate a single model"""
        
        print(f"[Agent {self.task_id}] Evaluating model {model_id}: {model.get('model_name', 'Unknown')}")
        
        # Generate implementation code
        prompt = self.prompts.get_model_eval_prompt(
            task_description,
            model.get("model_name", ""),
            model.get("example_code", "")
        )
        
        code = llm_client.get_completion(prompt, temperature=0.3)
        
        # Clean code
        code = self._clean_code(code)
        
        # Save initial code
        code_file = os.path.join(self.workspace_dir, f"init_code_{model_id}.py")
        with open(code_file, "w") as f:
            f.write(code)
        
        # Execute and evaluate
        result = code_executor.execute_code(
            code,
            filename=f"model_{model_id}_eval.py",
            working_dir=self.workspace_dir
        )
        
        # If failed, try debugging
        if not result["success"] and CONFIG.max_debug_rounds > 0:
            print(f"[Agent {self.task_id}] Debugging model {model_id}...")
            code, result = self._debug_code(code, result["stderr"], task_description)
            
            # Save debugged code
            with open(code_file, "w") as f:
                f.write(code)
        
        return {
            "model_id": model_id,
            "model_name": model.get("model_name", ""),
            "code": code,
            "success": result["success"],
            "score": result.get("score"),
            "execution_time": result.get("execution_time"),
            "error": result.get("stderr", "") if not result["success"] else None
        }
    
    def _merge_solutions(self, evaluation_results: List[Dict], task_description: str) -> Dict:
        """Merge best performing solutions"""
        
        # Filter successful solutions
        successful = [r for r in evaluation_results if r["success"] and r.get("score") is not None]
        
        if not successful:
            # Return best attempt even if failed
            return {
                "code": evaluation_results[0]["code"] if evaluation_results else "",
                "score": None
            }
        
        # Sort by score
        successful.sort(key=lambda x: x["score"], reverse=not CONFIG.lower_is_better)
        
        if len(successful) == 1:
            return {
                "code": successful[0]["code"],
                "score": successful[0]["score"]
            }
        
        # Merge top solutions
        best_solution = successful[0]
        current_code = best_solution["code"]
        current_score = best_solution["score"]
        
        for i in range(1, min(len(successful), CONFIG.num_model_candidates)):
            reference_solution = successful[i]
            
            print(f"[Agent {self.task_id}] Merging solution {i+1} into base solution...")
            
            # Generate merged code
            prompt = self.prompts.get_code_integration_prompt(
                current_code,
                reference_solution["code"]
            )
            
            merged_code = llm_client.get_completion(prompt, temperature=0.3)
            merged_code = self._clean_code(merged_code)
            
            # Evaluate merged solution
            result = code_executor.execute_code(
                merged_code,
                filename=f"merged_{i}.py",
                working_dir=self.workspace_dir
            )
            
            if result["success"] and result.get("score") is not None:
                # Check if improved
                if CONFIG.lower_is_better:
                    improved = result["score"] < current_score
                else:
                    improved = result["score"] > current_score
                
                if improved:
                    current_code = merged_code
                    current_score = result["score"]
                    print(f"[Agent {self.task_id}] Merge improved score to {current_score}")
        
        # Save final solution
        final_file = os.path.join(self.workspace_dir, "train0.py")
        with open(final_file, "w") as f:
            f.write(current_code)
        
        return {
            "code": current_code,
            "score": current_score
        }
    
    def _debug_code(self, code: str, error: str, task_description: str) -> tuple:
        """Debug code with LLM assistance"""
        
        for attempt in range(CONFIG.max_debug_rounds):
            # Get bug summary
            prompt = self.prompts.get_bug_summary_prompt(error)
            bug_summary = llm_client.get_completion(prompt, temperature=0.1)
            
            # Get fix
            prompt = self.prompts.get_bug_fix_prompt(
                task_description,
                code,
                bug_summary
            )
            
            fixed_code = llm_client.get_completion(prompt, temperature=0.3)
            fixed_code = self._clean_code(fixed_code)
            
            # Test fix
            result = code_executor.execute_code(
                fixed_code,
                filename=f"debug_attempt_{attempt}.py",
                working_dir=self.workspace_dir
            )
            
            if result["success"]:
                return fixed_code, result
            
            code = fixed_code
            error = result["stderr"]
        
        return code, result
    
    def _clean_code(self, code: str) -> str:
        """Clean code block from markdown"""
        
        # Remove markdown code blocks
        code = code.replace("```python", "").replace("```", "")
        
        # Remove any leading/trailing whitespace
        code = code.strip()
        
        return code
    
    def _save_results(self, results: Dict):
        """Save agent results"""
        
        results_file = os.path.join(self.workspace_dir, "initialization_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)


def run_initialization_agents(task_description: str, num_agents: int = None) -> List[Dict]:
    """Run multiple initialization agents in parallel"""
    
    num_agents = num_agents or CONFIG.num_solutions
    
    agents = [InitializationAgent(task_id=i+1) for i in range(num_agents)]
    
    results = []
    with ThreadPoolExecutor(max_workers=num_agents) as executor:
        futures = [executor.submit(agent.run, task_description) for agent in agents]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Agent failed: {str(e)}")
    
    return results