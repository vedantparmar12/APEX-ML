"""Main orchestrator for One-Above-All ML Engineering System"""

import os
import time
import json
import argparse
from typing import Optional
from datetime import datetime

from config.config import CONFIG
from agents.initialization.agent import run_initialization_agents
from agents.refinement.agent import run_refinement_agents
from agents.ensemble.agent import run_ensemble_agent
from agents.submission.agent import run_submission_agent


class OneAboveAll:
    """Main orchestrator for the ML engineering pipeline"""
    
    def __init__(self, task_name: Optional[str] = None):
        if task_name:
            CONFIG.task_name = task_name
        
        CONFIG.validate()
        
        self.task_dir = os.path.join(CONFIG.data_dir, CONFIG.task_name)
        self.results_dir = os.path.join(CONFIG.workspace_dir, CONFIG.task_name, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.start_time = time.time()
        self.results = {
            "task_name": CONFIG.task_name,
            "start_time": datetime.now().isoformat(),
            "config": self._get_config_dict(),
            "stages": {}
        }
    
    def run(self) -> dict:
        """Execute the complete ML engineering pipeline"""
        
        print("\n" + "="*60)
        print("ONE-ABOVE-ALL ML ENGINEERING SYSTEM")
        print("="*60)
        print(f"Task: {CONFIG.task_name}")
        print(f"Task Type: {CONFIG.task_type}")
        print(f"Model: {CONFIG.model_name}")
        print("="*60 + "\n")
        
        # Load task description
        task_description = self._load_task_description()
        
        if not task_description:
            print("ERROR: Task description not found!")
            return self.results
        
        try:
            # Stage 1: Initialization
            print("\n[STAGE 1] INITIALIZATION - Searching and evaluating models...")
            print("-" * 60)
            
            init_start = time.time()
            initialization_results = run_initialization_agents(
                task_description,
                num_agents=CONFIG.num_solutions
            )
            init_time = time.time() - init_start
            
            self.results["stages"]["initialization"] = {
                "duration": init_time,
                "num_agents": len(initialization_results),
                "successful_agents": sum(1 for r in initialization_results if r.get("best_solution")),
                "best_scores": [r.get("best_score") for r in initialization_results if r.get("best_score") is not None]
            }
            
            print(f"\n[STAGE 1] Completed in {init_time:.1f}s")
            print(f"Successful solutions: {self.results['stages']['initialization']['successful_agents']}")
            
            # Stage 2: Refinement
            print("\n[STAGE 2] REFINEMENT - Running ablation studies and improvements...")
            print("-" * 60)
            
            refine_start = time.time()
            refinement_results = run_refinement_agents(initialization_results)
            refine_time = time.time() - refine_start
            
            self.results["stages"]["refinement"] = {
                "duration": refine_time,
                "num_refined": len(refinement_results),
                "final_scores": [r.get("final_score") for r in refinement_results if r.get("final_score") is not None]
            }
            
            print(f"\n[STAGE 2] Completed in {refine_time:.1f}s")
            
            # Stage 3: Ensemble
            print("\n[STAGE 3] ENSEMBLE - Creating ensemble solutions...")
            print("-" * 60)
            
            ensemble_start = time.time()
            ensemble_result = run_ensemble_agent(refinement_results)
            ensemble_time = time.time() - ensemble_start
            
            self.results["stages"]["ensemble"] = {
                "duration": ensemble_time,
                "success": ensemble_result.get("best_ensemble_score") is not None,
                "ensemble_score": ensemble_result.get("best_ensemble_score"),
                "improved_over_individual": ensemble_result.get("ensemble_improved", False)
            }
            
            print(f"\n[STAGE 3] Completed in {ensemble_time:.1f}s")
            if ensemble_result.get("ensemble_improved"):
                print(f"Ensemble improved by {ensemble_result.get('improvement_percentage', 0):.2f}%")
            
            # Stage 4: Submission
            print("\n[STAGE 4] SUBMISSION - Creating final submission file...")
            print("-" * 60)
            
            submission_start = time.time()
            submission_result = run_submission_agent(
                task_description,
                refinement_results,
                ensemble_result
            )
            submission_time = time.time() - submission_start
            
            self.results["stages"]["submission"] = {
                "duration": submission_time,
                "success": submission_result.get("success", False),
                "best_score": submission_result.get("best_score"),
                "source": submission_result.get("source"),
                "submission_file": submission_result.get("submission_file")
            }
            
            print(f"\n[STAGE 4] Completed in {submission_time:.1f}s")
            
            # Summary
            total_time = time.time() - self.start_time
            self.results["total_duration"] = total_time
            self.results["end_time"] = datetime.now().isoformat()
            
            print("\n" + "="*60)
            print("PIPELINE COMPLETED")
            print("="*60)
            print(f"Total Duration: {total_time:.1f}s")
            print(f"Best Score: {submission_result.get('best_score')}")
            print(f"Best Solution: {submission_result.get('source')}")
            
            if submission_result.get("submission_file"):
                print(f"Submission File: {submission_result.get('submission_file')}")
            
            print("="*60 + "\n")
            
            # Save results
            self._save_results()
            
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            self.results["error"] = str(e)
            self.results["traceback"] = traceback.format_exc()
            self._save_results()
        
        return self.results
    
    def _load_task_description(self) -> Optional[str]:
        """Load task description from file"""
        
        task_file = os.path.join(self.task_dir, "task_description.txt")
        
        if not os.path.exists(task_file):
            # Try alternative names
            alternatives = ["description.txt", "task.txt", "README.txt"]
            for alt in alternatives:
                alt_file = os.path.join(self.task_dir, alt)
                if os.path.exists(alt_file):
                    task_file = alt_file
                    break
        
        if os.path.exists(task_file):
            with open(task_file, 'r') as f:
                return f.read()
        
        return None
    
    def _get_config_dict(self) -> dict:
        """Get configuration as dictionary"""
        
        return {
            "model_name": CONFIG.model_name,
            "task_type": CONFIG.task_type,
            "num_solutions": CONFIG.num_solutions,
            "num_model_candidates": CONFIG.num_model_candidates,
            "outer_loop_rounds": CONFIG.outer_loop_rounds,
            "inner_loop_rounds": CONFIG.inner_loop_rounds,
            "ensemble_loop_rounds": CONFIG.ensemble_loop_rounds,
            "lower_is_better": CONFIG.lower_is_better
        }
    
    def _save_results(self):
        """Save pipeline results"""
        
        results_file = os.path.join(self.results_dir, "pipeline_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="One-Above-All ML Engineering System"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=CONFIG.task_name,
        help="Task name (must match folder in tasks directory)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=CONFIG.model_name,
        help="Model to use via OpenRouter"
    )
    parser.add_argument(
        "--num-solutions",
        type=int,
        default=CONFIG.num_solutions,
        help="Number of parallel solutions to generate"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Update configuration
    if args.model:
        CONFIG.model_name = args.model
    if args.num_solutions:
        CONFIG.num_solutions = args.num_solutions
    if args.api_key:
        CONFIG.openrouter_api_key = args.api_key
        os.environ["OPENROUTER_API_KEY"] = args.api_key
    
    # Run pipeline
    system = OneAboveAll(task_name=args.task)
    system.run()


if __name__ == "__main__":
    main()