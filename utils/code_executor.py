"""Code execution utilities with safety and monitoring"""

import subprocess
import os
import sys
import tempfile
import time
import traceback
import json
import re
from typing import Dict, Any, Optional, Tuple, List
from config.config import CONFIG


class CodeExecutor:
    """Safe code execution with monitoring and result extraction"""
    
    def __init__(self, workspace_dir: Optional[str] = None):
        self.workspace_dir = workspace_dir or CONFIG.workspace_dir
        os.makedirs(self.workspace_dir, exist_ok=True)
    
    def execute_code(
        self,
        code: str,
        filename: str = "temp_code.py",
        timeout: Optional[int] = None,
        working_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute Python code and return results"""
        
        timeout = timeout or CONFIG.exec_timeout
        working_dir = working_dir or self.workspace_dir
        
        # Create temporary file
        filepath = os.path.join(working_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # Execute code
        start_time = time.time()
        
        try:
            # Run with subprocess
            env = os.environ.copy()
            env['PYTHONPATH'] = working_dir
            
            result = subprocess.run(
                [sys.executable, filepath],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir,
                env=env
            )
            
            execution_time = time.time() - start_time
            
            # Parse output for metrics
            output = result.stdout
            error = result.stderr
            
            # Extract validation score
            score = self._extract_score(output)
            
            # Extract any generated files
            generated_files = self._check_generated_files(working_dir)
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": output,
                "stderr": error,
                "execution_time": execution_time,
                "score": score,
                "generated_files": generated_files,
                "filepath": filepath
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout} seconds",
                "execution_time": timeout,
                "score": None,
                "generated_files": [],
                "filepath": filepath
            }
        except Exception as e:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}\n{traceback.format_exc()}",
                "execution_time": time.time() - start_time,
                "score": None,
                "generated_files": [],
                "filepath": filepath
            }
    
    def _extract_score(self, output: str) -> Optional[float]:
        """Extract validation score from output"""
        
        # Look for pattern: "Final Validation Performance: <number>"
        pattern = r"Final Validation Performance:\s*([+-]?\d*\.?\d+)"
        match = re.search(pattern, output, re.IGNORECASE)
        
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        # Alternative patterns
        patterns = [
            r"validation score:\s*([+-]?\d*\.?\d+)",
            r"val_score:\s*([+-]?\d*\.?\d+)",
            r"rmse:\s*([+-]?\d*\.?\d+)",
            r"mae:\s*([+-]?\d*\.?\d+)",
            r"accuracy:\s*([+-]?\d*\.?\d+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
        
        return None
    
    def _check_generated_files(self, working_dir: str) -> List[str]:
        """Check for newly generated files"""
        
        generated = []
        
        # Common output files
        common_outputs = ["submission.csv", "predictions.csv", "output.csv"]
        
        for filename in common_outputs:
            filepath = os.path.join(working_dir, filename)
            if os.path.exists(filepath):
                generated.append(filepath)
        
        return generated
    
    def run_ablation_study(
        self,
        base_code: str,
        ablation_code: str,
        working_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run ablation study and extract results"""
        
        working_dir = working_dir or self.workspace_dir
        
        # Execute ablation code
        result = self.execute_code(
            ablation_code,
            filename="ablation_study.py",
            working_dir=working_dir
        )
        
        if result["success"]:
            # Parse ablation results
            ablation_results = self._parse_ablation_output(result["stdout"])
            result["ablation_results"] = ablation_results
        
        return result
    
    def _parse_ablation_output(self, output: str) -> Dict[str, float]:
        """Parse ablation study output"""
        
        results = {}
        
        # Look for ablation results pattern
        lines = output.split('\n')
        
        for line in lines:
            # Pattern: "Ablation <name>: <score>"
            match = re.search(r"Ablation[:\s]+(.+?):\s*([+-]?\d*\.?\d+)", line)
            if match:
                name = match.group(1).strip()
                score = float(match.group(2))
                results[name] = score
        
        return results
    
    def debug_code(
        self,
        code: str,
        error: str,
        max_attempts: int = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Attempt to debug code based on error"""
        
        max_attempts = max_attempts or CONFIG.max_debug_rounds
        
        for attempt in range(max_attempts):
            # Analyze error and suggest fix
            if "ModuleNotFoundError" in error:
                # Add missing imports
                missing_module = re.search(r"No module named '(\w+)'", error)
                if missing_module:
                    module = missing_module.group(1)
                    code = self._add_import(code, module)
            
            elif "NameError" in error:
                # Fix undefined names
                undefined = re.search(r"name '(\w+)' is not defined", error)
                if undefined:
                    name = undefined.group(1)
                    code = self._fix_undefined_name(code, name)
            
            elif "FileNotFoundError" in error:
                # Fix file paths
                code = self._fix_file_paths(code)
            
            # Re-execute
            result = self.execute_code(code, f"debug_attempt_{attempt}.py")
            
            if result["success"]:
                return code, result
            
            error = result["stderr"]
        
        return code, result
    
    def _add_import(self, code: str, module: str) -> str:
        """Add missing import to code"""
        
        # Common module mappings
        import_map = {
            "sklearn": "from sklearn",
            "xgboost": "import xgboost as xgb",
            "lightgbm": "import lightgbm as lgb",
            "catboost": "from catboost import CatBoostRegressor, CatBoostClassifier",
            "torch": "import torch",
            "optuna": "import optuna"
        }
        
        if module in import_map:
            import_line = import_map[module]
            if import_line not in code:
                # Add import at the beginning
                lines = code.split('\n')
                # Find first non-import line
                insert_pos = 0
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith(('import', 'from')):
                        insert_pos = i
                        break
                
                lines.insert(insert_pos, import_line)
                code = '\n'.join(lines)
        
        return code
    
    def _fix_undefined_name(self, code: str, name: str) -> str:
        """Fix undefined name in code"""
        
        # Common fixes
        fixes = {
            "train_test_split": "from sklearn.model_selection import train_test_split",
            "mean_squared_error": "from sklearn.metrics import mean_squared_error",
            "StandardScaler": "from sklearn.preprocessing import StandardScaler",
            "np": "import numpy as np",
            "pd": "import pandas as pd"
        }
        
        if name in fixes:
            import_line = fixes[name]
            if import_line not in code:
                code = import_line + "\n" + code
        
        return code
    
    def _fix_file_paths(self, code: str) -> str:
        """Fix file paths in code"""
        
        # Replace relative paths with absolute paths
        code = code.replace("'train.csv'", "'./input/train.csv'")
        code = code.replace('"train.csv"', '"./input/train.csv"')
        code = code.replace("'test.csv'", "'./input/test.csv'")
        code = code.replace('"test.csv"', '"./input/test.csv"')
        
        return code


# Global executor instance
code_executor = CodeExecutor()