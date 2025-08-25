import os
from typing import Dict, Any, Optional, List


class Step:
    name: str = "base_step"
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.artifacts: Dict[str, Any] = {}
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class Pipeline:
    def __init__(self, steps: List[Step], run_dir: str):
        self.steps = steps
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)
    def run(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if context is None:
            context = {}
        for step in self.steps:
            step_dir = os.path.join(self.run_dir, step.name)
            os.makedirs(step_dir, exist_ok=True)
            context["_current_step_dir"] = step_dir
            context = step.run(context)
        context.pop("_current_step_dir", None)
        return context