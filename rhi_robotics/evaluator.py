import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from rhi_robotics.adk_runtime import run_agent
from rhi_robotics.utils import parse_json_from_response


class EvaluationResponse(BaseModel):
    winner: Literal["A", "B", "tie"]
    rationale: str
    actionable_takeaways: list[str]
    recurring_defect_tags: list[str]


class RHIEvaluator:
    def __init__(self, model_name: str = "gemini-3.8-flash"):
        self.model_name = model_name

    def _build_repo_summary(self, repo_dir: Path) -> str:
        artifact_paths = sorted(
            path
            for path in repo_dir.rglob("*")
            if path.is_file()
            and not path.name.startswith(".")
            and path.name not in {"logs.txt", "run_complete.txt"}
        )
        if not artifact_paths:
            raise ValueError(f"Cannot evaluate an empty submission workspace: {repo_dir}")

        summary = [f"Workspace Path: {repo_dir.name}\n## Generated Files:"]
        for path in artifact_paths[:80]:
            summary.append(f" - {path.relative_to(repo_dir)} ({path.stat().st_size} bytes)")

        summary.append("\n## Key Artifact Excerpts:")
        for key_file in [
            "research_report.md",
            "deliverables/index.json",
            "results/metrics.json",
            "results/model_comparison.json",
        ]:
            kf_path = repo_dir / key_file
            if kf_path.exists():
                content = kf_path.read_text(encoding="utf-8", errors="ignore")
                summary.append(f"### File: '{key_file}'\n{content[:1200]}\n")
            else:
                summary.append(f"### File: '{key_file}' [MISSING]")
        return "\n".join(summary)

    def compare(self, task_prompt: str, eval_criteria: str, repo_a: Path, repo_b: Path) -> dict:
        summary_a = self._build_repo_summary(repo_a)
        summary_b = self._build_repo_summary(repo_b)

        system_prompt = f"""
        {eval_criteria}
        
        Rules:
        - The # Task block is the only contract for what must be produced.
        - Prefer concrete citations (paths, JSON keys, discrepancies).
        - Choose A or B when one submission clearly better satisfies the task across deliverables, rigor, reproducibility, and alignment.
        - Return ONLY a valid JSON object with the schema:
          {{
             "winner": "A" | "B" | "tie",
             "rationale": "<string citing concrete evidence>",
             "actionable_takeaways": ["<point 1>", "<point 2>"],
             "recurring_defect_tags": ["report-metrics inconsistency", "reproducibility gap", "calibration error missing", etc.]
          }}
        """

        user_prompt = f"""
        # Task
        {task_prompt}

        # Submission A (Candidate from earlier iteration)
        {summary_a}

        # Submission B (Candidate from current iteration)
        {summary_b}
        """

        response_text = run_agent(
            agent_name="rhi_evaluator",
            model_name=self.model_name,
            instruction=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            output_schema=EvaluationResponse,
        )
        return parse_json_from_response(response_text)
