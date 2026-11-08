import json
from pathlib import Path

from pydantic import BaseModel

from rhi_robotics.adk_runtime import run_agent
from rhi_robotics.utils import parse_json_from_response


class HarnessOptimizationResponse(BaseModel):
    improved_multi_agent_design: str
    changes_from_previous: list[str]
    why_it_should_improve: list[str]
    evidence_used: list[str]
    verification_checks: list[str]


class RHIHarnessOptimizer:
    def __init__(self, model_name: str = "gemini-3.8-flash"):
        self.model_name = model_name

    def optimize_harness(
        self,
        task_prompt: str,
        current_harness: str,
        current_repo: Path,
        history_summary: list,
    ) -> str:
        system_prompt = """
        You are a principal prompt engineer for autonomous coding agents.
        Your task is to improve only the 'multi agent design' block of the query prompt to improve the quality of deliverables.
        Prioritize harnesses that multi-agent systems can do well, such as:
        1. Stronger agent-to-agent communication contracts (exact schema for outputs passed to orchestrator).
        2. More orchestrator-subagent hops with explicit iterative recall / re-delegation instead of one-pass execution.
        3. Auxiliary acceptance gates and verification sentinels.

        You must ground changes in concrete evidence from submission artifacts and pairwise comparison history.
        Return ONLY valid JSON matching this schema:
        {
          "improved_multi_agent_design": "<full replacement text starting with 'Create an agent team with the following agent candidates to solve this problem:'>",
          "changes_from_previous": ["<change 1>", "..."],
          "why_it_should_improve": ["<rationale 1>", "..."],
          "evidence_used": ["<evidence 1>", "..."],
          "verification_checks": ["<check 1>", "..."]
        }
        """

        evidence_files = [str(p.relative_to(current_repo)) for p in current_repo.rglob("*") if p.is_file()]

        user_prompt = f"""
        # Task
        {task_prompt}

        # Current Multi-Agent Design
        {current_harness}

        # Current Submission Evidence
        Files generated in current workspace: {evidence_files}

        # Pairwise History Summary
        {json.dumps(history_summary, indent=2)}

        # Instructions
        Improve the harness design to fix defects identified in the history summary while preserving strengths.
        Explicitly strengthen output schemas and add targeted recall hops.
        """

        response_text = run_agent(
            agent_name="rhi_harness_optimizer",
            model_name=self.model_name,
            instruction=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            output_schema=HarnessOptimizationResponse,
        )
        result = parse_json_from_response(response_text)
        return result["improved_multi_agent_design"]
