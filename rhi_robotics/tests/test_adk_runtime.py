import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from google.genai import types

from rhi_robotics import adk_runtime
from rhi_robotics.agent_runner import RHIExecutionAgent
from rhi_robotics.evaluator import EvaluationResponse, RHIEvaluator
from rhi_robotics.harness_optimizer import HarnessOptimizationResponse, RHIHarnessOptimizer


class FakeSessionService:
    async def create_session(self, *, app_name: str, user_id: str) -> SimpleNamespace:
        return SimpleNamespace(id="mock-session")


class FakeRunner:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def run_async(self, **kwargs: object):
        async def events():
            yield SimpleNamespace(is_final_response=lambda: False, content=None)
            yield SimpleNamespace(
                is_final_response=lambda: True,
                content=types.Content(
                    parts=[
                        types.Part(text="Mocked "),
                        types.Part(text="ADK response"),
                    ]
                ),
            )

        return events()


class AdkRuntimeTests(unittest.TestCase):
    def test_timeout_and_server_errors_are_retryable(self) -> None:
        class TemporaryServerError(Exception):
            status_code = 503

        self.assertTrue(adk_runtime._is_retryable_provider_error(TimeoutError()))
        self.assertTrue(
            adk_runtime._is_retryable_provider_error(TemporaryServerError())
        )
        self.assertFalse(adk_runtime._is_retryable_provider_error(ValueError()))

    def test_run_agent_uses_adk_runner_and_structured_output(self) -> None:
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "mock-key"}),
            patch.object(adk_runtime, "InMemorySessionService", FakeSessionService),
            patch.object(adk_runtime, "LlmAgent") as llm_agent,
            patch.object(adk_runtime, "Runner", FakeRunner),
        ):
            response = adk_runtime.run_agent(
                agent_name="rhi_evaluator",
                model_name="gemini-3.8-flash",
                instruction="Return JSON.",
                user_prompt="Compare two results.",
                temperature=0.1,
                output_schema=EvaluationResponse,
            )
        self.assertEqual(response, "Mocked ADK response")
        self.assertEqual(response, "Mocked ADK response")
        agent_arguments = llm_agent.call_args.kwargs
        self.assertEqual(agent_arguments["name"], "rhi_evaluator")
        self.assertEqual(agent_arguments["model"], "gemini-3.8-flash")
        self.assertIs(agent_arguments["output_schema"], EvaluationResponse)

    def test_run_agent_requires_google_key_without_calling_runner(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(adk_runtime, "Runner") as runner,
            self.assertRaisesRegex(RuntimeError, "GOOGLE_API_KEY"),
        ):
            adk_runtime.run_agent(
                agent_name="rhi_code_executor",
                model_name="gemini-3.6-flash",
                instruction="Generate files.",
                user_prompt="Build the project.",
                temperature=0.3,
            )

        runner.assert_not_called()

    def test_components_default_to_separate_gemini_models(self) -> None:
        self.assertEqual(
            RHIExecutionAgent().model_name,
            "gemini-3.6-flash",
        )
        self.assertEqual(
            RHIEvaluator().model_name,
            "gemini-3.8-flash",
        )
        self.assertEqual(
            HarnessOptimizationResponse.model_fields.keys(),
            {
                "improved_multi_agent_design",
                "changes_from_previous",
                "why_it_should_improve",
                "evidence_used",
                "verification_checks",
            },
        )
        self.assertEqual(
            RHIHarnessOptimizer().model_name,
            "gemini-3.8-flash",
        )

    def test_evaluator_and_optimizer_use_their_structured_output_schemas(self) -> None:
        evaluation = (
            '{"winner":"A","rationale":"Evidence","actionable_takeaways":[],'
            '"recurring_defect_tags":[]}'
        )
        optimization = (
            '{"improved_multi_agent_design":"Updated design",'
            '"changes_from_previous":[],"why_it_should_improve":[],'
            '"evidence_used":[],"verification_checks":[]}'
        )
        with (
            patch.object(Path, "rglob", return_value=[]),
            patch.object(Path, "exists", return_value=False),
            patch.object(RHIEvaluator, "_build_repo_summary", return_value="submission"),
            patch("rhi_robotics.evaluator.run_agent", return_value=evaluation) as evaluate,
            patch(
                "rhi_robotics.harness_optimizer.run_agent",
                return_value=optimization,
            ) as optimize,
        ):
            result = RHIEvaluator().compare(
                task_prompt="task",
                eval_criteria="criteria",
                repo_a=Path("candidate_a"),
                repo_b=Path("candidate_b"),
            )
            improved_design = RHIHarnessOptimizer().optimize_harness(
                task_prompt="task",
                current_harness="current",
                current_repo=Path("candidate"),
                history_summary=[],
            )

        self.assertEqual(result["winner"], "A")
        self.assertEqual(improved_design, "Updated design")
        self.assertEqual(evaluate.call_args.kwargs["model_name"], "gemini-3.8-flash")
        self.assertIs(evaluate.call_args.kwargs["output_schema"], EvaluationResponse)
        self.assertEqual(optimize.call_args.kwargs["model_name"], "gemini-3.8-flash")
        self.assertIs(
            optimize.call_args.kwargs["output_schema"],
            HarnessOptimizationResponse,
        )

    def test_completed_workspace_is_not_regenerated(self) -> None:
        executor = RHIExecutionAgent()
        with (
            patch.object(Path, "mkdir"),
            patch.object(Path, "exists", return_value=True),
            patch("rhi_robotics.agent_runner.run_agent") as run_agent,
            self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"),
        ):
            executor.run_iteration("task", "harness", Path("existing_workspace"))

        run_agent.assert_not_called()

    def test_run_iteration_extracts_files_and_writes_resume_log(self) -> None:
        response = "FILE: result.txt\n```text\nresult content\n```"
        opened_files = []

        def open_exclusive(path: Path, mode: str, encoding: str):
            output_file = MagicMock()
            opened_files.append((path, mode, encoding, output_file))
            return output_file

        with (
            patch.object(Path, "mkdir"),
            patch.object(Path, "exists", return_value=False),
            patch.object(Path, "open", open_exclusive),
            patch("rhi_robotics.agent_runner.run_agent", return_value=response),
        ):
            result = RHIExecutionAgent().run_iteration(
                task_prompt="task",
                harness_prompt="harness",
                output_dir=Path("workspace"),
            )

        self.assertEqual(result, Path("workspace"))
        self.assertEqual(len(opened_files), 3)
        self.assertTrue(opened_files[0][0].name == "result.txt")
        self.assertTrue(opened_files[1][0].name == "logs.txt")
        self.assertTrue(opened_files[2][0].name == "run_complete.txt")
        self.assertEqual([item[1:3] for item in opened_files], [("x", "utf-8")] * 3)
        self.assertEqual(
            [item[3].__enter__.return_value.write.call_args.args[0] for item in opened_files],
            ["result content", response, "result.txt"],
        )

    def test_generated_path_cannot_escape_output_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "escapes output directory"):
            RHIExecutionAgent._resolve_output_path("../outside.txt", Path("workspace"))

    def test_generated_absolute_windows_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be relative"):
            RHIExecutionAgent._resolve_output_path(
                "C:/outside.txt",
                Path("workspace"),
            )

    def test_extraction_preserves_existing_files(self) -> None:
        output_file = MagicMock()
        output_file.__enter__.side_effect = FileExistsError
        with (
            patch.object(Path, "mkdir"),
            patch.object(Path, "open", return_value=output_file) as open_file,
        ):
            RHIExecutionAgent()._extract_and_write_files(
                "FILE: result.txt\n```text\nnew content\n```",
                Path("workspace"),
            )

        open_file.assert_called_once_with("x", encoding="utf-8")
        output_file.write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
