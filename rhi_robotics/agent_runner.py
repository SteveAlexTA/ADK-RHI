import logging
import re
from pathlib import Path, PureWindowsPath

from rhi_robotics.adk_runtime import run_agent

logger = logging.getLogger(__name__)


class RHIExecutionAgent:
    def __init__(self, model_name: str = "gemini-3.6-flash"):
        self.model_name = model_name

    def run_iteration(self, task_prompt: str, harness_prompt: str, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_path = output_dir / "logs.txt"
        completion_path = output_dir / "run_complete.txt"
        if logs_path.exists():
            raise FileExistsError(f"Refusing to overwrite the completed run log: {logs_path}")

        system_instruction = f"""
        You are an autonomous ML research team operating under the following multi-agent harness specification:
        {harness_prompt}
        
        Your mission is to produce a complete, production-grade, reproducible code repository and deliverables.
        You have no shell, browser, or other tools. Never invent tool calls, environment checks, or claim to
        have executed code. Return only complete file blocks in this exact format:
        FILE: <relative_path>
        ```<language>
        <file_content>
        ```
        Do not include an introduction, plan, tool-call syntax, or text outside file blocks. Ensure every
        required deliverable is represented by a separate file block. If the response cannot fit all files,
        prioritize executable code and include a concise report and manifest rather than emitting plans.
        """

        logger.info("Running code executor with model %s", self.model_name)
        response_text = run_agent(
            agent_name="rhi_code_executor",
            model_name=self.model_name,
            instruction=system_instruction,
            user_prompt=task_prompt,
            temperature=0.3,
        )
        written_files = self._extract_and_write_files(response_text, output_dir)
        if not written_files:
            raise RuntimeError(
                "The code model returned no usable file blocks. The response was not marked complete; "
                "inspect the provider output and retry in a fresh iteration directory."
            )
        self._write_new_file(logs_path, response_text)
        self._write_new_file(completion_path, "\n".join(written_files))
        return output_dir

    def _extract_and_write_files(self, text: str, base_dir: Path) -> list[str]:
        file_pattern = re.compile(
            r"FILE:\s*([^\n\r]+)\s*```[a-zA-Z0-9_-]*\s*\n(.*?)```",
            re.DOTALL,
        )
        matches = file_pattern.findall(text)

        if not matches:
            alt_pattern = re.compile(
                r"(?:### File:|File:|###)\s*[`'\"]?([^`'\"\n\r]+)[`'\"]?"
                r"\s*```[a-zA-Z0-9_-]*\s*\n(.*?)```",
                re.DOTALL,
            )
            matches = alt_pattern.findall(text)

        written_files = []
        for rel_path, content in matches:
            clean_path = rel_path.strip().replace("\\", "/").strip("`'\" :")
            if clean_path.lower() in ["python", "json", "markdown", "bash"]:
                continue
            full_path = self._resolve_output_path(clean_path, base_dir)
            if full_path == base_dir.resolve() / "logs.txt":
                raise ValueError("Generated files cannot replace the RHI resume log.")
            full_path.parent.mkdir(parents=True, exist_ok=True)
            if self._write_new_file(full_path, content.strip()):
                logger.info("Generated %s", clean_path)
                written_files.append(clean_path)
        return written_files

    @staticmethod
    def _resolve_output_path(relative_path: str, base_dir: Path) -> Path:
        windows_path = PureWindowsPath(relative_path)
        requested_path = Path(relative_path)
        if requested_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
            raise ValueError(f"Generated file path must be relative: {relative_path}")

        resolved_base = base_dir.resolve()
        resolved_path = (resolved_base / requested_path).resolve()
        try:
            resolved_path.relative_to(resolved_base)
        except ValueError as error:
            raise ValueError(
                f"Generated file path escapes output directory: {relative_path}"
            ) from error
        return resolved_path

    @staticmethod
    def _write_new_file(path: Path, content: str) -> bool:
        try:
            with path.open("x", encoding="utf-8") as output_file:
                output_file.write(content)
        except FileExistsError:
            logger.warning("Skipped existing file to preserve user data: %s", path)
            return False
        return True
