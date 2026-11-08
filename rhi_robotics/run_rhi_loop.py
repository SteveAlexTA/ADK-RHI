import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rhi_robotics.config import ROBOTICS_TASK_PROMPT, ROBOTICS_EVAL_PROMPT
from rhi_robotics.harness_v0 import INITIAL_ROBOTICS_HARNESS
from rhi_robotics.agent_runner import RHIExecutionAgent
from rhi_robotics.evaluator import RHIEvaluator
from rhi_robotics.harness_optimizer import RHIHarnessOptimizer

def main():
    base_dir = Path("outputs_rhi_robotics")
    base_dir.mkdir(exist_ok=True)
    
    code_model = os.getenv("GROQ_CODE_MODEL", "groq/qwen/qwen3.8-27b")
    review_model = os.getenv("GROQ_REVIEW_MODEL", "groq/openai/gpt-oss-120b")

    executor = RHIExecutionAgent(model_name=code_model)
    evaluator = RHIEvaluator(model_name=review_model)
    optimizer = RHIHarnessOptimizer(model_name=review_model)

    current_harness = INITIAL_ROBOTICS_HARNESS
    history_file = base_dir / "history.json"
    history = []
    if history_file.exists():
        history = json.loads(history_file.read_text(encoding="utf-8"))
        if not isinstance(history, list):
            raise ValueError(f"History file must contain a JSON list: {history_file}")
        print(f"[*] Loaded {len(history)} previous comparison records.")

    num_iterations = 2
    previous_repo = None

    for i in range(num_iterations + 1):
        iter_dir = base_dir / f"iteration_{i}"
        ws_dir = iter_dir / "workspace"
        completion_path = ws_dir / "run_complete.txt"
        harness_file = iter_dir / f"harness_{i}.txt"
        
        print(f"\n==========================================")
        print(f"[*] BẮT ĐẦU RHI ITERATION {i}")
        print(f"==========================================")
        
        iter_dir.mkdir(exist_ok=True)
        if harness_file.exists():
            current_harness = harness_file.read_text(encoding="utf-8")
        else:
            harness_file.write_text(current_harness, encoding="utf-8")

        if completion_path.exists() and (ws_dir / "logs.txt").exists():
            print(f"[*] Reusing completed workspace for iteration {i}: {ws_dir}")
            current_repo = ws_dir
        elif (ws_dir / "logs.txt").exists():
            raise RuntimeError(
                f"Iteration {i} has an old incomplete workspace: {ws_dir}. "
                "Move or rename that workspace before restarting so its files are preserved."
            )
        else:
            current_repo = executor.run_iteration(
                task_prompt=ROBOTICS_TASK_PROMPT,
                harness_prompt=current_harness,
                output_dir=ws_dir,
            )

        time.sleep(2)

        comparison_key = f"iteration_{i-1} vs iteration_{i}"
        already_evaluated = any(h.get("comparison") == comparison_key for h in history)
        
        if previous_repo is not None and not already_evaluated:
            print(f"[*] Comparing iteration {i-1} (A) with iteration {i} (B)...")
            eval_result = evaluator.compare(
                task_prompt=ROBOTICS_TASK_PROMPT,
                eval_criteria=ROBOTICS_EVAL_PROMPT,
                repo_a=previous_repo,
                repo_b=current_repo,
            )
            print(f"[+] Winner: {eval_result.get('winner')}")
            print(f"[+] Rationale: {eval_result.get('rationale')}")

            history.append({
                "comparison": comparison_key,
                "winner": eval_result.get("winner"),
                "rationale": eval_result.get("rationale"),
                "defects": eval_result.get("recurring_defect_tags", []),
                "takeaways": eval_result.get("actionable_takeaways", []),
            })
            
            history_file.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
            time.sleep(2)

        if i < num_iterations:
            next_harness_file = base_dir / f"iteration_{i+1}" / f"harness_{i+1}.txt"
            if next_harness_file.exists():
                print(f"[*] Reusing existing harness H[{i+1}].")
                current_harness = next_harness_file.read_text(encoding="utf-8")
            else:
                print(f"[*] Optimizing harness for iteration {i+1}...")
                current_harness = optimizer.optimize_harness(
                    task_prompt=ROBOTICS_TASK_PROMPT,
                    current_harness=current_harness,
                    current_repo=current_repo,
                    history_summary=history,
                )
                (base_dir / f"iteration_{i+1}").mkdir(exist_ok=True)
                next_harness_file.write_text(current_harness, encoding="utf-8")
                print(f"[+] Generated harness H[{i+1}].")

        previous_repo = current_repo

    print("\n[SUCCESS] RHI optimization loop completed.")

if __name__ == "__main__":
    main()
