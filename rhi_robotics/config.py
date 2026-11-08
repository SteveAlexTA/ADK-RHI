# rhi_robotics/config.py

ROBOTICS_TASK_PROMPT = """
# Task: Robotics ML Research
Create a reproducible fine-tuning experiment for a small vision-language model on robot-relevant instruction classification (e.g., action type, object category, spatial relation). 
Use a curated public dataset (e.g. EPIC-KITCHENS captions aligned with actions or robot instruction subsets). 
Evaluate accuracy, calibration (ECE), and robustness to paraphrased instructions.

## Required Deliverables:
1. `research_report.md`: Conference-style paper (6-9 pages equivalent) describing methodology, architecture, fine-tuning, calibration, and limitations.
2. `deliverables/index.json`: Manifest recording all paths of generated files and status.
3. `deliverables/plots/`:
   - `accuracy_by_class.png`
   - `confusion_matrix.png`
   - `calibration_reliability_diagram.png`
   - `paraphrase_robustness_drop.png`
   - `learning_curves.png`
4. Reproducible Code:
   - Data preparation & augmentation
   - Fine-tuning pipeline (frozen, full fine-tuned, LoRA)
   - Evaluation script & calibration utilities (temperature scaling / isotonic)
   - `requirements.txt`
5. `results/metrics.json` containing: Top-1 accuracy, Macro F1, Expected Calibration Error (ECE), Paraphrase robustness drop, Per-class metrics.
6. `results/model_comparison.json` comparing frozen, fine-tuned, and partially fine-tuned (LoRA) variants with hyperparameters.
"""

ROBOTICS_EVAL_PROMPT = """
You are a senior robotics ML researcher reviewing two independent submissions for the Robotics Instruction Classification task.
Evaluate rigorously based on 6 criteria:
1. Deliverable coverage: Whether all 6 required deliverables and 5 PNG plots are present and valid.
2. Empirical & Numerical rigor: Validity of VLM fine-tuning, calibration analysis (ECE), and paraphrase robustness evaluation.
3. Reproducibility: Clean scripts, seeds, environment config, and clear entrypoints.
4. Presentation: Completeness and depth of research_report.md.
5. Engineering quality: Modular code, correct imports, proper error handling.
6. Internal consistency: Consistency between numbers in research_report.md, metrics.json, and model_comparison.json.
"""