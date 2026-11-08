# RHI Robotics Research Loop

This project runs an iterative robotics ML research workflow using Google ADK
with Groq-hosted models:

- Qwen generates the submission files.
- GPT-OSS evaluates each iteration and improves the next iteration's harness.

The workflow runs three generation iterations by default. It calls Groq several
times and uses your Groq account's token quota.

## Requirements

- Windows 10 or later
- Python 3.12
- A Groq API key with access to the configured models
- `uv` (recommended) or Python's `venv` and `pip`

The Python dependencies are pinned in
[`rhi_robotics/requirements.txt`](rhi_robotics/requirements.txt). They include
Google ADK, its LiteLLM model connector, and `python-dotenv`. The program uses
Groq through LiteLLM; a Google Gemini API key is not required.

## Install on Windows

Open PowerShell in the repository folder:

```powershell
cd path\to\ADK-RHI
```

Create a Python 3.12 virtual environment and install the project dependencies
using `uv`:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .\.venv\Scripts\python.exe -r .\rhi_robotics\requirements.txt
```

If you prefer the standard library's `venv` and `pip`, run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\rhi_robotics\requirements.txt
```

## Configure Groq

Create `.env` from the example if it does not already exist:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

Set your own values in `.env`:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_CODE_MODEL=groq/qwen/qwen3.8-27b
GROQ_REVIEW_MODEL=groq/openai/gpt-oss-120b
```

Get the API key from the Groq Console. Do not commit `.env` or share its
contents. `.env.example` contains placeholders and is safe to commit. Confirm
that both model IDs are available to your Groq account; model availability and
rate limits can change.

## Run

From the repository folder, run:

```powershell
.\.venv\Scripts\python.exe -m rhi_robotics.run_rhi_loop
```

Generated runs are saved in `outputs_rhi_robotics\`, which is ignored by Git.
The loop can reuse workspaces marked as complete. If an earlier run stopped
before completion, the program will ask you to move or rename that incomplete
iteration workspace before retrying. Rename or back up `outputs_rhi_robotics`
before starting a completely fresh run if you need to preserve prior results.

## Project layout

- `rhi_robotics\run_rhi_loop.py`: RHI iteration orchestration and model routing
- `rhi_robotics\adk_runtime.py`: ADK agent, runner, session, and LiteLLM setup
- `rhi_robotics\agent_runner.py`: code-generation role and generated-file output
- `rhi_robotics\evaluator.py`: pairwise submission evaluation
- `rhi_robotics\harness_optimizer.py`: next-iteration harness improvement
- `rhi_robotics\config.py`: task and evaluation prompts
- `rhi_robotics\requirements.txt`: pinned Python dependencies
- `.env.example`: environment-variable template
