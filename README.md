# RHI Robotics Research Loop

This project runs an iterative robotics ML research workflow using Google ADK
with Gemini models:

- Gemini 3.6 Flash generates the submission files.
- Gemini 3.8 Flash evaluates each iteration and improves the next iteration's harness.

The workflow runs three generation iterations by default. It makes several
Gemini API requests and uses your Gemini API quota.

## Requirements

- Windows 10 or later
- Python 3.12
- A Google AI Studio API key with access to the configured models
- `uv` (recommended) or Python's `venv` and `pip`

The Python dependencies are pinned in
[`rhi_robotics/requirements.txt`](rhi_robotics/requirements.txt). They include
Google ADK and `python-dotenv`. ADK uses its native Gemini model integration;
no separate LiteLLM or OpenAI package is required.

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

## Configure Gemini

Create `.env` from the example if it does not already exist:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

Set your own values in `.env`:

```dotenv
GOOGLE_API_KEY=your_google_ai_studio_api_key
GEMINI_CODE_MODEL=gemini-3.6-flash
GEMINI_REVIEW_MODEL=gemini-3.8-flash
```

Get an API key from Google AI Studio. Do not commit `.env` or share its
contents. `.env.example` contains placeholders and is safe to commit. Model
availability, free-tier eligibility, and rate limits can change; confirm current
access and quotas in AI Studio before running.

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

Alternatively, open `run_rhi.ipynb` in VS Code, select the repository's `.venv`
Python kernel, and run its cell. The notebook launches the same entry point from
the repository root and displays subprocess output and errors in the cell.

## Project layout

- `rhi_robotics\run_rhi_loop.py`: RHI iteration orchestration and model routing
- `rhi_robotics\adk_runtime.py`: ADK agent, runner, Gemini model, and session setup
- `rhi_robotics\agent_runner.py`: code-generation role and generated-file output
- `rhi_robotics\evaluator.py`: pairwise submission evaluation
- `rhi_robotics\harness_optimizer.py`: next-iteration harness improvement
- `rhi_robotics\config.py`: task and evaluation prompts
- `rhi_robotics\requirements.txt`: pinned Python dependencies
- `.env.example`: environment-variable template
