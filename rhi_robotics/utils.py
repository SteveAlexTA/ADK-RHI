import json
import re
from typing import Any


def parse_json_from_response(text: str) -> dict[str, Any]:
    """Parse a JSON response, including JSON wrapped in a Markdown code block."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\})", raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise
