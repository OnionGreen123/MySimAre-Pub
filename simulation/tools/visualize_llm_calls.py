import json
import html
from pathlib import Path
from typing import Dict


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>LLM Calls Viewer</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    .call {{ border: 1px solid #ddd; padding: 12px; margin-bottom: 16px; border-radius: 8px; }}
    .meta {{ color: #555; font-size: 13px; margin-bottom: 8px; }}
    details {{ margin: 8px 0; }}
    .content {{ white-space: pre-wrap; background: #f7f7f7; padding: 8px; border-radius: 6px; }}
    .label {{ font-weight: bold; margin-top: 8px; }}
  </style>
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
      }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body>
  <h1>LLM Calls Viewer</h1>
  {body}
</body>
</html>
"""


def _normalize_latex(text: str) -> str:
    # Convert escaped backslashes (from JSON logs) into single backslashes for MathJax.
    return text.replace("\\\\", "\\")


def render_call(entry: Dict, idx: int) -> str:
    meta = (
        f"<div class='meta'>#{idx + 1} | {html.escape(entry.get('timestamp', ''))} "
        f"| model: {html.escape(entry.get('model_name', ''))} "
        f"| temp: {entry.get('temperature')} | max_tokens: {entry.get('max_tokens')} | n: {entry.get('n')}</div>"
    )
    system_prompt = html.escape(_normalize_latex(entry.get("system_prompt", "")))
    user_prompt = html.escape(_normalize_latex(entry.get("user_prompt", "")))
    output_raw = entry.get("output", "")
    if isinstance(output_raw, list):
        output_raw = "\n---\n".join([str(item) for item in output_raw])
    else:
        output_raw = str(output_raw)
    output = html.escape(_normalize_latex(output_raw))

    return (
        "<div class='call'>"
        f"{meta}"
        "<details><summary>System Prompt</summary>"
        f"<div class=\"content\">{system_prompt}</div></details>"
        "<details open><summary>User Prompt</summary>"
        f"<div class=\"content\">{user_prompt}</div></details>"
        "<details><summary>Output</summary>"
        f"<div class=\"content\">{output}</div></details>"
        "</div>"
    )


def visualize(input_path: Path, output_path: Path) -> None:
    entries = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    body = "".join(render_call(entry, idx) for idx, entry in enumerate(entries))
    html_out = HTML_TEMPLATE.format(body=body)
    output_path.write_text(html_out, encoding="utf-8")
    print(f"Wrote: {output_path}")


def main() -> None:
    input_path = Path(r"D:\MySimAre\logs\llm_calls_20260212_224251.jsonl")
    output_path = Path(r"D:\MySimAre\logs\llm_calls_20260212_224251.html")
    visualize(input_path, output_path)


if __name__ == "__main__":
    main()

