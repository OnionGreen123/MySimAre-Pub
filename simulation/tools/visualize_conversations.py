import json
import html
from pathlib import Path
from typing import Any, Dict, List


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Conversation Viewer</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    .conversation {{ margin-bottom: 48px; }}
    .meta {{ color: #555; font-size: 13px; margin-bottom: 8px; }}
    .turn {{ margin: 10px 0; padding: 10px 12px; border-radius: 8px; }}
    .user {{ background: #f2f7ff; border: 1px solid #cfe1ff; }}
    .assistant {{ background: #f7f7f7; border: 1px solid #e0e0e0; }}
    .role {{ font-weight: bold; margin-bottom: 6px; }}
    .content {{ white-space: pre-wrap; }}
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
  <h1>Conversation Viewer</h1>
  {body}
</body>
</html>
"""


def render_conversation_block(item: Dict[str, Any], idx: int) -> str:
    problem = html.escape(item.get("problem", ""))
    problem_id = html.escape(str(item.get("problem_id", "")))
    meta = f'<div class="meta">Problem ID: {problem_id}</div>'
    header = f"<h2>Conversation {idx + 1}</h2><div class=\"meta\">{problem}</div>"

    turns_html: List[str] = []
    for turn in item.get("conversation", []):
        role = turn[0] if isinstance(turn, list) and len(turn) > 0 else "unknown"
        content = turn[1] if isinstance(turn, list) and len(turn) > 1 else ""
        role_class = "user" if role == "user" else "assistant"
        turn_html = (
            f'<div class="turn {role_class}">'
            f'<div class="role">{html.escape(role)}</div>'
            f'<div class="content">{html.escape(content)}</div>'
            f"</div>"
        )
        turns_html.append(turn_html)

    return f'<div class="conversation">{header}{meta}{"".join(turns_html)}</div>'


def visualize(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    blocks = []
    for idx, item in enumerate(data):
        blocks.append(render_conversation_block(item, idx))

    html_out = HTML_TEMPLATE.format(body="".join(blocks))
    output_path.write_text(html_out, encoding="utf-8")
    print(f"Wrote: {output_path}")


def main() -> None:
    input_path = Path(r"D:\MySimAre\output\competition_math\gpt-5-mini\dynamic-knowledge-state_20260212_224342.json")
    output_path = Path(r"D:\MySimAre\output\competition_math\gpt-5-mini\dynamic-knowledge-state_20260212_224342.html")
    visualize(input_path, output_path)


if __name__ == "__main__":
    main()

