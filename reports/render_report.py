"""
Convert final_report.md into a publication-ready HTML whitepaper.
"""
from pathlib import Path
from markdown_it import MarkdownIt

def render_report():
    report_dir = Path(__file__).resolve().parent
    md_path = report_dir / "final_report.md"
    html_path = report_dir / "final_report.html"

    if not md_path.exists():
        print(f"Error: {md_path} not found")
        return

    content = md_path.read_text(encoding="utf-8")

    md = MarkdownIt("commonmark").enable("table").enable("strikethrough")
    rendered_body = md.render(content)

    html_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Engineering & Evaluation Report: @AppleSupport AI Agent</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
      --color-primary: #0071e3;
      --color-text: #0f172a;
      --color-muted: #475569;
      --color-border: #e2e8f0;
      --color-bg: #ffffff;
      --color-code-bg: #f1f5f9;
    }}
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      font-family: var(--font-sans);
      color: var(--color-text);
      background-color: #f8fafc;
      line-height: 1.65;
      padding: 40px 20px;
    }}
    .report-container {{
      max-width: 900px;
      margin: 0 auto;
      background: var(--color-bg);
      padding: 56px 64px;
      border-radius: 16px;
      border: 1px solid var(--color-border);
      box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.06);
    }}
    h1 {{
      font-size: 2rem;
      font-weight: 800;
      letter-spacing: -0.025em;
      color: #0f172a;
      line-height: 1.25;
      margin-bottom: 12px;
      border-bottom: 2px solid var(--color-primary);
      padding-bottom: 16px;
    }}
    h2 {{
      font-size: 1.35rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: #1e293b;
      margin-top: 36px;
      margin-bottom: 14px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--color-border);
    }}
    h3 {{
      font-size: 1.1rem;
      font-weight: 600;
      color: #334155;
      margin-top: 24px;
      margin-bottom: 10px;
    }}
    p {{
      font-size: 0.95rem;
      color: #334155;
      margin-bottom: 16px;
    }}
    strong {{
      color: #0f172a;
      font-weight: 600;
    }}
    ul, ol {{
      margin-bottom: 18px;
      padding-left: 24px;
    }}
    li {{
      font-size: 0.93rem;
      color: #334155;
      margin-bottom: 8px;
      line-height: 1.55;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 24px 0;
      font-size: 0.85rem;
    }}
    th, td {{
      padding: 10px 14px;
      text-align: left;
      border: 1px solid var(--color-border);
    }}
    th {{
      background-color: #f1f5f9;
      font-weight: 600;
      color: #0f172a;
    }}
    tr:nth-child(even) td {{
      background-color: #f8fafc;
    }}
    pre {{
      background-color: #0f172a;
      color: #f8fafc;
      padding: 18px;
      border-radius: 8px;
      overflow-x: auto;
      font-family: var(--font-mono);
      font-size: 0.82rem;
      line-height: 1.45;
      margin: 20px 0;
    }}
    code {{
      font-family: var(--font-mono);
      font-size: 0.82em;
      background: var(--color-code-bg);
      color: #0071e3;
      padding: 2px 6px;
      border-radius: 4px;
      border: 1px solid #e2e8f0;
    }}
    pre code {{
      background: transparent;
      color: inherit;
      border: none;
      padding: 0;
    }}
    hr {{
      border: none;
      border-top: 1px solid var(--color-border);
      margin: 32px 0;
    }}
    .print-bar {{
      max-width: 900px;
      margin: 0 auto 16px auto;
      display: flex;
      justify-content: flex-end;
    }}
    .print-btn {{
      padding: 8px 18px;
      background: var(--color-primary);
      color: #ffffff;
      border: none;
      border-radius: 8px;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
    }}
    @media print {{
      body {{
        background: #ffffff;
        padding: 0;
      }}
      .report-container {{
        border: none;
        box-shadow: none;
        padding: 0;
        max-width: 100%;
      }}
      .print-bar {{
        display: none;
      }}
      @page {{
        margin: 15mm;
      }}
    }}
  </style>
</head>
<body>
  <div class="print-bar">
    <button class="print-btn" onclick="window.print()">Print / Save as PDF</button>
  </div>
  <div class="report-container">
    {rendered_body}
  </div>
</body>
</html>
"""
    html_path.write_text(html_document, encoding="utf-8")
    print(f"[SUCCESS] Wrote styled report to: {html_path}")

if __name__ == "__main__":
    render_report()
