import re
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parent.parent
readme_path = ROOT / "README.md"
index_html_path = ROOT / "index.html"
index_md_path = ROOT / "index.md"

with open(readme_path, "r", encoding="utf-8") as f:
    raw_content = f.read()

# Copy to index.md for Jekyll compatibility as well
with open(index_md_path, "w", encoding="utf-8") as f:
    f.write(raw_content)

# Strip YAML frontmatter for HTML conversion
body_md = re.sub(r"^---\s*\n.*?\n---\s*\n", "", raw_content, flags=re.DOTALL)

# Convert Mermaid blocks to <pre class="mermaid">
def mermaid_replacer(match):
    code = match.group(1)
    return f'<div class="mermaid">\n{code}\n</div>'

body_md = re.sub(r"```mermaid\n(.*?)```", mermaid_replacer, body_md, flags=re.DOTALL)

# Convert markdown to HTML
html_body = markdown.markdown(body_md, extensions=["tables", "fenced_code", "attr_list"])

# Full HTML template
html_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>When Do AI Agents Actually Matter? | TigerGraph Agentic GraphRAG</title>
    <meta name="description" content="Benchmarking Vanilla RAG, GraphRAG, and Agentic GraphRAG with TigerGraph on 150 complex multi-hop, temporal, and aggregation queries.">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
    </script>
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
            }}
        }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        :root {{
            --bg-color: #0d1117;
            --card-bg: #161b22;
            --border-color: #30363d;
            --text-color: #c9d1d9;
            --heading-color: #ffffff;
            --accent-orange: #ff5722;
            --accent-cyan: #58a6ff;
            --accent-green: #3fb950;
            --accent-purple: #bc8cff;
            --code-bg: #1f242c;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.7;
            font-size: 16px;
            padding: 0;
        }}
        .navbar {{
            background: rgba(22, 27, 34, 0.85);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 100;
            padding: 0.8rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .navbar-brand {{
            font-weight: 700;
            font-size: 1.1rem;
            color: #fff;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .navbar-brand span {{
            color: var(--accent-orange);
        }}
        .nav-links {{
            display: flex;
            gap: 1.2rem;
            align-items: center;
        }}
        .nav-link {{
            color: #8b949e;
            text-decoration: none;
            font-size: 0.9rem;
            font-weight: 500;
            transition: color 0.2s;
        }}
        .nav-link:hover {{
            color: var(--accent-cyan);
        }}
        .nav-btn {{
            background: var(--accent-orange);
            color: #fff;
            padding: 0.4rem 0.9rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
            text-decoration: none;
            transition: opacity 0.2s;
        }}
        .nav-btn:hover {{
            opacity: 0.9;
        }}
        .container {{
            max-width: 920px;
            margin: 2.5rem auto;
            padding: 0 1.5rem 4rem 1.5rem;
        }}
        h1, h2, h3, h4 {{
            color: var(--heading-color);
            font-weight: 700;
            margin-top: 2.2rem;
            margin-bottom: 1rem;
            letter-spacing: -0.02em;
        }}
        h1 {{
            font-size: 2.3rem;
            line-height: 1.25;
            margin-top: 1rem;
            background: linear-gradient(135deg, #ffffff 40%, #ff8a65 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        h2 {{
            font-size: 1.6rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5rem;
            margin-top: 2.8rem;
        }}
        h3 {{
            font-size: 1.25rem;
            color: #e6edf3;
        }}
        p {{
            margin-bottom: 1.2rem;
        }}
        a {{
            color: var(--accent-cyan);
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        blockquote {{
            border-left: 4px solid var(--accent-orange);
            background: var(--card-bg);
            padding: 1rem 1.2rem;
            margin: 1.5rem 0;
            border-radius: 0 8px 8px 0;
            color: #e6edf3;
            font-style: italic;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            background: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }}
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background: #21262d;
            color: #f0f6fc;
            font-weight: 600;
            font-size: 0.9rem;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}
        code {{
            font-family: 'JetBrains Mono', monospace;
            background: var(--code-bg);
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-size: 0.88em;
            color: #f0883e;
            border: 1px solid #30363d;
        }}
        pre {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.2rem;
            overflow-x: auto;
            margin: 1.5rem 0;
        }}
        pre code {{
            background: transparent;
            padding: 0;
            border: none;
            color: #c9d1d9;
            font-size: 0.9rem;
        }}
        .mermaid {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1.8rem 0;
            text-align: center;
        }}
        ul, ol {{
            margin: 1rem 0 1.2rem 1.8rem;
        }}
        li {{
            margin-bottom: 0.4rem;
        }}
        hr {{
            border: none;
            border-top: 1px solid var(--border-color);
            margin: 2.5rem 0;
        }}
        img {{
            max-width: 100%;
            border-radius: 8px;
        }}
        .footer {{
            border-top: 1px solid var(--border-color);
            text-align: center;
            padding: 2.5rem 1rem;
            color: #8b949e;
            font-size: 0.9rem;
            margin-top: 4rem;
        }}
    </style>
</head>
<body>
    <header class="navbar">
        <a href="#" class="navbar-brand">
            🐯 <span>TigerGraph</span> Agentic GraphRAG
        </a>
        <div class="nav-links">
            <a href="https://github.com/Arkz-Deepak/agentic-graphrag-tigergraph" target="_blank" class="nav-link">GitHub Repo</a>
            <a href="https://github.com/Arkz-Deepak/agentic-graphrag-tigergraph/blob/main/benchmark_results/public_benchmark_summary.json" target="_blank" class="nav-link">Benchmarks</a>
            <a href="https://github.com/Arkz-Deepak/agentic-graphrag-tigergraph/blob/main/benchmark_results/eval_hidden_submission.jsonl" target="_blank" class="nav-link">50 Hidden Eval</a>
            <a href="https://github.com/Arkz-Deepak/agentic-graphrag-tigergraph" target="_blank" class="nav-btn">Star on GitHub ⭐</a>
        </div>
    </header>

    <main class="container">
        {html_body}
    </main>

    <footer class="footer">
        <p>Built with ❤️ by <strong>Deepak R (Team Arkz)</strong> for the <strong>TigerGraph Agentic GraphRAG Hackathon 2026</strong>.</p>
        <p style="margin-top: 0.4rem;">Open Source under the Apache 2.0 License • Hosted natively on GitHub Pages</p>
    </footer>
</body>
</html>
"""

with open(index_html_path, "w", encoding="utf-8") as f:
    f.write(html_document)

print(f"Generated index.html ({len(html_document)} bytes) and updated index.md ({len(raw_content)} bytes)")
