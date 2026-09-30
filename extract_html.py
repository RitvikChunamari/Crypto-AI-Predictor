import re

with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Extract HTML content
match = re.search(r'html_content = """(.*?)"""\n    return HTMLResponse', content, re.DOTALL)
html = match.group(1)

# Replace fetch URL to be absolute
html = html.replace('await fetch(`/predict', 'await fetch(`https://crypto-ai-api-y2j3.onrender.com/predict')

with open("crypto-terminal.html", "w", encoding="utf-8") as f:
    f.write(html.strip())
