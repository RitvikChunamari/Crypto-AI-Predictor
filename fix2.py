with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace("document.getElementById('ui-conf').innerText = '--';", "document.getElementById('ui-conf').innerText = '--';\n                document.getElementById('ui-acc').innerText = '--';")
content = content.replace("document.getElementById('ui-conf').innerText = `${data.confidence}%`;", "document.getElementById('ui-conf').innerText = `${data.confidence}%`;\n                        document.getElementById('ui-acc').innerText = `${data.accuracy}%`;")

with open("main.py", "w", encoding="utf-8") as f:
    f.write(content)
