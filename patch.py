import re

with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Insert the cache dictionary
if "loaded_models = {}" not in content:
    content = content.replace("app = FastAPI()", "app = FastAPI()\n\nloaded_models = {}")

# Replace the loading logic
old_logic = """
        if os.path.exists(model_path):
            model = load_model(model_path)
        elif os.path.exists(fallback_path):
            # Use Transfer Learning! The features are min-max scaled dynamically per coin, 
            # meaning the BTC-USD Neural Net is universally applicable to any asset's normalized geometric patterns!
            model = load_model(fallback_path)
        else:
            return {"error": "Critical Error: Core Neural Network weights missing from server."}
"""

new_logic = """
        global loaded_models
        
        if model_path in loaded_models:
            model = loaded_models[model_path]
        elif os.path.exists(model_path):
            model = load_model(model_path)
            loaded_models[model_path] = model
        elif fallback_path in loaded_models:
            model = loaded_models[fallback_path]
        elif os.path.exists(fallback_path):
            model = load_model(fallback_path)
            loaded_models[fallback_path] = model
        else:
            return {"error": "Critical Error: Core Neural Network weights missing from server."}
"""

content = content.replace(old_logic.strip(), new_logic.strip())

with open("main.py", "w", encoding="utf-8") as f:
    f.write(content)
