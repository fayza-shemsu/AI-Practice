import os
import re

files = [
    'src/week7/integration_thursday.py',
    'src/week7/ocr_tuesday.py',
    'src/week7/receipt_friday.py',
    'src/week7/speech_wednesday.py',
    'src/week7/vision_demo.py',
    'src/week7/vision_monday.py',
    'src/week8/indexer_thursday.py',
    'src/week8/invoice_monday.py',
    'src/week8/search_friday.py',
    'src/week8/search_wednesday.py',
    'src/week9/openai_monday.py',
    'src/week9/prompt_engineering_tuesday_v2.py',
    'src/week9/context_window_friday.py',
    'src/week9/system_message_wednesday.py',
    'src/week9/system_message_wednesday_v2.py',
    'src/week9/terminal_chat_thursday.py',
    'src/week10/add_your_data_thursday.py',
    'src/week10/embeddings_monday.py',
    'src/week10/rag_flow_wednesday.py',
    'src/week10/upload_documents.py',
    'src/week10/vector_search_tuesday.py',
    'src/week10/friday_chunk_optimization.py',
    'src/week11/friday_agents/agents_friday.py',
    'src/week11/thursday_deploy/score.py',
    'src/week11/tuesday_custom_tools.py',
    'src/week11/wednesday_eval/groundedness_eval.py',
    'src/week11/wednesday_evaluation.py',
]

for filepath in files:
    if not os.path.exists(filepath):
        print(f"⊘ Not found: {filepath}")
        continue
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Add imports at the top if not present
    if 'from dotenv import load_dotenv' not in content and 'import os' in content:
        content = content.replace('import os', 'import os\nfrom dotenv import load_dotenv\n\nload_dotenv()', 1)
    elif 'from dotenv import load_dotenv' not in content:
        lines = content.split('\n')
        lines.insert(0, 'import os')
        lines.insert(1, 'from dotenv import load_dotenv')
        lines.insert(2, '')
        lines.insert(3, 'load_dotenv()')
        content = '\n'.join(lines)
    
    # Replace patterns: KEY = "string" or KEY = 'string'
    content = re.sub(r'KEY\s*=\s*["\']EVsspW5fNPdxmxoQPJeWl2zgF2g4HlEs1aJd4Br5M3qPJ1I1vFhDJQQJ99CEACYeBjFXJ3w3AAAFACOGynap["\']', 'KEY = os.getenv("AZURE_VISION_KEY")', content)
    content = re.sub(r'VISION_KEY\s*=\s*["\']EVsspW5fNPdxmxoQPJeWl2zgF2g4HlEs1aJd4Br5M3qPJ1I1vFhDJQQJ99CEACYeBjFXJ3w3AAAFACOGynap["\']', 'VISION_KEY = os.getenv("AZURE_VISION_KEY")', content)
    content = re.sub(r'SPEECH_KEY\s*=\s*["\']6yQKLUGc1Ffo4SUPVgxvYb4NizTaQaaRDgXdAYRsnD6OmgtYFj0iJQQJ99CEACYeBjFXJ3w3AAAYACOGbcaE["\']', 'SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")', content)
    content = re.sub(r'DI_KEY\s*=\s*["\']49TWKpsWyt3iFly4IpoqdumsW8rcaodkjp2ac1i86LSkHNA5Ew0KJQQJ99CEACYeBjFXJ3w3AAALACOGxHae["\']', 'DI_KEY = os.getenv("AZURE_FORM_RECOGNIZER_KEY")', content)
    content = re.sub(r'SEARCH_KEY\s*=\s*["\']P8vmYuqS7rOctpch0i8SMVFjOBokUtCpufq9B1s9cmAzSeCFyHJC["\']', 'SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")', content)
    content = re.sub(r'AZURE_OAI_KEY\s*=\s*["\'][^"\']*["\']', 'AZURE_OAI_KEY = os.getenv("AZURE_OPENAI_KEY")', content)
    content = re.sub(r'AZURE_OAI_ENDPOINT\s*=\s*["\']https://fayz-openai\.openai\.azure\.com/["\']', 'AZURE_OAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")', content)
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✓ {filepath}")

print("\n✅ All Python files updated!")
