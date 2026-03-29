#!/usr/bin/env python3
"""Unified LLM Chat Client."""

# ===== STANDARD LIBRARY IMPORTS =====
import os                          # Access environment variables (BASE_URL, API keys)
import sys                         # Handle CLI errors with sys.exit()
import requests                    # Make HTTP POST requests to LLM APIs
import argparse                    # Parse command line arguments (model, key)

# ===== GLOBAL CONSTANTS =====
BASE_URL = "https://apim-genaion-dev-uksouth-001.azure-api.net"  # Shared API domain for all models

# ===== MODEL CONFIGURATION (api_version, path_slug, model_name) =====
MODELS = {
    'gpt-5.1': ('2025-04-01-preview', 'gpt-5.1', 'gpt-5.1'),                    # GPT-5.1 config tuple
    'gpt-4o-mini': ('2025-01-01-preview', 'gpt-4o-mini', 'gpt-4o-mini'),        # GPT-4o-mini config tuple  
    'gpt-5': ('2025-01-01-preview', 'gpt-5', 'gpt-5'),                         # GPT-5 config tuple
    'gpt-5-mini': ('2025-01-01-preview', 'gpt-5-mini', 'gpt-5-mini'),          # GPT-5-mini config tuple
    'grok-3': ('2025-01-01-preview', 'grok-3', 'grok-3'),                      # Grok-3 config tuple
    'llama-3.3-70b-instruct': ('2024-05-01-preview', 'llama-3-3-70b-instruct', 'llama-3.3-70b-instruct')  # Llama config tuple
}

# ===== MAIN CLIENT CLASS =====
class Client:
    def __init__(self, model_id: str, key: str):
        # Validate model_id exists in MODELS dict, exit if invalid
        if model_id not in MODELS: sys.exit(f"Use: {', '.join(MODELS)}")
        
        # Extract (version, path, model) from selected model config
        version, path, model = MODELS[model_id]
        
        # Build full URL: BASE_URL/{path}/chat/completions (env override supported)
        self.url = f"{os.getenv('BASE_URL', BASE_URL)}/{path}/chat/completions"
        
        # Set API version query param (env override: MODELID_VERSION)
        self.params = {'api-version': os.getenv(f'{model_id.upper().replace("-", "")}_VERSION', version)}
        
        # Set standard HTTP headers: Content-Type + API subscription key
        self.headers = {'Content-Type': 'application/json', 'Ocp-Apim-Subscription-Key': key}
        
        # Store model name for request payload
        self.model = model
        
        # Format display name (Gpt 5.1 → "Gpt-5.1")
        self.name = model_id.title()

    def __call__(self, messages):
        # Send POST request to /chat/completions endpoint
        r = requests.post(self.url, headers=self.headers, params=self.params, 
                         json={'messages': messages, 'model': self.model})
        
        # Raise exception for HTTP 4xx/5xx errors (auth failures, rate limits)
        r.raise_for_status()
        
        # Extract and return assistant message content from response
        return r.json()['choices'][0]['message']['content']

# ===== CLI EXECUTION (runs only when script called directly) =====
if __name__ == "__main__":
    # Create argument parser for CLI usage
    p = argparse.ArgumentParser(description="LLM Chat")
    
    # Define required args: model (from MODELS), API key (-k flag)
    p.add_argument('model', choices=MODELS)
    p.add_argument('-k', '--key', required=True)
    
    # Parse command line arguments into args object
    args = p.parse_args()
    
    # Initialize client with parsed model_id and API key
    c = Client(args.model, args.key)
    
    # Print welcome message with pretty model name
    print(f"{c.name} Chat (exit=quit):")
    
    # Initialize conversation with system prompt
    m = [{'role': 'system', 'content': 'You are helpful.'}]
    
    # Main chat loop: continues until user types 'exit'
    while (i := input("You: ").strip()).lower() != 'exit':
        # Append user message + assistant response to conversation history
        m += [{'role': 'user', 'content': i}, {'role': 'assistant', 'content': c(m)}]
        
        # Print assistant's latest response
        print(f"{c.name}: {m[-1]['content']}")
