import argparse
import json
import urllib.request
import urllib.error
from parser_code import ResumeParser

def call_local_llm(prompt: str, context_section: str, model: str = "llama3", url: str = "http://localhost:11434/api/generate") -> str:
    """
    Sends the prompt and context section to a local LLM API (default expects Ollama format).
    """
    # Combine the system prompt with the specific section text
    full_prompt = f"{prompt}\n\n--- DOCUMENT SECTION ---\n{context_section}\n------------------------\n"
    
    # Request payload for Ollama
    # If you use an OpenAI-compatible server (like vLLM, llama.cpp server, text-generation-webui), 
    # adjust this to use the /v1/chat/completions schema with 'messages'.
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get('response', '')
    except urllib.error.URLError as e:
        return f"Connection error calling local LLM: {e.reason}. Ensure the server is running at {url}."
    except Exception as e:
        return f"Error calling local LLM: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Process a PDF by dividing it into sections and running each through a local LLM.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--prompt", required=False, help="Instructional prompt for the LLM (e.g. 'Summarize this section:')")
    parser.add_argument("--model", default="llama3", help="Name of the local model to query (default: llama3)")
    parser.add_argument("--url", default="http://localhost:11434/api/generate", help="Local LLM API endpoint (default: Ollama's endpoint)")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Approximate max characters per section (default: 2000)")
    
    args = parser.parse_args()
    
    # HARDCODED PROMPT (Overrides the --prompt argument for now)
    hardcoded_prompt = """Please analyze the following section. Extract any key entities, summarize the main points, and identify any specific traits or details of interest."""
    active_prompt = hardcoded_prompt
    
    print(f"[*] Parsing PDF using ResumeParser: {args.pdf_path}")
    try:
        parser = ResumeParser()
        parsed_data = parser.parse_pdf(args.pdf_path)
        sections_dict = parsed_data.get("sections", {})
    except Exception as e:
        print(f"[!] Error reading PDF: {e}")
        return

    print(f"[*] Found {len(sections_dict)} sections to process.\n")
    
    for i, (section_name, section_content) in enumerate(sections_dict.items(), 1):
        # Convert section_content to a string if it's a list (like EXPERIENCE)
        if isinstance(section_content, list):
            section_text = json.dumps(section_content, indent=2)
        else:
            section_text = str(section_content)

        print("=" * 60)
        print(f"Processing Section {i}/{len(sections_dict)}: {section_name} ({len(section_text)} characters)")
        print("=" * 60)
        
        # --- DEBUG: Print the parsed section ---
        print("\n--- Parsed Section Content ---")
        print(section_text)
        print("------------------------------\n")
        
        response = call_local_llm(prompt=active_prompt, context_section=section_text, model=args.model, url=args.url)
        
        print("\n--- LLM Output ---\n")
        print(response)
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()
