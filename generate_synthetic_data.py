import json
import requests
import random
import time
import os

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b-instruct-q4_K_M"
OUTPUT_FILE = "modified-prompt-2.jsonl"
NUM_EXAMPLES_TO_GENERATE = 100 # Adjust this to generate more/less

# Read the traits from the initial prompt file

def load_traits():
    traits = []
    try:
        with open("initial_prompt.md", "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                clean_line = line.strip()
                # Assuming the traits are just listed without bullets or numbers
                # Exclude the intro text by checking for length or skipping headers
                if clean_line and len(clean_line) > 10 and not clean_line.startswith(("you are", "the parser", "The model", "=====")):
                    # Strip out numbers if present (e.g. "1. Vague bullet points" -> "Vague bullet points")
                    stripped = clean_line.lstrip("1234567890. -*")
                    if stripped:
                        traits.append(stripped)
    except Exception as e:
        print(f"Error reading initial_prompt.md: {e}")
        # Fallback traits
        traits = ["Claiming disproportionate credit", "Vague, non-specific bullet points", "Keyword stuffing for ATS"]
    return traits

TRAITS = load_traits()

SECTIONS = [
    "EXPERIENCE", "PROJECTS", "EDUCATION", "SKILLS", "SUMMARY"
]

PERSONAS = [
    "You are an exhausted Senior Staff Engineer reviewing a junior's resume. You despise buzzwords and corporate jargon. Read the prompt and generate the flawed resume section.",
    "You are a chaotic Ivy-League tech recruiter. Read the prompt and generate the flawed resume section focusing on their ego or timeline gaps.",
    "You are the Gordon Ramsay of Software Engineering. Read the prompt and generate the 'raw' unstructured resume section.",
    "You are a brutally honest Startup Founder who has fired dozens of engineers. Read the prompt and generate the fake resume section focusing on their lack of real-world impact and business reality."
]

def generate_synthetic_example():
    target_trait = random.choice(TRAITS)
    target_section = random.choice(SECTIONS)
    persona = random.choice(PERSONAS)
    
    prompt = (
        f"Generate a highly realistic but terrible '{target_section}' section of a tech resume that actively demonstrates this specific flaw: '{target_trait}'.\n\n"
        "Format your response EXACTLY as follows:\n"
        "[RESUME SECTION]\n"
        "<Write the fake resume snippet here>\n"
        "[END SECTION]\n\n"
        "CRITICAL: Do NOT add any extra notes, summaries, or conversational filler. Output ONLY the requested blocks."
        # "<thoughts>\n"
        # "<Write your internal analysis spotting the flaw here>\n"
        # "</thoughts>\n\n"
        # "[ROAST]\n"
        # "<Write your brutal roast here>\n"

    )
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": persona},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.95, # High temp to prevent pattern collapse
        "top_p": 0.90
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        
        # In the chat API, the response is streamed line by line unless stream=False
        # But this endpoint defaults to stream=True. We must disable it.
        payload["stream"] = False
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response_json = response.json()
        
        content = response_json.get("message", {}).get("content", "")
        return {
            "trait": target_trait,
            "section": target_section,
            "persona": persona[:30] + "...", # Just saving a snippet of which persona was used
            "raw_output": content
        }
    except Exception as e:
        print(f"API Error: {e}")
        return None

def main():
    if not TRAITS:
        print("No traits loaded. Exiting.")
        return
        
    print(f"Loaded {len(TRAITS)} traits. Starting generation of {NUM_EXAMPLES_TO_GENERATE} examples...")
    
    success_count = 0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for i in range(NUM_EXAMPLES_TO_GENERATE):
            print(f"Generating example {i+1}/{NUM_EXAMPLES_TO_GENERATE}...")
            
            data = generate_synthetic_example()
            if data and "[RESUME SECTION]" in data["raw_output"]:
                # Parse out the components cleanly for the dataset if desired, 
                # or just store the raw block for standard instruction tuning.
                
                # Write to JSONL
                json_record = json.dumps(data, ensure_ascii=False)
                f.write(json_record + "\n")
                f.flush()
                success_count += 1
            else:
                print("  -> Failed or badly formatted output. Skipping.")
                
            # Slight delay to prevent overwhelming the local server
            time.sleep(3)
            
    print(f"\nDone! Successfully saved {success_count} synthetic pairs to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
    
