import json
import os
import sys
import random
import time

try:
    from groq import Groq
except ImportError:
    print("Error: The 'groq' library is not installed. Please run: pip install groq")
    sys.exit(1)

# Configuration
MODEL = "openai/gpt-oss-120b"
OUTPUT_FILE = "modified-prompt-groq-4.jsonl"

NUM_EXAMPLES_TO_GENERATE = 100 

def load_traits():
    traits = []
    try:
        with open("initial_prompt.md", "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                clean_line = line.strip()
                if clean_line and len(clean_line) > 10 and not clean_line.startswith(("you are", "the parser", "The model", "=====")):
                    stripped = clean_line.lstrip("1234567890. -*")
                    if stripped:
                        traits.append(stripped)
    except Exception as e:
        print(f"Error reading initial_prompt.md: {e}")
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

def generate_synthetic_example(client):
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
    )
    
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": persona},
                {"role": "user", "content": prompt}
            ],
            temperature=0.95,
            top_p=0.90,
            stream=False
        )
        
        content = completion.choices[0].message.content
        return {
            "trait": target_trait,
            "section": target_section,
            "persona": persona[:30] + "...", 
            "raw_output": content
        }
    except Exception as e:
        print(f"API Error: {e}")
        return None

def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: Please set the GROQ_API_KEY environment variable.")
        sys.exit(1)
        
    client = Groq(api_key=api_key)

    if not TRAITS:
        print("No traits loaded. Exiting.")
        return
        
    print(f"Loaded {len(TRAITS)} traits. Starting generation of {NUM_EXAMPLES_TO_GENERATE} examples using {MODEL}...")
    
    success_count = 0
    # Use 'a' to append so you can resume if you hit daily limits
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for i in range(NUM_EXAMPLES_TO_GENERATE):
            print(f"Generating example {i+1}/{NUM_EXAMPLES_TO_GENERATE}...")
            
            data = generate_synthetic_example(client)
            if data and "[RESUME SECTION]" in data["raw_output"]:
                json_record = json.dumps(data, ensure_ascii=False)
                f.write(json_record + "\n")
                f.flush()
                success_count += 1
            else:
                print("  -> Failed or badly formatted output. Skipping.")
                
            # SAFETY DELAY: 3 seconds handles both 30 RPM and helps with TPM limits
            time.sleep(3)
            
    print(f"\nDone! Successfully saved {success_count} synthetic pairs to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
