import json
import os

# Configuration
INPUT_FILE = "roast_results_gpt-oss-120b_4.jsonl"
OUTPUT_FILE = "llama3-roast-dataset.jsonl"
SYSTEM_PROMPT = "You are a brutally honest, sarcastic tech recruiter. Your task is to analyze resume sections, identify red flags, and provide a brutal roast."

def format_entry(entry):
    """Formats a single JSONL entry into Llama 3.1 Instruct format with Reasoning."""
    resume = entry.get("resume", "")
    red_flags = ", ".join(entry.get("red_flags", []))
    roast = entry.get("actual_roast", "")
    
    # We combine red_flags as the 'reasoning' and actual_roast as the 'output'
    # This teaches the model to 'think' about flaws before roasting.
    full_response = f"<thought>\nRed Flags identified: {red_flags}\n</thought>\n\n{roast}"
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Roast this resume section:\n\n{resume}"},
            {"role": "assistant", "content": full_response}
        ]
    }

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please ensure your data file is in the current directory.")
        return

    processed_count = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                formatted = format_entry(data)
                f_out.write(json.dumps(formatted) + "\n")
                processed_count += 1
            except Exception as e:
                print(f"Error processing line: {e}")
                
    print(f"Successfully formatted {processed_count} examples into {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
