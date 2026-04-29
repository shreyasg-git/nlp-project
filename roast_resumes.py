import json
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
except ImportError:
    print("Error: The 'groq' library is not installed. Please run: pip install groq")
    sys.exit(1)

def roast_resumes(input_file, output_file):
    """
    Reads resume sections from a JSONL file, roasts them using Groq API,
    and saves the results to another JSONL file.
    """
    # Load the API key from environment variable
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        print("Error: Please set the GROQ_API_KEY environment variable.")
        sys.exit(1)
        
    client = Groq(api_key=api_key)
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Reading from {input_file}...")
    
    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total = len(lines)
    print(f"Found {total} entries. Starting the roast session...")

    # Open the output file in write mode to start fresh
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for i, line in enumerate(lines):
            if not line.strip():
                continue
                
            for attempt in range(2):
                try:
                    data = json.loads(line)
                    resume_text = data.get('raw_output', data.get('raw_text', ''))
                    
                    if not resume_text:
                        break
                        
                    prompt = f"""
                    You are a brutally honest, sarcastic tech recruiter. 
                    Roast the following resume section. 
                    Then, identify specific red flags as a list of short topics.
                    
                    Return ONLY a JSON object with the following structure:
                    {{
                        "roast": "Your brutal roast here",
                        "red_flags": ["Topic 1", "Topic 2"]
                    }}

                    Resume Section:
                    {resume_text}
                    """
                    
                    completion = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"}
                    )
                    
                    response_data = json.loads(completion.choices[0].message.content)
                    
                    result_entry = {
                        "resume": resume_text,
                        "red_flags": response_data.get("red_flags", []),
                        "actual_roast": response_data.get("roast", "")
                    }
                    
                    # Write immediately and flush to disk
                    out_f.write(json.dumps(result_entry) + '\n')
                    out_f.flush() 
                    
                    print(f"[{i+1}/{total}] Roasted and saved.")
                    break
                    
                except Exception as e:
                    if attempt == 0:
                        print(f"[{i+1}/{total}] Error: {e}. Retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        print(f"[{i+1}/{total}] Failed after retry. Error: {e}")

            
    print(f"\nAll done! Results stored in {output_file}")

if __name__ == "__main__":
    batch = 3
    roast_resumes(f'modified-prompt-{batch}.jsonl', f'roast_results_gpt-oss-120b_{batch}.jsonl')
