import json
import os
import sys
import requests

def call_ollama(prompt, model="llama3.1:8b-instruct-q4_K_M"):
    """
    Calls the local Ollama API. 
    Assumes Ollama is running on the default port 11434.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return json.loads(response.json()['response'])
    except requests.exceptions.RequestException as e:
        print(f"Ollama API Error: {e}")
        return None
    except json.JSONDecodeError:
        print("Error: Ollama did not return valid JSON.")
        return None

def main(input_file, output_file, local_model="llama3"):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run the Groq script first.")
        return

    # Read the teacher results
    with open(input_file, 'r', encoding='utf-8') as f:
        entries = [json.loads(line) for line in f]

    total = len(entries)
    print(f"Found {total} teacher responses. Starting local model comparison...")

    # Open output file for immediate writing
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for i, entry in enumerate(entries):
            resume_text = entry.get('resume', '')
            teacher_roast = entry.get('actual_roast', '')
            teacher_flags = entry.get('red_flags', [])

            if not resume_text:
                continue

            prompt = f"""
            You are a brutally honest, sarcastic tech recruiter. 
            Roast the following resume section. 
            Identify specific red flags as a list of short topics.
            
            Return ONLY a JSON object with:
            {{
                "roast": "Your sarcastic roast",
                "red_flags": ["Topic 1", "Topic 2"]
            }}

            Resume Section:
            {resume_text}
            """

            # Call Ollama
            student_data = call_ollama(prompt, model=local_model)

            if student_data:
                # Combine teacher and student data for fine-tuning analysis
                comparison_entry = {
                    "resume": resume_text,
                    "teacher_roast": teacher_roast,
                    "teacher_flags": teacher_flags,
                    "student_roast": student_data.get('roast', ""),
                    "student_flags": student_data.get('red_flags', [])
                }
                
                # Write immediately
                out_f.write(json.dumps(comparison_entry) + '\n')
                out_f.flush()
                print(f"[{i+1}/{total}] Local model responded and saved.")
            else:
                print(f"[{i+1}/{total}] Failed to get response from local model.")

    print(f"\nWorkflow complete! Comparison data saved in {output_file}")

if __name__ == "__main__":
    # You can change 'llama3' to whatever model you have pulled in Ollama (e.g., 'mistral', 'phi3')
    target_model = "llama3" 
    main('roast_results.jsonl', 'finetune_comparison.jsonl', local_model=target_model)
