# Synthetic Dataset Generation Guide: "Resume Roaster"

## The Recommended Model
For generating high-quality synthetic roasts, the undisputed best choice in the small-model tier is **Meta's Llama-3-8B-Instruct**. 
- **Why?** It fits easily on consumer hardware (requires ~8GB VRAM in 4-bit quant), has an incredibly sharp understanding of sarcasm/humor, and responds very well to strict output formatting (JSON).

## Architecture Overview
You will run a local inference server (like **Ollama**) hosting the Llama-3-8B model. 
Then, you will run a simple Python script (`generator.py`) that loops through a folder of real resumes (or snippets) and asks the local model to generate roasts for them, saving the output.

## Step 1: Set Up Local Inference
1. Download **Ollama** (https://ollama.com).
2. Start the local server by running this command in your terminal:
   `ollama run llama3:8b-instruct`
3. This creates a local API endpoint on your machine at `http://localhost:11434`.

## Step 2: The Multi-Persona System Prompts
To avoid "pattern collapse", your script should randomly pick one of these system prompts for every resume it processes:

- **Persona A (The Brutal Tech Lead):** "You are an exhausted Senior Staff Engineer reviewing a junior's resume. You hate buzzwords. Read the parsed resume and provide internal thoughts in `<thoughts>` tags identifying specific flaws, then write a funny, brutal, 3-sentence roast focusing on those flaws."
- **Persona B (The Gatekeeping Recruiter):** "You are a chaotic Ivy-League tech recruiter. Read the resume, use `<thoughts>` tags to identify timeline gaps and over-inflated titles, and then write a highly sarcastic roast that tears apart their self-importance."
- **Persona C (Gordon Ramsay of Code):** "You are the Gordon Ramsay of Software Engineering. Look at this resume, identify the 'raw' unstructured mess inside `<thoughts>`, and deliver a blistering, metaphorical kitchen-nightmare roast of the applicant."

## Step 3: Python Generator Script Blueprint
Use the `requests` library to ping your local Ollama server. 

```python
import json
import requests
import random

PERSONAS = [
    "...Persona A text...",
    "...Persona B text..."
]

SECTIONS = ["EXPERIENCE", "PROJECTS", "EDUCATION", "SKILLS"]
NEGATIVE_TRAITS = [
    "Claiming disproportionate credit",
    "Missing dates or hidden timeline gaps",
    "Unprofessional email address",
    "Vague, non-specific bullet points"
]

def generate_synthetic_section_pair():
    target_trait = random.choice(NEGATIVE_TRAITS)
    target_section = random.choice(SECTIONS)
    
    prompt = f"Generate a highly realistic but terrible '{target_section}' section of a tech resume that actively demonstrates this flaw: '{target_trait}'. \nThen, provide the '<thoughts>' section identifying the flaw, followed by a specific, sarcastic roast of this exact section."
    
    prompt_payload = {
        "model": "llama3:8b-instruct",
        "messages": [
            {"role": "system", "content": random.choice(PERSONAS)},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.95, # High temperature for creativity
    }
    
    response = requests.post("http://localhost:11434/api/chat", json=prompt_payload)
    return response.json()

# Run this in a loop to generate 3000 entirely synthetic Section->Roast pairs!
```

## Step 4: Quality Filtering (Best-of-N)
1. Modify your script to request 3 roasts for *each* resume.
2. Filter the responses based on length constraints or keyword diversity (e.g. automatically delete responses that use the cliché word "delve").
3. Save the surviving, high-quality entries into a final `dataset.jsonl` file formatted for fine-tuning.
