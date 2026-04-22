import sys
import json
import argparse
from nlp_extractor import analyze_traits

def main():
    parser = argparse.ArgumentParser(description="Analyze resume JSON output for negative traits using NLP.")
    parser.add_argument('json_path', help='Path to the resume parsed JSON file or md file')
    args = parser.parse_args()

    with open(args.json_path, 'r', encoding='utf-8') as f:
        content = f.read()

    try:
        parsed_data = json.loads(content)
    except json.JSONDecodeError:
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx != -1 and end_idx != -1:
            try:
                json_str = content[start_idx:end_idx+1]
                parsed_data = json.loads(json_str, strict=False)
            except json.JSONDecodeError as e:
                print(f"Error: Could not extract valid JSON from the provided file. Exception: {e}")
                sys.exit(1)
        else:
            print("Error: No JSON object found in the file.")
            sys.exit(1)
            
    traits_result = analyze_traits(parsed_data)
    
    print("=" * 60)
    print("RESUME TRAITS ANALYSIS")
    print("=" * 60)
    for trait, found in traits_result.items():
        icon = "[FAIL]" if found else "[ OK ]"
        print(f"{icon} {trait}")

if __name__ == "__main__":
    main()
