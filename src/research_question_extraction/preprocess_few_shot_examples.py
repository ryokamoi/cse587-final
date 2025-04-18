from pathlib import Path
import json
import re

from src.path import few_shot_examples_path


def parse_examples(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into examples using "Example " followed by a number and colon
    raw_examples = re.split(r'Example \d+:', content)
    parsed_data = []

    for example in raw_examples:
        # 
        if "abstract" not in example:
            continue

        if example.strip() == "":
            continue  # skip empty

        # Extract fields using regex
        abstract_match = re.search(r'abstract:\s*(.*?)\s*research question:', example, re.DOTALL)
        rq_match = re.search(r'research question:\s*(.*?)\s*approach:', example, re.DOTALL)
        approach_match = re.search(r'approach:\s*(.*)\s*proposed a new method:', example, re.DOTALL)
        proposed_a_new_method_match = re.search(r'proposed a new method:\s*(.*)', example, re.DOTALL)

        parsed_entry = {
            "abstract": abstract_match.group(1).strip() if abstract_match else "",
            "research_question": rq_match.group(1).strip() if rq_match else "",
            "approach": approach_match.group(1).strip() if approach_match else "",
            "proposed_a_new_method_match": proposed_a_new_method_match.group(1).strip() if proposed_a_new_method_match else "",
        }

        parsed_data.append(parsed_entry)

    return parsed_data

def save_as_jsonl(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    few_shot_examples_path.parent.mkdir(parents=True, exist_ok=True)

    input_txt = 'Few-shot Examples.txt'
    examples = parse_examples(input_txt)
    save_as_jsonl(examples, few_shot_examples_path)

    print(f"Extracted {len(examples)} examples to {few_shot_examples_path}")
