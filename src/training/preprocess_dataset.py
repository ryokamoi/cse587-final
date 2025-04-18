import re
import json
from pathlib import Path

from src.config import dataset_name
from src.research_question_extraction.run_research_question_extraction \
    import ResearchQuestionExtractionTap
from src.path import get_extracted_research_question_model_responses_path, \
    sharegpt_dataset_dir, llama_factory_dir, get_evaluation_dataset_path


def preprocess_extracted_research_questions_and_approaches(
        model_response: dict[str, str]) -> dict:
    # Use regular expressions to capture content after each label
    match = re.search(
        r"Research Question:\s*(.*?)\s*Approach:\s*(.*)",
        model_response["output"], re.DOTALL
    )
    
    if match:
        research_question = match.group(1).strip()
        approach = match.group(2).strip()
        return {
            "id": model_response["id"],
            "research_question": research_question,
            "approach": approach
        }
    else:
        return {
            "id": model_response["id"],
            "research_question": None,
            "approach": None
        }


def get_chat_template_in_sharegpt(role: str, content: str) -> dict:
    if role not in ["human", "gpt"]:
        raise ValueError("Role must be 'human' or 'gpt'")
    return {"from": role, "value": content}


def main():
    args = ResearchQuestionExtractionTap().parse_args()

    # load extracted research questions and approaches
    extracted_path = get_extracted_research_question_model_responses_path(
        model_name=args.model_name, split="train"
    )
    with open(extracted_path, "r") as f:
        raw_extracted_research_questions = [
            json.loads(line) for line in f.readlines()
        ]
    
    # preprocess research questions and approaches
    processed_research_questions = []
    for example in raw_extracted_research_questions:
        if "This work proposes a new method: no" in example["output"]:
            # skip the example
            continue

        processed_example = preprocess_extracted_research_questions_and_approaches(
            example
        )
        processed_research_questions.append(processed_example)
    
    evaluation_dataset_path = get_evaluation_dataset_path(
        model_name=args.model_name
    )
    evaluation_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(evaluation_dataset_path, "w") as f:
        for example in processed_research_questions:
            f.write(json.dumps(example) + "\n")

    # convert into ShareGPT format
    share_gpt_format_dataset = []
    for example in processed_research_questions:
        conversations = []
        conversations.append(get_chat_template_in_sharegpt(
            role="human",
            content=example["research_question"]
        ))
        conversations.append(get_chat_template_in_sharegpt(
            role="gpt",
            content=example["approach"]
        ))
        share_gpt_format_dataset.append({"conversations": conversations})
    
    # save to file
    output_path = sharegpt_dataset_dir / \
        args.model_name.split("/")[-1] / "train.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for example in share_gpt_format_dataset:
            f.write(json.dumps(example) + "\n")
    print(f"Saved to {output_path}")

    # update dataset_info.json in llama factory
    dataset_info_path = llama_factory_dir / "data/dataset_info.json"
    with open(dataset_info_path, "r") as f:
        dataset_info = json.load(f)

    dataset_info[dataset_name] = {
        "file_name": str(Path("../../cse587-final" / output_path)),
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
        }
    }

    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"Updated {dataset_info_path} with {dataset_name} dataset info.")

if __name__ == "__main__":
    main()
