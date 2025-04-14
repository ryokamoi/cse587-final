from pathlib import Path

downloaded_abstracts_dir = Path("downloaded_abstracts")

outputs_dir = Path("outputs")  # model responses


extracted_research_questions_dir = outputs_dir / "extracted_research_questions"


def get_model_short_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def get_extracted_research_question_model_responses_path(
        model_name: str, split: str):
    
    model_short_name = get_model_short_name(model_name)
    return extracted_research_questions_dir / model_short_name / f"{split}.jsonl"

