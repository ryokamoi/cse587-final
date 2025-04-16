from pathlib import Path

# dataset
dataset_dir = Path("dataset")
abstracts_dir = dataset_dir / "abstracts"
few_shot_examples_path = dataset_dir / "few_shot_examples.jsonl"
extracted_research_questions_dir = dataset_dir / "extracted_research_questions"
evaluation_dataset_dir = dataset_dir / "evaluation_dataset"

sharegpt_dataset_dir = dataset_dir / "sharegpt_dataset"

# raw resources
downloaded_abstracts_dir = Path("downloaded_abstracts")

# model responses
outputs_dir = Path("outputs")  # model responses
extracted_research_questions_dir = outputs_dir / "extracted_research_questions"

# llama factory
llama_factory_dir = Path("../LLaMA-Factory-CSE587")


def get_model_short_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def get_extracted_research_question_model_responses_path(
        model_name: str, split: str):
    
    model_short_name = get_model_short_name(model_name)
    return extracted_research_questions_dir / model_short_name / f"{split}.jsonl"


def get_evaluation_dataset_path(model_name: str):
    model_short_name = get_model_short_name(model_name)
    return evaluation_dataset_dir / f"extraction={model_short_name}" / "evaluation.jsonl"


def get_predicted_research_question_model_responses_path(
        model_name: str
    ) -> Path:
    model_short_name = get_model_short_name(model_name)
    return outputs_dir / "predicted_research_questions" / f"{model_short_name}.jsonl"
