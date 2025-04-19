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

# human annotation
human_annotation_dir = Path("human_annotation")
human_annotation_csv_dir = human_annotation_dir / "csv"


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
        model_name: str, few_shot: bool = False
    ) -> Path:
    model_short_name = get_model_short_name(model_name)
    if few_shot:
        model_short_name = f"few_shot_{model_short_name}"

    return outputs_dir / "predicted_approaches" / f"{model_short_name}.jsonl"
