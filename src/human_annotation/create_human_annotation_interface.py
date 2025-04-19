import json
import csv

from src.path import human_annotation_csv_dir


predicted_approaches = {
    "extracted": "dataset/evaluation_dataset/extraction=Llama-3.3-70B-Instruct/evaluation.jsonl",
    "llama31-8B": "outputs/predicted_approaches/few_shot_Llama-3.1-8B-Instruct.jsonl",
    "llama33-70B": "outputs/predicted_approaches/few_shot_Llama-3.3-70B-Instruct.jsonl",
    "llama31-8B-finetuned": "outputs/predicted_approaches/Llama-3.1-8B-Instruct_cse587spring2025final_1.0e-5.jsonl",
}


target_size = 20

def main():
    with open(predicted_approaches["extracted"], "r") as f:
        extracted_data = [json.loads(line) for line in f][:target_size]
    
    data_ids = [e["id"] for e in extracted_data]
    approaches = {
        "extracted": [e["approach"] for e in extracted_data],
    }

    # predicted approaches
    for method_name in ["llama31-8B", "llama33-70B", "llama31-8B-finetuned"]:
        with open(predicted_approaches[method_name], "r") as f:
            data = [json.loads(line) for line in f][:target_size]

        # check data id
        if not all(data[idx]["id"] == data_ids[idx] for idx in range(len(data))):
            print(f"Data ID mismatch in {method_name} approach.")
            raise ValueError("Data ID mismatch in predicted approaches.")
        
        # approaches
        approaches[method_name] = [d["output"] for d in data if d["id"] in data_ids]
    
    # write to csv file
    human_annotation_csv_dir.mkdir(parents=True, exist_ok=True)
    for method_name in ["extracted", "llama31-8B", "llama33-70B"]:
        with open(human_annotation_csv_dir / f"{method_name}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "approach1", "approach2"])
            for idx, approach in enumerate(approaches[method_name]):
                writer.writerow(
                    [
                        data_ids[idx],
                        approach, approaches["llama31-8B-finetuned"][idx]
                    ]
                )


if __name__ == "__main__":
    main()
