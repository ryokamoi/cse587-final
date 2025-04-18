import json

import vllm
from tap import Tap
from tqdm import tqdm

from src.path import get_evaluation_dataset_path, \
    get_predicted_research_question_model_responses_path
from src.utils.chat_template import get_chat_template


class InferenceTap(Tap):
    extraction_model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    model_name: str = "llama_factory_finetuned_models/Llama-3.1-8B-Instruct_cse587spring2025final_1.0e-5"
    max_tokens: int = 512
    batch_size: int = 16


def main():
    args = InferenceTap().parse_args()

    # load evaluation dataset
    evalaution_dataset_path = get_evaluation_dataset_path(
        model_name = args.extraction_model_name
    )
    with open(evalaution_dataset_path, "r") as f:
        dataset = [json.loads(line) for line in f.readlines()]

    # load vllm model
    model = vllm.LLM(model=args.model_name)
    if "gemma" in args.model_name:
        # gemma use different name for max tokens to generate
        sampling_params = vllm.SamplingParams(
            max_new_tokens=args.max_tokens,
            temperature=0,
        )
    else:
        sampling_params = vllm.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0,
        )
    
    ### for debug
    dataset = dataset[:32]
    ###
    
    # make batched prompt
    research_questions = [
        example["research_question"] for example in dataset
    ]
    batched_research_questions = []
    for i in range(0, len(research_questions), args.batch_size):
        batched_research_questions.append(
            research_questions[i : i + args.batch_size]
        )
    
    # generate
    outputs_list = []
    for batch in tqdm(batched_research_questions):
        converted_prompt = [
            [
                get_chat_template(
                    role="user",
                    content=example
                )
            ] for example in batch
        ]

        response = model.chat(
            messages=converted_prompt,
            sampling_params=sampling_params
        )
        outputs_list.extend(
            [o.outputs[0].text for o in response]
        )

    # save outputs
    save_path = get_predicted_research_question_model_responses_path(
        args.model_name
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        for i, output in enumerate(outputs_list):
            data_id = dataset[i]["id"]
            f.write(
                json.dumps({"id": data_id, "output": output}) + "\n"
            )


if __name__ == "__main__":
    main()
