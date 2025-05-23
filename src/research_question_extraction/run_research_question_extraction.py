import json
from pathlib import Path

import torch
import vllm
from tap import Tap
from tqdm import tqdm

from src.path import get_extracted_research_question_model_responses_path, \
    few_shot_examples_path
from src.utils.chat_template import get_chat_template


class ResearchQuestionExtractionTap(Tap):
    abstracts_dir: str = "dataset/abstracts"  # jsonl
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    max_tokens: int = 2048
    batch_size: int = 16


user_input_template = """Your task is to extract the main research question and approach from the given abstract.
* If this work does not propose a new method, such as evaluation work or survey, please answer "no" and do not provide any research question, approach, or explanation.
* Your task is extraction and summarization. You should not add any additional information or explanation. You should not come up with any new research question or approach.
* Please refer to the previous examples as few-shot examples to understand the format of the output.

Abstract: {abstract}"""


response_template_yes_no_only = "This work proposes a new method: {yes_no}"

response_template = response_template_yes_no_only + """

Research Question: {research_question}

Approach: {approach}"""


def main():
    args = ResearchQuestionExtractionTap().parse_args()

    # load few-shot examples
    with open(few_shot_examples_path, "r") as f:
        raw_few_shot_examples = [json.loads(line) for line in f.readlines()]
    
    # make few-shot examples
    few_shot_examples = []
    for example in raw_few_shot_examples:
        user_input = get_chat_template(
            role="user",
            content=user_input_template.format(
                abstract=example["abstract"]
            )
        )

        if example["proposed_a_new_method_match"] == "yes":
            content = response_template.format(
                yes_no="yes",
                research_question=example["research_question"],
                approach=example["approach"]
            )
        elif example["proposed_a_new_method_match"] == "no":
            content = response_template_yes_no_only.format(
                yes_no="no"
            )
        else:
            raise ValueError(
                f"proposed_a_new_method_match should be yes or no, but got {example['proposed_a_new_method_match']}"
            )

        assistant_output = get_chat_template(
            role="assistant" if "gemma" not in args.model_name else "model",
            content=content
        )

        few_shot_examples.extend([user_input, assistant_output])
    
    print("Few-shot examples:")
    for example in few_shot_examples:
        print(example)
    
    # load vllm model
    num_gpus = torch.cuda.device_count()
    model = vllm.LLM(model=args.model_name, tensor_parallel_size=num_gpus)

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

    # do inference
    for split in ["train", "test"]:
        
        # load dataset
        # abstract with few-shot demonstrations
        with open(Path(args.abstracts_dir) / f"{split}.jsonl", "r") as f:
            dataset = [json.loads(line) for line in f.readlines()]
        
        # ### for debug
        # dataset = dataset[:32]
        # ###
        
        # make batched prompt
        abstracts = [d["abstract"] for d in dataset]
        batched_abstracts = []
        for i in range(0, len(abstracts), args.batch_size):
            batched_abstracts.append(abstracts[i : i + args.batch_size])
        
        # generate
        outputs_list = []
        for batch in tqdm(batched_abstracts):
            converted_prompt = [
                few_shot_examples + [
                    get_chat_template(
                        role="user",
                        content=user_input_template.format(
                            abstract=example
                        )
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
        save_path = get_extracted_research_question_model_responses_path(
            args.model_name, split
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
