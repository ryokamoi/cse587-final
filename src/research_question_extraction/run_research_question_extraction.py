import json
from pathlib import Path

import vllm
from tap import Tap
from tqdm import tqdm

from src.path import get_extracted_research_question_model_responses_path, \
    few_shot_examples_path
from src.utils.chat_template import get_chat_template


class ResearchQuestionExtractionTap(Tap):
    abstracts_dir: str = "dataset/abstracts"  # jsonl
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_tokens: int = 2048
    batch_size: int = 16


user_input_template = """Your task is to extract the main research question and approach from the given abstract.

Abstract: {abstract}"""

response_template = """Research Question: {research_question}

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

        assistant_output = get_chat_template(
            role="assistant" if "gemma" not in args.model_name else "model",
            content=response_template.format(
                research_question=example["research_question"],
                approach=example["approach"]
            )
        )

        few_shot_examples.extend([user_input, assistant_output])
    
    print("Few-shot examples:")
    for example in few_shot_examples:
        print(example)
    
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

    # do inference
    for split in ["train", "test"]:
        
        # load dataset
        # abstract with few-shot demonstrations
        with open(Path(args.abstracts_dir) / f"{split}.jsonl", "r") as f:
            dataset = [json.loads(line) for line in f.readlines()]
        
        ### for debug
        dataset = dataset[:32]
        ###
        
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

            print(converted_prompt)

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
