import json

import torch
import vllm
from tap import Tap
from tqdm import tqdm

from src.path import get_evaluation_dataset_path, \
    get_predicted_research_question_model_responses_path, \
    few_shot_examples_path
from src.utils.chat_template import get_chat_template


class InferenceTap(Tap):
    extraction_model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    model_name: str = "llama_factory_finetuned_models/Llama-3.1-8B-Instruct_cse587spring2025final_1.0e-5"
    max_tokens: int = 2048
    batch_size: int = 16
    few_shot: bool = False


research_question_generation_input_format = """Your task is to generate a possible approach to the research question.
* You should only generate the approach and do not include any other information.
* You should generate an approach that is novel and reasonably feasible.
* Your response should be a paragraph without any bullet points or lists.
* Your response should follow the format and style of the examples in the previous messages.

Research Question: {research_question}"""


def main():
    args = InferenceTap().parse_args()

    # load evaluation dataset
    evalaution_dataset_path = get_evaluation_dataset_path(
        model_name = args.extraction_model_name
    )
    with open(evalaution_dataset_path, "r") as f:
        dataset = [json.loads(line) for line in f.readlines()][:100]

    # make few-shot example
    few_shot_examples = []
    if args.few_shot:
        with open(few_shot_examples_path, "r") as f:
            raw_few_shot_examples = [
                json.loads(line) for line in f.readlines()
            ]

        for example in raw_few_shot_examples:
            few_shot_examples.append(
                get_chat_template(
                    role="user",
                    content=research_question_generation_input_format.format(
                        research_question=example["research_question"]
                    )
                )
            )
            few_shot_examples.append(
                get_chat_template(
                    role="assistant",
                    content=example["approach"]
                )
            )


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
    
    # ### for debug
    # dataset = dataset[:32]
    # ###
    
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
        # when using few-shot examples, input format should be updated
        if args.few_shot:
            batch = [
                research_question_generation_input_format.format(
                    research_question=example
                ) for example in batch
            ]

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
        args.model_name, few_shot=args.few_shot
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
