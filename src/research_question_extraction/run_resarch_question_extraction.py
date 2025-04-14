import jsonl

import vllm
from tap import Tap

from src.path import get_extracted_research_question_model_responses_path


class ResearchQuestionExtractionConfig(Tap):
    dataset_dir: str  # jsonl
    model_name: str = "google/gemma-3-27b-it"
    max_tokens: int = 2048
    batch_size: int = 16


def main():
    args = ResearchQuestionExtractionConfig().parse_args()
    
    for split in ["train", "test"]:
        
        # load dataset
        # abstract with few-shot demonstrations
        with open(args.dataset_path, "r") as f:
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
        
        # make batched prompt
        prompts = [d["prompt"] for d in dataset]
        batched_prompts = []
        for i in range(0, len(prompts), args.batch_size):
            batched_prompts.append(prompts[i : i + args.batch_size])
        
        # generate
        outputs = []
        for batch in batched_prompts:
            response = model.generate(prompt, sampling_params=sampling_params)
            outputs.extend(
                [o[0].text for o in outputs]
            )

        # save outputs
        save_path = get_extracted_research_question_model_responses_path(
            args.model_name, split
        )


if __name__ == "__main__":
    main()
