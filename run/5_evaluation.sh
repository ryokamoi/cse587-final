# fine-tuned model
python src/inference/run_inference.py --model_name llama_factory_finetuned_models/Llama-3.1-8B-Instruct_cse587spring2025final_1.0e-5

# baseline models (not fine-tuned)
python src/inference/run_inference.py --few_shot --model_name meta-llama/llama-3.1-8B-Instruct
python src/inference/run_inference.py --few_shot --model_name meta-llama/Llama-3.3-70B-Instruct
