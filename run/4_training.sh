conda activate cse587-training

python src/training/generate_yaml.py
python src/training/preprocess_dataset.py

cd ../LLaMA-Factory-CSE587
llamafactory-cli train ../cse587-final/training_config/Llama-3.1-8B-Instruct_cse587spring2025final_1.0e-5.yaml
