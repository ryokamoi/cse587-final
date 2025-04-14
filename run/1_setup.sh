export PYTHONPATH="${PYTHONPATH}:./:../cse587-final"

###
# Environment for inference
conda env create -f environments/inference_environment.yml
conda activate cse587-inference

# huggingface
pip install torch==2.4 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.47.1

# MAX_JOBS=4 pip install flash-attn==2.7.2.post1 --no-build-isolation

# accelerate inference
pip install vllm==0.8.2

conda deactivate

###
# Environment for training
conda env create -f environments/training_environment.yml
conda activate cse587-training

# install LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git ../LLaMA-Factory-CSE587
cd ../LLaMA-Factory-FoVer

git checkout 4a5d0f0
pip install -e ".[torch,metrics]"

cd ../FoVer

# additional dependencies
pip install deepspeed==0.16.4
MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation

conda deactivate
