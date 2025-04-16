from pathlib import Path

from src.config import dataset_name, working_directory_name


llama_factory_config_dir = Path("training_config")
base_config_file_path = llama_factory_config_dir / "base_config.yaml"


def get_yaml_file_path(model_name: str, dataset_name: str,
                       learning_rate: str) -> Path:
    return llama_factory_config_dir / \
        f"{model_name}_{dataset_name}_{learning_rate}.yaml"


def main():
    llama_factory_config_dir.mkdir(exist_ok=True)
    num_gpus = 2

    train_dataset_name = dataset_name

    # load config template
    with open(base_config_file_path, "r") as file:
        base_config = file.read()
    
    for model_name in ["meta-llama/Llama-3.1-8B-Instruct"]:
        # batch size
        per_device_batch_size = 4
        virtual_batch_size = 16
        
        gradient_accumulation_steps = \
            virtual_batch_size // (per_device_batch_size * num_gpus)
        
        # save path
        model_short_name = model_name.split("/")[-1]
        
        for learning_rate in ["1.0e-5"]:
            output_dir = f"../{working_directory_name}/llama_factory_finetuned_models/{model_short_name}_{train_dataset_name}_{learning_rate}"
            
            # fill in the config template
            config = base_config.format(
                model_name=model_name,
                dataset_name=train_dataset_name,
                output_dir=output_dir,
                per_device_train_batch_size=per_device_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
            )
            
            # save config to yaml file
            yaml_file_path = get_yaml_file_path(
                model_short_name, train_dataset_name,
                learning_rate=learning_rate
            )
            with open(yaml_file_path, "w") as file:
                file.write(config)


if __name__ == '__main__':
    main()
