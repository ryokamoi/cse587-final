# This is the code for making jsonl files from the downloaded abstracts


import json
import random

from src.path import downloaded_abstracts_dir, abstracts_dir


downloaded_directory_name = {
    "train": "2023-Train set",
    "test": "2024-Test set",
}


def main():
    abstracts_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "test"]:
        directory_name = downloaded_directory_name[split]
        directory = downloaded_abstracts_dir / directory_name

        # all files in the directory
        # sort to make sure the order is reproducible
        file_names_list = sorted(list(directory.glob("*.txt")))

        file_names = []
        urls = []
        abstracts = []
        for file_name in file_names_list:
            with open(file_name, "r") as f:
                lines = f.readlines()
            
            # extract urls
            for line in lines:
                if "URL: " in line:
                    urls.append(line.split("URL: ")[1].strip())
            
            # extract abstract
            for line in lines:
                if "Abstract: Abstract" in line:
                    abstracts.append(line.split("Abstract: Abstract")[1].strip())
            
            # check the number of urls and abstracts
            if len(urls) != len(abstracts):
                raise ValueError(f"Number of urls and abstracts do not match in {directory_name}")
            
            file_names.extend([file_name.stem] * len(urls))
        
        output = []
        # save to jsonl
        for idx in range(len(urls)):
            output.append(
                {
                    "id": f"{split}_{idx:06}",
                    "source": str(file_names[idx]),
                    "url": urls[idx],
                    "abstract": abstracts[idx],
                }
            )
        
        # shuffle
        output = random.Random(42).sample(output, len(output))

        # save to jsonl
        output_path = abstracts_dir / f"{split}.jsonl"
        with open(output_path, "w") as f:
            for line in output:
                f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
