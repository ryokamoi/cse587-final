Spring 2025 CSE 587 Final Project

Please refer to [our report](./report.pdf) for the details of our project.

## Members

* Jiamu Bai
* Ryo Kamoi
* Divya Navuluri

## Environment

We use Anaconda to manage our environment. You can reproduce our environment by running [run/1_setup.sh](./run/1_setup.sh).

## Run Code

Please refer to the [run](./run) directory for the scripts to run the code. The scripts are organized as follows:

* [1_setup.sh](./run/1_setup.sh): Set up the environment.
* [2_download_abstracts.sh](./run/2_download_abstracts.sh): Download the abstracts from the ACL venues.
* [3_extract_research_questions.sh](./run/3_extract_research_questions.sh): Extract the research questions from the abstracts (dataset creation).
* [4_training.sh](./run/4_training.sh): Train the models.
* [5_evaluation.sh](./run/5_evaluation.sh): Evaluate the models.
