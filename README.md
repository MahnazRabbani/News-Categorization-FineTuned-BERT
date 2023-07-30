# News Article Classification with Fine-Tuned BERT

This repository contains a multiclass classifies news articles into one of four categories of World, Sports, Business, and Sci/Tech using a fine-tuned BERT model. The project uses the AG News dataset available from the Hugging Face datasets library.



## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#Dataset)
- [Modles](#modles)
- [Results](#results)
- [Contributing](#contributing)

## Installation

Clone this repository and navigate to the downloaded directory. Then, install the necessary dependencies with:

```bash
pip install -r requirements.txt
 ```


## Usage

To run the project, follow these steps:

1. Clone the repository and navigate to the downloaded directory.

2. Install the necessary dependencies:

   ```bash
   pip install -r requirements.txt
   ```


4. - If you want to tweak and change the training process use the model training script:      
   cd to scr/models/ and open train.py
 
   - If you want to use the trained models load the models from `models/` directory and use it for making predictions on new data.

<!-- To visualize the results, run the visualization script:

```bash
cd ../visualization/
python plots.py
``` -->


## Project Structure  

**README.md**: this document, a general overview of the project.    
**requirements.txt**: the Python dependencies required for this project.     
**src/**: the main source code for the project. models/train.py is the final script for fine-tuning BERT.      
**notebooks/**: Jupyter notebooks for exploratory data analysis and initial model training, includes:      

- **text_analysis.ipynb**: comparative analysis of Naive Bayes and MLP classification algoritjms with different vectorization methods. 
- **initial_training.ipynb**: initial fine-tuning of BERT on the classification task. 
- **BERT_tune_on_data.ipynb**: fine-tuning of BERT on 1) data (domain adaptation), 2) on classification task , and 3) fine-tuning of domain adaptated BERT on the task.    

<!--  *data/**: The raw and processed datasets used for the project.    -->  
**models/**: the trained models are utilized to generate predictions. It's **important to note**, however, that due to file size restrictions on GitHub, these trained models aren't part of the GitHub project and are specified in the .gitignore file to be excluded.    
<!-- tests/: Unit tests for the project code.     
docs/: Additional project documentation.# News-Categorization-FineTuned-BERT    -->   


## Dataset

The dataset used in this project is the [AG News dataset](https://huggingface.co/datasets/ag_news) from Hugging Face. It consists of 120,000 training examples and 7,600 testing examples from four different categories: World, Sports, Business, and Sci/Tech. Each example includes a title and a description, both of which are used as input features for the classification model.

AG News is a collection of more than 1 million news articles. News articles have been gathered from more than 2000 news sources by ComeToMyHead in more than 1 year of activity. ComeToMyHead is an academic news search engine which has been running since July, 2004. The dataset is provided by the academic comunity for research purposes in data mining (clustering, classification, etc), information retrieval (ranking, search, etc), xml, data compression, data streaming, and any other non-commercial activity.

The data is available in English, as it is collected from English web pages. Each sample in the dataset is labeled with one of four categories:

- 0: World
- 1: Sports
- 2: Business
- 3: Sci/Tech

For more information about the dataset and its usage, refer to the [dataset card](https://huggingface.co/datasets/ag_news) on Hugging Face.


## Models

**Models that were trained:**    

Classic classification models:     

- Naive Bayes classifier using BoW
- Naive Bayes classifier with TF-IDF
- MLP in combination with TF-IDF
- MLP with BERT's feature extraction

BERT fine-tuned models:

- fine-tuning of a pretrained BERT model on the 'ag_news' dataset, a process known as domain adaptation.
- fine-tuning of a pretrained BERT on the news classification task.
- fine-tuning of the domain-adapted model on the classification task.

## Results

After fine-tuning the BERT model on the task, we initially achieved an accuracy of 0.885 on the sampled dataset. However, through additional improvements, we were able to enhance the accuracy to 0.886. This improvement was accomplished by first fine-tuning BERT on the dataset through domain adoption techniques, followed by further fine-tuning of the adjusted model using labeled data specific to the task.

It is worth noting that these training iterations were performed on a limited sample of 1000 datapoints without a dedicated GPU, due to computational constraints. Therefore, it is expected that the model's accuracy will further improve when trained on the full dataset. This step will be pursued in the future stages of the project.     


## Contributing

Your contributions are always welcome! If you have suggestions for improving this project, please open an issue describing the enhancement or bug you've identified. If you're able, make the necessary changes and submit a pull request. Any contributions you make are greatly appreciated!

