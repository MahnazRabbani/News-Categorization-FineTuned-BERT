# News Article Classification with Fine-Tuned BERT

This repository contains a machine learning project that classifies news articles into one of four categories (World, Sports, Business, Sci/Tech) using a fine-tuned BERT model. The project uses the AG News dataset available from the Hugging Face datasets library.



## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#Dataset)
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


4. Run the model training script:

   ```bash
   cd ../models/
   python bert_classifier.py
   ```

The trained models are saved in the `models/trained_models/` directory and can be used for making predictions on new data.

To visualize the results, run the visualization script:

```bash
cd ../visualization/
python plots.py
```


## Project Structure  

README.md: This document, a general overview of the project.    
requirements.txt: The Python dependencies required for this project.     
src/: The main source code for the project.      
notebooks/: Jupyter notebooks for exploratory data analysis and model training.     
data/: The raw and processed datasets used for the project.      
models/: The trained models.      
tests/: Unit tests for the project code.      
docs/: Additional project documentation.# News-Categorization-FineTuned-BERT      


## Dataset

The dataset used in this project is the [AG News dataset](https://huggingface.co/datasets/ag_news) from Hugging Face. It consists of 120,000 training examples and 7,600 testing examples from four different categories: World, Sports, Business, and Sci/Tech. Each example includes a title and a description, both of which are used as input features for the classification model.

AG News is a collection of more than 1 million news articles. News articles have been gathered from more than 2000 news sources by ComeToMyHead in more than 1 year of activity. ComeToMyHead is an academic news search engine which has been running since July, 2004. The dataset is provided by the academic comunity for research purposes in data mining (clustering, classification, etc), information retrieval (ranking, search, etc), xml, data compression, data streaming, and any other non-commercial activity.

The data is available in English, as it is collected from English web pages. Each sample in the dataset is labeled with one of four categories:

- 0: World
- 1: Sports
- 2: Business
- 3: Sci/Tech

For more information about the dataset and its usage, refer to the [dataset card](https://huggingface.co/datasets/ag_news) on Hugging Face.

## Results



## Contributing

Your contributions are always welcome! If you have suggestions for improving this project, please open an issue describing the enhancement or bug you've identified. If you're able, make the necessary changes and submit a pull request. Any contributions you make are greatly appreciated!

