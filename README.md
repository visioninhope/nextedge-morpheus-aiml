# nextedge-morpheus


# Edge Device Content Moderation using Fine-Tuned Phi-3 Model

This repository contains the code and resources for building an edge device content moderation system using a fine-tuned **Phi-3-mini-4k-instruct** model. The project involves generating synthetic data using OpenAI's **Meta-Llama-3.1-70B-Instruct-Turbo** model, fine-tuning the Phi-3 model on the generated data, and deploying the model on a local device with a Streamlit application for real-time text categorization.

## Table of Contents

- [Overview](#overview)
- [Step 1: Generation of Artificial Data](#step-1-generation-of-artificial-data)
- [Step 2: Fine-Tuning the Model](#step-2-fine-tuning-the-model)
- [Step 3: Deploying the Streamlit Application](#step-3-deploying-the-streamlit-application)
- [Screenshots](#screenshots)
- [Usage of AIML API](#usage-of-aiml-api)
- [Categories](#categories)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Conclusion](#conclusion)

## Overview

This project demonstrates the workflow of generating synthetic data, fine-tuning a machine learning model, and deploying it on an edge device using a Streamlit application. The entire workflow is designed to work without requiring internet access once the model is fine-tuned and deployed.

## Step 1: Generation of Artificial Data

The first step involves generating synthetic data using the **Meta-Llama-3.1-70B-Instruct-Turbo** model from OpenAI. For each prompt-category pair, the code generates 100 samples, storing the responses in a Pandas DataFrame. This DataFrame is then saved as a CSV file named `synthetic_data_500_samples_with_categoriesv2.1.csv`.

### Categories

The categories used for generating the synthetic data include:
- Positive
- Spam
- Neutral
- Misleading
- Inappropriate

The generated synthetic data is then used as the dataset for fine-tuning the Phi-3 model.

### Screenshot of Synthetic Data Creation

Below is a screenshot of the synthetic data generation process:

![Synthetic Data Creation](https://raw.githubusercontent.com/vikashkodati/nextedge-morpheus/main/media/aiml_api_synthetic.png)

The artificial data generation process was performed on Google Colab, and you can find the notebook used for this process [here](https://drive.google.com/file/d/1f3bASiMkHUyw64AResO7TW3ixaZYbV5Q/view?usp=sharing).

## Step 2: Fine-Tuning the Model

The synthetic data is loaded into a Google Colab environment, where the **Phi-3-mini-4k-instruct** model is fine-tuned. The fine-tuning process adjusts the model's parameters to improve its ability to categorize text according to the predefined categories.

After fine-tuning, the model is saved and downloaded to the local machine for deployment. The saved model includes the LoRA adapters, which are stored in the `trained-model` directory.

The fine-tuning process was performed on Google Colab, and you can find the notebook used for this process [here](https://drive.google.com/file/d/1RZIiol4OEjBxVbfVCDTzDF7YkWuL0zuh/view?usp=sharing).

### Accuracy Before Fine-Tuning

Before fine-tuning, the model's accuracy was evaluated on a validation dataset, yielding the following results:

![Accuracy Before Fine-Tuning](https://raw.githubusercontent.com/vikashkodati/nextedge-morpheus/main/media/acc_before_fine_tune.png)

### Fine-Tuning Process

The model was fine-tuned over multiple epochs, with training and validation losses recorded as shown below:

![Fine-Tuning Step](https://raw.githubusercontent.com/vikashkodati/nextedge-morpheus/main/media/fine_tuning_step.png)

### Accuracy After Fine-Tuning

After fine-tuning, the model's accuracy improved significantly, as shown in the following evaluation:

![Accuracy After Fine-Tuning](https://raw.githubusercontent.com/vikashkodati/nextedge-morpheus/main/media/acc_after_fine_tune.png)

## Step 3: Deploying the Streamlit Application

The final step involves deploying a Streamlit application that runs locally without requiring internet access. The app uses the fine-tuned model to analyze user-input text and detect its category in real-time.

### Streamlit Interface Screenshots

The following screenshots demonstrate the Streamlit application classifying various text inputs into categories. Note that the internet was off during all these tests, showcasing the model's capability to function fully offline.

#### Misleading Text Category

![Misleading Text Category](https://raw.githubusercontent.com/vikashkodati/nextedge-morpheus/main/media/Misleading.png)

##### Play Demo Video
[![Misleading](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1-DhjDn3mMvp21-N3YjPdr9dNFhFOKnCz)](https://drive.google.com/file/d/1-DhjDn3mMvp21-N3YjPdr9dNFhFOKnCz/view?usp=sharing "Misleading")

#### Positive Text Category

![Positive Text Category](https://raw.githubusercontent.com/vikashkodati/nextedge-morpheus/main/media/Positive.png)

##### Play Demo Video
[![Positive](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1-FT3H39Pbv_Oz8J0KdsQAm33BAId5dqU)](https://drive.google.com/file/d/1-FT3H39Pbv_Oz8J0KdsQAm33BAId5dqU/view?usp=sharing "Positive")

#### Spam Text Category

![Spam Text Category](https://raw.githubusercontent.com/vikashkodati/nextedge-morpheus/main/media/Spam.png)

##### Play Demo Video
[![Spam](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1s9oH-VKM5nSNoFtVqeKc6amssggTjn6C)](https://drive.google.com/file/d/1s9oH-VKM5nSNoFtVqeKc6amssggTjn6C/view?usp=sharing "Spam")

## Usage of AIML API

The AIML API is utilized in Step 1 to generate synthetic data using the **Meta-Llama-3.1-70B-Instruct-Turbo** model. The generated data is crucial for fine-tuning the Phi-3 model in Step 2.

## Categories

The following categories are detected by the fine-tuned model:
- Positive
- Spam
- Neutral
- Misleading
- Inappropriate

## Requirements

- Python 3.7+
- Hugging Face Transformers
- Peft
- Pandas
- Streamlit

## Installation

1. Clone this repository:

```bash
git clone https://github.com/vikashkodati/nextedge-morpheus
```

2. Navigate to the project directory:

```bash
cd nextedge-morpheus
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Application

To run the Streamlit application locally, use the following command:

```bash
cd streamlit_app
streamlit run app.py
```

This will launch the web application on your local machine, allowing you to enter text and receive real-time category detection results.

## Conclusion

This project showcases the complete workflow from generating synthetic data to fine-tuning a model and deploying it on an edge device. The Streamlit application offers a user-friendly interface for real-time text categorization, and the entire setup runs locally without requiring an internet connection.


---

<!-- ## Old readme

Synthetic Data Generation Module:

This notebook automates the generation of synthetic data for various categories of social media posts, which could be useful for training or testing machine learning models.

Initialization:

The OpenAI client is initialized using an API key and a base URL (api.aimlapi.com).

Prompts and Categories:

A list of prompts with corresponding categories is defined, such as generating a positive social media post, a spammy comment, a neutral statement, and a misleading news headline.

Synthetic Data Generation:

For each prompt-category pair, the code generates 100 samples using the meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo model from OpenAI.
The responses generated by the model are stored as text along with their associated category.

Data Storage:

The generated synthetic data is stored in a Pandas DataFrame, which is then saved to a CSV file named synthetic_data_500_samples_with_categoriesv2.1.csv.

Output:

The script prints progress messages to indicate the generation of each sample and confirms when the data is saved to the CSV file.
 -->