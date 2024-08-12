# Transformer-Based Question Answering Model

## Overview

This project implements a Transformer-based model for question answering (QA) using TensorFlow and a SentencePiece tokenizer. The model is trained on a small subset of the SQuAD dataset and utilizes a simplified version of Google's T5 architecture. Given a context and a question, the model generates a corresponding answer.

**Note:** The current model is trained for only 100 epochs using a small portion of the SQuAD dataset. This limited training results in suboptimal performance. For better results, the model should be trained further on a significantly larger dataset.

## Features

- **Natural Language Processing**: Tokenizes input text using a SentencePiece tokenizer.
- **Masked Language Modeling**: Masks tokens during training to allow the model to learn to predict missing words.
- **Custom Transformer Model**: Implements a Transformer architecture with custom layers and training routines.
- **Question Answering**: Uses the trained model to generate answers based on input questions and context.

## Project Structure

- **`data/`**: Contains the input datasets, including a subset of the SQuAD dataset and other training data.
- **`models/`**: Stores the pre-trained SentencePiece tokenizer and Transformer model weights.
- **`transformer_utils.py`**: A utility script containing custom layers, loss functions, and training routines for the Transformer model.
- **`app.py`**: A Streamlit application for interactive question answering using the trained model.
- **`train.py`**: Script for training and fine-tuning the Transformer model on the QA task.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- TensorFlow
- TensorFlow Text
- Streamlit
- Termcolor
- Numpy

You can install the required packages using pip:

```bash
pip install tensorflow tensorflow-text streamlit termcolor numpy
```

### Running the Streamlit App

To start the interactive question answering app, run:

```bash
streamlit run app.py
```

### Training the Model

If you wish to train or fine-tune the model:

1. Adjust the hyperparameters in the `train_model` or `finetune_model` functions in the script.
2. Run the training script:

```bash
python train.py
```

### Using the Model

After loading the model, you can use the `answer_question` function to generate answers based on the input question and context.

Example:

```python
question = "How old are you? I'm 4 years old"
result = answer_question(tf.constant(question), transformer, tokenizer)
print(pretty_decode(result, sentinels, tokenizer).numpy()[0])
```

## How It Works

1. **Text Tokenization**: The input texts are tokenized into subword units using the SentencePiece tokenizer.
2. **Masking and Training**: During training, a percentage of tokens are masked, and the model is trained to predict the masked tokens.
3. **Transformer Model**: The custom Transformer model is trained to generate answers based on the question-context pair.
4. **Inference**: At inference, the model takes a question-context input, processes it through the trained Transformer, and generates the most probable answer.

## Model Limitations

- **Limited Training**: The model is trained on a small portion of the SQuAD dataset for only 100 epochs, leading to less-than-ideal performance.
- **Training Time**: Proper training with a larger dataset and more epochs would require significant computational resources and time.

## Future Improvements

- **Extended Training**: Train the model on the full SQuAD dataset or other large QA datasets for improved performance.
- **Model Optimization**: Experiment with different Transformer architectures and hyperparameters to optimize the model further.
- **Deployment**: Deploy the model as an API or integrate it into a web application for broader use.

## Acknowledgments

This project is based on the T5 model architecture and utilizes data from the SQuAD dataset. Special thanks to the open-source community for providing tools and resources that made this project possible.
