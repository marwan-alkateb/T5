import os
import time
import json
from termcolor import colored
import string
import textwrap
import numpy as np
import tensorflow_text as tf_text
import tensorflow as tf
import transformer_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# maximum line width of 70 characters.
wrapper = textwrap.TextWrapper(width=70)
np.random.seed(42)

ENCODER_MAXLEN = 150
DECODER_MAXLEN = 50


def load_natural_language_texts(file='data/t5/c4-en-10k.jsonl'):
    with open(file, 'r') as f:  # r: read mode
        example_jsons = [json.loads(line.strip()) for line in f]
    # Grab text field from dictionary
    return [example_json['text'] for example_json in example_jsons]


def load_tokenizer(file="models/sentencepiece.model"):
    # The SentencePieceTokenizer, used in the code snippet, tokenizes text into subword units,
    # enhancing handling of complex word structures, OOV words, & multilingual support.
    with open(file, "rb") as f:
        pre_trained_tokenizer = f.read()
    return tf_text.SentencepieceTokenizer(pre_trained_tokenizer, out_type=tf.int32)


def get_sentinels(tokenizer, display=False):
    sentinels = {}
    for i, char in enumerate(reversed(string.ascii_letters), start=1):
        decoded_text = tokenizer.detokenize([tokenizer.vocab_size() - i]).numpy().decode("utf-8")
        sentinels[decoded_text] = f'<{char}>'  # Sentinels, ex: <Z> - <a>
        if display:
            print(f'The sentinel is <{char}> and the decoded token is:', decoded_text)
    return sentinels


def pretty_decode(str_tf, sentinels, tokenizer):
    if not tf.is_tensor(str_tf) or str_tf.dtype != tf.string:
        return pretty_decode(tokenizer.detokenize(str_tf), sentinels, tokenizer)
    for token, char in sentinels.items():
        str_tf = tf.strings.regex_replace(str_tf, token, char)
    return str_tf


def tokenize_and_mask(text, noise=0.15, randomizer=np.random.uniform, tokenizer=None):
    """
    Tokenizes input text and applies masking based on a weighted coin flip.

    Args:
        text (str or bytes): Text input.
        noise (float, optional): Probability of masking a token. Defaults to 0.15.
        randomizer (function, optional): Function that generates a random value between 0 and 1.
        tokenizer (function, optional): Tokenizer function. Defaults to tokenize.

    Returns:
        inputs, targets: Lists of integers associated with inputs and targets.
    """
    cur_sentinel_num = 0
    inputs, targets = [], []
    vocab_size = int(tokenizer.vocab_size())
    eos = tokenizer.string_to_id("</s>").numpy()  # EOS token ID (end of sequence)
    prev_no_mask = True  # True if the previous token was NOT masked

    for token in tokenizer.tokenize(text).numpy():
        if noise > randomizer():  # Apply masking with probability 'noise'
            if prev_no_mask:
                cur_sentinel_num += 1
                sentinel_id = vocab_size - cur_sentinel_num
                inputs.append(sentinel_id)
                targets.append(sentinel_id)
            targets.append(token)
            prev_no_mask = False
        else:  # no masking
            inputs.append(token)
            prev_no_mask = True

    targets.append(eos)  # Add EOS token to the end of targets
    return inputs, targets


# # Tokenize the sentence
# input_text = "there is a cat on the mat in the house and it is playing with the ball"
# inputs, targets = tokenize_and_mask(input_text, noise=0.15, tokenizer=tokenizer)
# print("Input tokens:", inputs)
# print("Target tokens:", targets)
# # Input tokens: [132, 19, 3, 9, 1712, 30, 31999, 6928, 16, 8, 31998, 11, 34, 19, 1556, 28, 8, 1996]
# # Target tokens: [31999, 8, 31998, 629, 1] : [sentinel1, token1, sentinel2, token2, <eos>]
# sys.exit("HERE  WE STOP")


def initialize_model(tokenizer,
                     num_layers=2,
                     embedding_dim=128,
                     fully_connected_dim=128,
                     num_heads=2,
                     positional_encoding_length=256,
                     ):
    """
    Initialize the Transformer model.

    Args:
        tokenizer (tf_text.SentencepieceTokenizer): The tokenizer used for tokenizing text.
        model_params (dict): A dictionary containing model parameters.

    Returns:
        model (transformer_utils.Transformer): The initialized Transformer model.
    """
    vocab_size = int(tokenizer.vocab_size())

    transformer = transformer_utils.Transformer(
        num_layers,
        embedding_dim,
        num_heads,
        fully_connected_dim,
        vocab_size,
        vocab_size,
        positional_encoding_length,
        positional_encoding_length,
    )

    return transformer


natural_language_texts = load_natural_language_texts()
tokenizer = load_tokenizer()
sentinels = get_sentinels(tokenizer, display=True)
print(pretty_decode(tf.constant("I want to dress up as an Intellectual this halloween."), sentinels, tokenizer))
# Apply tokenize_and_mask
inputs_targets_pairs = [tokenize_and_mask(text.encode('utf-8', errors='ignore').decode('utf-8'), tokenizer=tokenizer)
                        for text in natural_language_texts[0:2000]]
transformer = initialize_model(tokenizer)
learning_rate = transformer_utils.CustomSchedule(transformer.get_embedding_dim())
optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
train_loss = tf.keras.metrics.Mean(name='train_loss')
# Here you will store the losses, so you can later plot them
losses = []

inputs = tf.keras.preprocessing.sequence.pad_sequences([x[0] for x in inputs_targets_pairs], maxlen=ENCODER_MAXLEN,
                                                       padding='post', truncating='post')
targets = tf.keras.preprocessing.sequence.pad_sequences([x[1] for x in inputs_targets_pairs], maxlen=DECODER_MAXLEN,
                                                        padding='post', truncating='post')

inputs = tf.cast(inputs, dtype=tf.int32)
targets = tf.cast(targets, dtype=tf.int32)

# Create the final training dataset.
BUFFER_SIZE = 10000
BATCH_SIZE = 64


def train_model(epochs=1, ):
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # Training loop
    for epoch in range(epochs):

        start = time.time()
        train_loss.reset_states()
        number_of_batches = len(list(enumerate(dataset)))

        for (batch, (inp, tar)) in enumerate(dataset):
            print(f'Epoch {epoch + 1}, Batch {batch + 1}/{number_of_batches}', end='\r')
            transformer_utils.train_step(inp, tar, transformer, loss_object, optimizer, train_loss)

        print(f'Epoch {epoch + 1}, Loss {train_loss.result():.4f}')
        losses.append(train_loss.result())

        print(f'Time taken for one epoch: {time.time() - start} sec')

        # Save the pretrained model
        transformer.save_weights('./model_c4_temp')


# Load a pretrained model
transformer.load_weights('models/t5/model_c4')

# TODO: Fine Tuning
with open('data/t5/train-v2.0.json', 'r') as f:
    example_jsons = json.load(f)

example_jsons = example_jsons['data']
print('Number of articles: ' + str(len(example_jsons)))

"""
The structure of each article is as follows:

title: The article title
paragraphs: A list of paragraphs and questions related to them
context: The actual paragraph text
qas: A set of question related to the paragraph
question: A question
id: The question unique identifier
is_imposible: Boolean, specifies if the question can be answered or not
answers: A set of possible answers for the question
text: The answer
answer_start: The index of the character that starts the sentence containing the explicit answer to the question
"""

example_article = example_jsons[0]
print("Title: " + example_article["title"])
print(example_article["paragraphs"][0])


def parse_squad(dataset):
    """Extract all the answers/questions pairs from the SQuAD dataset

    Args:
        dataset (dict): The imported JSON dataset

    Returns:
        inputs, targets: Two lists containing the inputs and the targets for the QA model
    """
    inputs, targets = [], []
    # Loop over all the articles
    for article in dataset:
        # Loop over each paragraph of each article
        for paragraph in article['paragraphs']:
            # Extract context from the paragraph
            context = paragraph['context']
            # Loop over each question of the given paragraph
            for qa in paragraph['qas']:
                # If this question is not impossible and there is at least one answer
                if len(qa['answers']) > 0 and not (qa['is_impossible']):
                    # Create the question/context sequence
                    question_context = 'question: ' + qa['question'] + ' context: ' + context
                    # Create the answer sequence. Use the text field of the first answer
                    answer = 'answer: ' + qa['answers'][0]['text']
                    # Add the question_context to the inputs list
                    inputs.append(question_context)
                    # Add the answer to the targets list
                    targets.append(answer)
    return inputs, targets


inputs, targets = parse_squad(example_jsons)
print("Number of question/answer pairs: " + str(len(inputs)))
print('\nFirst Q/A pair:\n\ninputs: ' + colored(inputs[0], 'blue'))
print('\ntargets: ' + colored(targets[0], 'green'))
print('\nLast Q/A pair:\n\ninputs: ' + colored(inputs[-1], 'blue'))
print('\ntargets: ' + colored(targets[-1], 'green'))

# You will use 40000 samples for training and 5000 samples for testing
# 40K pairs for training
inputs_train = inputs[0:40000]
targets_train = targets[0:40000]
# 5K pairs for testing
inputs_test = inputs[40000:45000]
targets_test = targets[40000:45000]

inputs_str = [tokenizer.tokenize(s) for s in inputs_train]
targets_str = [tf.concat([tokenizer.tokenize(s), [1]], 0) for s in targets_train]

inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs_str, maxlen=ENCODER_MAXLEN, padding='post',
                                                       truncating='post')
targets = tf.keras.preprocessing.sequence.pad_sequences(targets_str, maxlen=DECODER_MAXLEN, padding='post',
                                                        truncating='post')

inputs = tf.cast(inputs, dtype=tf.int32)
targets = tf.cast(targets, dtype=tf.int32)

# Create the final training dataset.
BUFFER_SIZE = 10000
BATCH_SIZE = 64
dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Define the number of epochs
losses = []


def finetune_model(epochs=100):
    """
    To get a model that works properly, you would need to train for about 100 epochs. This might take hours on a PC.
    This loop was already executed and model was saved at 'models/t5/model_qa3'
    """
    # Training loop
    for epoch in range(epochs):

        start = time.time()
        train_loss.reset_states()
        number_of_batches = len(list(enumerate(dataset)))

        for (batch, (inp, tar)) in enumerate(dataset):
            print(f'Epoch {epoch + 1}, Batch {batch + 1}/{number_of_batches}', end='\r')
            transformer_utils.train_step(inp, tar, transformer, loss_object, optimizer, train_loss)

        print(f'Epoch {epoch + 1}, Loss {train_loss.result():.4f}')
        losses.append(train_loss.result())

        print(f'Time taken for one epoch: {time.time() - start} sec')
        # if epoch % 15 == 0:
        #     transformer.save_weights('./pretrained_models/model_qa_temp')

        # Save the final model
        transformer.save_weights('./pretrained_models/model_qa_temp')


# Restore the weights
transformer.load_weights('models/t5/model_qa3')


def answer_question(question, model, tokenizer):
    """
    A function for question answering using the transformer model
    Arguments:
        question (tf.Tensor): Input data with question and context
        model (tf.keras.model): The transformer model
        tokenizer (function): The SentencePiece tokenizer
        ENCODER_MAXLEN (number): Max length of the encoded sequence
        DECODER_MAXLEN (number): Max length of the decoded sequence
    Returns:
        _ (str): The answer to the question
    """

    # Tokenize the question
    tokenized_question = tokenizer.tokenize(question)

    # Add an extra dimension to the tensor
    tokenized_question = tf.expand_dims(tokenized_question, 0)

    # Pad the question tensor
    padded_question = tf.keras.preprocessing.sequence.pad_sequences(tokenized_question,
                                                                    maxlen=ENCODER_MAXLEN,
                                                                    padding='post',
                                                                    truncating='post')
    padded_question = tf.cast(padded_question, dtype=tf.int32)

    # ANSWER SETUP

    # Tokenize the answer
    # Hint: All answers begin with the string "answer: "
    tokenized_answer = tokenizer.tokenize("answer: ")

    # Add an extra dimension to the tensor
    tokenized_answer = tf.expand_dims(tokenized_answer, 0)

    # Get the id of the EOS token
    eos = tokenizer.string_to_id("</s>").numpy()

    # Loop for DECODER_MAXLEN iterations
    for i in range(DECODER_MAXLEN):

        # Predict the next word using the model, the input document and the current state of output
        next_word = transformer_utils.next_word(padded_question, tokenized_answer, model)

        # Concat the predicted next word to the output
        tokenized_answer = tf.concat([tokenized_answer, next_word], axis=-1)

        # The text generation stops if the model predicts the EOS token
        if next_word == eos:
            break

    return tokenized_answer


idx = 10408
result = answer_question(inputs_train[idx], transformer, tokenizer)
print(colored(pretty_decode(result, sentinels, tokenizer).numpy()[0], 'blue'))
print()
print(inputs_train[idx])
print(colored(targets_train[idx], 'green'))

idx = 110
result = answer_question(inputs_test[idx], transformer, tokenizer)
print(colored(pretty_decode(result, sentinels, tokenizer).numpy()[0], 'blue'))
print()
print(inputs_test[idx])
print(colored(targets_test[idx], 'green'))

idx = 311
result = answer_question(inputs_test[idx], transformer, tokenizer)
print(colored(pretty_decode(result, sentinels, tokenizer).numpy()[0], 'blue'))
print()
print(inputs_test[idx])
print(colored(targets_test[idx], 'green'))



print(50*'_')
question = tf.constant("How old are you? I'm 4 years old")
result = answer_question(question, transformer, tokenizer)
print(colored(pretty_decode(result, sentinels, tokenizer).numpy()[0], 'blue'))
