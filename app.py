import streamlit as st
import tensorflow as tf
from t5_question_answerer import answer_question, pretty_decode, load_tokenizer, get_sentinels, initialize_model

# Streamlit app interface
st.title("Question Answering with T5 Model")

# Warning message about model training
st.warning("Note: The model used in this app is trained for only 100 epochs on a small chunk of the SQuAD dataset. "
           "This limited training is not sufficient for perfect performance. For better results, the model should be "
           "trained further with a significantly larger dataset to achieve high-quality performance.")

# Create a placeholder for loading messages
loading_message = st.empty()

# Display a loading message while initializing the model and tokenizer
loading_message.text('Loading model and tokenizer... Please wait.')

# Initialize the model and tokenizer
tokenizer = load_tokenizer()
sentinels = get_sentinels(tokenizer)
transformer = initialize_model(tokenizer)

# Load the model weights
transformer.load_weights('models/t5/model_qa3')

# Update the placeholder to indicate that loading is complete
loading_message.text('Model loaded successfully!')

# Input for the question
question = st.text_input("Enter your question", "How old are you?")

# Input for the context
context = st.text_area("Enter the context", "I'm 4 years old")

# Button to trigger the model inference
if st.button("Get Answer"):
    # Update the message to indicate processing
    loading_message.text('Generating answer...')

    question_context = tf.constant(question + context)

    # Get the answer from the model
    answer = answer_question(question_context, transformer, tokenizer)
    answer = pretty_decode(answer, sentinels, tokenizer).numpy()[0]

    # Display the answer and clear the loading message
    st.subheader("Answer")
    st.write(answer)
    loading_message.empty()  # Clear the loading message
