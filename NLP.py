import keras as ks
import numpy as np
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences


def data():
    dataSet = {
        "Hi": "Hello",
        "How are you": "I'm fine, thanks for asking",
        "How was your day": "It was fine, thank you",
        "What's your name?": "I'm a robot",
        "What do you do?": "I chat with people",
        "Good morning": "Good morning! How can I assist you today?",
        "Good night": "Good night! Sweet dreams",
        "What time is it?": "I don't have a clock, but you can check your device",
        "Where are you from?": "I'm from the internet",
        "Tell me a joke": "Why don't scientists trust atoms? Because they make up everything!",
        "What's your favorite color?": "I like all the colors equally",
        "Help": "Sure, how can I assist you?",
        "Thank you": "You're welcome",
        "Bye": "Goodbye! Have a great day",
        "Can you help me?": "Of course, what do you need help with?",
        "What's the weather like?": "I'm not sure, but you can check your weather app",
        "Do you like music?": "I don't have preferences, but I can tell you about music",
        "What's the capital of France?": "The capital of France is Paris",
        "Are you human?": "No, I'm a robot",
        "What's 2+2?": "2+2 is 4",
        "Open the door": "I'm afraid I can't do that",
        "What is love?": "Love is a complex mix of emotions, behaviors, and beliefs",
        "Tell me a fun fact": "Did you know honey never spoils?"
    }
    return dataSet





# Prepare data
dataSet = data()
sentences = list(dataSet.keys())
responses = list(dataSet.values())

# Tokenize the sentences
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(sentences + responses)
sequences = tokenizer.texts_to_sequences(sentences)
response_sequences = tokenizer.texts_to_sequences(responses)

# Pad the sequences
max_length = max(max(len(seq) for seq in sequences), max(len(seq) for seq in response_sequences))
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
padded_response_sequences = pad_sequences(response_sequences, maxlen=max_length, padding='post')

# Convert responses to numpy array
response_sequences = np.array(padded_response_sequences)

# Define the model
model = ks.Sequential([
    ks.layers.Embedding(input_dim=1000, output_dim=64, input_length=max_length),
    ks.layers.SimpleRNN(64, return_sequences=True),
    ks.layers.SimpleRNN(64),
    ks.layers.Dense(64, activation='relu'),
    ks.layers.Dense(max_length, activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Prepare the target data for training
target_data = np.expand_dims(response_sequences.argmax(axis=-1), axis=-1)

# Train the model
model.fit(padded_sequences, target_data, epochs=200, verbose=1)

# Function to get response


def get_response(sentence):
    # Tokenize and pad the input sentence
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    # Predict the response
    predicted_response_seq = model.predict(padded_sequence)
    predicted_response_idx1 = np.argmax(predicted_response_seq, axis=-1)
    print(predicted_response_idx1)
    

    # Convert the predicted indices back to words
    reverse_word_index = {value: key for key, value in tokenizer.word_index.items()}

    predicted_response = ' '.join([reverse_word_index.get(idx, '') for idx in predicted_response_idx1 if idx > 0])

    return predicted_response


# Test the response mechanism
test_sentences = [
    "Hey",
    "Hello",
    "Hi",
    "How are you",
    "How was your day",
    "What's your name?",
    "Tell me a joke",
    "What is love?",
    "Good night",
    "Can you help me?",
    "What's the capital of France?",
    "What's 2+2?"
]

for sentence in test_sentences:
    response = get_response(sentence)
    print(f"Input: {sentence}\nResponse: {response}\n")
