import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM

# Configurable Variables
TRAINING_FILE = "ShakeGen_Training.txt"
MODEL_FILE = "ShakeSphereGenA.h5"
EPOCHS = 64
BATCH_SIZE = 256
LAYER_COUNT = 6  # Number of Dense layers

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / temperature  # Prevent log(0)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.argmax(np.random.multinomial(1, preds, 1))

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = sentence = text[start_index: start_index + SEQ_LENGTH]
    
    for _ in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(char_to_index)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_char = index_to_char[sample(predictions, temperature)]

        generated += next_char
        sentence = sentence[1:] + next_char

    return generated

with open(TRAINING_FILE, 'rb') as f:
    text = f.read().decode(encoding='utf-8').lower()

characters = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i: c for i, c in enumerate(characters)}

SEQ_LENGTH = 40
STEP_SIZE = 3

sentences, next_char = [], []
for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_char.append(text[i + SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.uint8)
y = np.zeros((len(sentences), len(characters)), dtype=np.uint8)

for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_char[i]]] = 1

tf.config.run_functions_eagerly(True)

try:
    model = load_model(MODEL_FILE)
except:
    model = Sequential()
    model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
    
    for _ in range(LAYER_COUNT):
        model.add(Dense(len(characters), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01), metrics=['accuracy'])
    model.fit(x, y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2)
    model.save(MODEL_FILE)

print(generate_text(300, 0.2))
print(generate_text(300, 0.4))
print(generate_text(300, 0.5))
print(generate_text(300, 0.6))
print(generate_text(300, 0.7))
