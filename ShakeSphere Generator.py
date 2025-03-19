import random
import numpy as np
import tensorflow as tf 
#model import
from tensorflow.keras.models import Sequential
#optimizer import
from tensorflow.keras.optimizers import RMSprop
#Layers import
from tensorflow.keras.layers import Activation, Dense, LSTM
#import model save api
from tensorflow.keras.models import load_model

#function for sampling from probability distribution
#preds(predictions) is a probability distribution, one dimensional array
#temperature controls the level of randomness, higher temperature (e.g. >1.0)means more random
def sample(preds, temperature=1.0):
    #convberts preds into a float64array
    preds = np.asarray(preds).astype('float64')
    #applies a log transformation to the predicitons and divides them by the temperature
    #this conversion to log and back increases numerical stability (less likely for numerical underflow), mathematical convenience(multipling probs in log becomes addition), gradient descent (gradient descent involves calculation derivatives and log makes these calculations straightforward), probability interpretation (logs are commonly used in softmax)
    preds = np.log(preds) / temperature
    #exponentiates the adjusted probs, bring them back to o.g state
    exp_preds = np.exp(preds)
    #normalizes probs to make them add up to 1.0, turning them into valid probability distrubutuion
    preds = exp_preds / np.sum(exp_preds)
    #performes actual sampling using multinomial distribution
    probas = np.random.multinomial(1, preds, 1)
    #returns the indes of the max value in sampled array
    return np.argmax(probas)

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(char_to_index)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1
        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions, temperature)

        next_char = index_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char
    return generated

def InputParams():
    Verbose = input('verbose?')
    Epochs = input('Epochs?')
    BatchSize = input('batch Size?')

    #downloads the file
    #filepath = tf.keras.utils.get_file('ShakeSphere_Training.txt')
text = open('ShakeGen_Training.txt', 'rb').read().decode(encoding='utf-8').lower()
    #rb means to read as binary, while for text r will be enough, rb is still better to ensure correct decoding
    #print(text)

    #set() turns text into an unordered collection of unique elements, deletes repetitive elements.
    #sorted sorts the set into alphabetical order
characters = sorted(set(text))

    #these lines convert characters to indexes and vice versa
    #maps each character (c) to their respective indexes(i)
char_to_index = dict((c, i)for i, c in enumerate(characters))
    #maps each index to their respective characters
index_to_char = dict((i, c)for i, c in enumerate(characters))

    #defines the length of each sequence, ie the number of characters to look back on 
SEQ_LENGTH = 40
    #defines the size of each step, ie the number of characters sach sequence moves by
STEP_SIZE = 3

sentences = []
next_char = []

    #creates sequences of Input target characters
    #the range starts at zero and ends at the length of the text minus sequence length, and each step size is step_size
for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    #generates sequence sentences with elements length of seq_length 
    sentences.append(text[i: i+SEQ_LENGTH])
    #generates sequence next_char
    next_char.append(text[i+SEQ_LENGTH])


x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool_)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool_)

    #one-hot encodes the arrays for x and y
for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):    
        #one-hot encodes input sequence,  i is the corresponding index to the current sample in the dataset,  t represents the position of the character withen the sequence (range 0 ~ Seq_length -1)  
        #assignment sets the value at specified indices to 1, indicates presentce of character at specific position in sequence
        x[i, t, char_to_index[character]] = 1
    #one-hoit encodes the target sequence,  i correspondes to the current dataset sample (sequence of chars)
    #next_char[i] accesses the next character in sequence, target access, char_to_index[next_char[i]] maps target char to index
    y[i, char_to_index[next_char[i]]] = 1

tf.config.run_functions_eagerly(True)
try:
    model = tf.keras.models.load_model('ShakeSphereGenA.h5')

except:
    #initiates a sequetial model //  you can add layers one by one in a stack
    model = Sequential()
    #creates LSTM layer with 128 neurons  Input_shape defines the shape of the input data, 
    model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
    #creates dense layer with neurons equal to characters
    model.add(Dense(len(characters)))
    model.add(Dense(len(characters)))
    model.add(Dense(len(characters)))
    model.add(Dense(len(characters)))
    model.add(Dense(len(characters)))
    model.add(Dense(len(characters)))
    #uses a softmax function to produce probability scores for each character, adds up all to one
    model.add(Activation('softmax'))
    model.summary()
    #compiles the model using categorical crossentropy as the loss function and RMSprop as the optimizer with a learning rate of 0.01
    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(learning_rate=0.01),
        metrics = ['accuracy'])

    #Training
    #fits (trains) the model using x as input data and y as target data each batch is 256 and it iterates 4 times Verbose refers to the level;
    model.fit(x, y, batch_size=8192, epochs=1024, verbose = 2)
    model.evaluate(x, y, batch_size=12, verbose = 2)

    model.save('ShakeSphereGenA.h5')

print(generate_text(300, 0.2))
print(generate_text(300, 0.4))
print(generate_text(300, 0.5))
print(generate_text(300, 0.6))          
print(generate_text(300, 0.7))