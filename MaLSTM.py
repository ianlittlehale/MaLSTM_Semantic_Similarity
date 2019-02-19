from tensorflow import keras
import tensorflow.keras.backend as kb
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import csv

class DataTools:

    def __init__(self):
        self.tokenizer = keras.preprocessing.text.Tokenizer()

    def get_data(self):
        with open("data/data.csv") as csvfile:
            csvreader = csv.reader(csvfile)
            rows = [row[3:] for row in csvreader]
        all_x = []
        self.x1 = []
        self.x2 = []
        self.Y = []
        for row in rows[1:]:
            all_x += [row[0].lower(), row[1].lower()]
            self.x1.append(row[0].lower())
            self.x2.append(row[1].lower())
            self.Y.append(int(row[2]))
        self.tokenizer.fit_on_texts(all_x)


    def train_test_split(self, test_size):
        xtrain, xtest, ytrain, ytest = train_test_split(list(zip(self.x1, self.x2)), self.Y, test_size=test_size)
        return xtrain, xtest, ytrain, ytest

    def text_to_seqs(self, X):
        x1 = []
        x2 = []
        for i in X:
            x1.append(i[0])
            x2.append(i[1])
        X1 = self.tokenizer.texts_to_sequences(x1)
        X1 = keras.preprocessing.sequence.pad_sequences(X1, maxlen=300, padding="post")
        X2 = self.tokenizer.texts_to_sequences(x2)
        X2 = keras.preprocessing.sequence.pad_sequences(X2, maxlen=300, padding="post")

        return X1, X2

    def make_embedding(self):
        word_index = self.tokenizer.word_index
        with open("GloVe/glove.6B.100d.txt", "r") as file:
            self.glove = {}
            for line in file:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype="float32")
                self.glove[word] = vector
        embedding_matrix = np.zeros(shape=(len(word_index)+1, 100))
        for word, index in word_index.items():
            if word in self.glove:
                if self.glove[word] is not None:
                    embedding_matrix[index] = self.glove[word]
        np.save("GloVe/embedding_matrix.np", embedding_matrix)

        return embedding_matrix

    def load_embedding(self):
        embedding_matrix = np.load("GloVe/embedding_matrix.npy")
        return embedding_matrix


class Similarity:

    def __init__(self,embedding_matrix, layer_dim):
        input1 = keras.layers.Input(shape=(300,))
        input2 = keras.layers.Input(shape=(300,))
        embedding = keras.layers.Embedding(95597, 100, weights = [embedding_matrix], trainable=False)
        lstm_input1 = embedding(input1)
        lstm_input2 = embedding(input2)
        lstm = keras.layers.LSTM(layer_dim)
        output1 = lstm(lstm_input1)
        output2 = lstm(lstm_input2)
        exp_neg_manhatten_distance = lambda x1, x2: kb.exp(-kb.sum(kb.abs(x1-x2), axis=1, keepdims=True))
        malstm_distance = keras.layers.Lambda(function=lambda x: exp_neg_manhatten_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))
        final_output = malstm_distance([output1, output2])
        self.model = keras.models.Model([input1, input2],final_output)
        self.model.summary()
        self.model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.RMSprop(), metrics=["accuracy"])

    def train(self, x1, x2, y, batch_size, n_epochs, validation_split):
        self.history = self.model.fit([x1, x2], y, batch_size=batch_size, epochs=n_epochs, validation_split=validation_split)





tools = DataTools()
tools.get_data()
xtrain,xtest,ytrain,ytest = tools.train_test_split(test_size=0.7)
X1train, X2train = tools.text_to_seqs(xtrain)
# X1test, X2test = tools.text_to_seqs(xtest)
embedding_matrix = tools.load_embedding()


s = Similarity(embedding_matrix, 128)
s.train(X1train, X2train, ytrain, batch_size=346, n_epochs=5, validation_split=0.25)
