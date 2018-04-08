
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
import os
#os.environ['KERAS_BACKEND']='theano'
#import theano.tensor as T
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import optimizers
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Conv1D, Multiply
from keras.layers import LSTM, Add
from keras.models import Model
from keras import backend as K
import tensorflow as tf
from keras.initializers import RandomUniform, Constant
from keras.constraints import non_neg
import random as rn
from keras.layers.advanced_activations import ThresholdedReLU
from keras.constraints import max_norm, unit_norm, non_neg, Constraint


#https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1234)
tf.set_random_seed(1234)
rn.seed(1234)


batch_size = 10#50
epochs = 100
VOCAB = ['a','c','g','t']
split_ratio = [0.8, 0.1, 0.1]#[0.7, 0.15, 0.15]
N_LSTM = 10
filters = 100


##########################################################
# DEFINING FUNCTION USED FOR PRE-PROCESSING OF INPUT DATA
##########################################################

def read_fasta_file_and_prepare_data(fasta_file):

    """
    This function extracts sequences from fasta files
    """
    seq_data = []
    ids = []
    with open(fasta_file,"r") as f:
        for line in f:
            line = line.strip().lower()
            if line[0] != ">":
                seq_data.append(line)
            else:
                tmp = line[::-1]
                tmp = tmp[:tmp.index('_')]
                tmp = tmp[::-1]
                ids.append(tmp)
    return seq_data, ids




def encode_seqs(input_a, input_b, vocab, freq):
    """
    This function onehot embeds the sequences
    """

    inputs = input_a + input_b
    y_out = np.asarray([[1]] * len(input_a) + [[0]] * len(input_b))

    X_out = []
    for input in inputs:
        x = np.zeros((len(input),len(vocab)))

        for i in range(len(input)):
            x[i][vocab.index(input[i])] += freq[vocab.index(input[i])]#1
        X_out.append(x.flatten())

    X_out = np.asarray(X_out)
    X_out.reshape((len(inputs), 1, len(inputs[0])*len(vocab) )).astype('float32')
    return X_out, y_out




def split_data(seqs, ratios):
    """
    This function splits the input data into training, validation and test sets.
    """
    assert ratios[0] + ratios[1] < 1.0

    ntrain = int(round(ratios[0] * len(seqs)))
    nval = int(round(ratios[1] * len(seqs)))

    train = seqs[:ntrain]
    val = seqs[ntrain:ntrain+nval]
    test = seqs[ntrain+nval:]

    return train, val, test





def shuffle_seqs(seqs):
    """
    This function shuffles the inserted sequences and outputs a list
    with new shuffled sequences.
    """
    sf_sq = []
    for i in seqs:
        tmp = list(i)
        rn.shuffle(tmp)
        tmp = ''.join(tmp)
        sf_sq.append(tmp)

    return sf_sq







##########################################################
# IMPORTING DATA
##########################################################
print('Importing the FASTA file dna_TCCCACAAAC_2000_100.fa')
seqs, sids = read_fasta_file_and_prepare_data('dna_TCCCACAAAC_2000_100.fa')
bkg = shuffle_seqs(seqs)

a = 0
c = 0
g = 0
t = 0

for i in range(len(seqs)):
    a += seqs[i].count('a')
    c += seqs[i].count('c')
    g += seqs[i].count('g')
    t += seqs[i].count('t')

freq = np.array([a, c, g, t]) / float(a + c + g + t)



print('Splitting the data into training, validation ang test set')
xtr, xva, xte = split_data(seqs, split_ratio)
bgtr, bgva, bgte = split_data(bkg, split_ratio)
trs, vas, tes = split_data(sids, split_ratio)

print('Encoding the data')
x_train, y_train = encode_seqs(xtr, bgtr, VOCAB, freq)
x_val, y_val = encode_seqs(xva, bgva, VOCAB, freq)
x_test, y_test = encode_seqs(xte, bgte, VOCAB, freq)

print('x_train shape', x_train.shape)
print('y_train shape', y_train.shape)
print('x_val shape', x_val.shape)
print('y_val shape', y_val.shape)
print('x_test shape', x_test.shape)
print('y_test shape', y_test.shape)



##########################################################
# DEFINING THE NEURAL NETWORK
##########################################################
par3 = RandomUniform(minval= 0.001, maxval= 0.01, seed=None)
par2 = Constant(1.)



def wta_last_ax(args):
    return K.max(args, axis=-1, keepdims=True)




inp = Input(batch_shape=(batch_size,) + x_train.shape[1:])
# , kernel_regularizer=l1_reg
inp_res = Reshape((x_train.shape[-1],1))(inp)

con = Conv1D(2, kernel_size=10*4, use_bias=False,
             padding='valid', activation='relu', strides=len(VOCAB))(inp_res)
ccc = Lambda(wta_last_ax)(con)
#ccc = Multiply()([ccc,ccc])
flstm = Flatten()(ccc)

out = Dense(1, activation='sigmoid', trainable=True,
            use_bias=True)(flstm)


lstm_model = Model(inp, out)
opt = optimizers.Adam(lr=0.001)
lstm_model.compile(optimizer=opt, loss='binary_crossentropy',
                metrics=['accuracy'])
lstm_model.summary()
plot_model(lstm_model, to_file='I_<3_U_LSTM.png')







if __name__ == "__main__":

    #"""
    mcp = ModelCheckpoint('lstm_mod2.best.hdf5', monitor="val_acc",
                              save_best_only=True, save_weights_only=False)

    lstm_model.fit(x_train, y_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            callbacks=[mcp])
    #"""


    print('Predicting...')
    print("Loading the model")
    lstm_model.load_weights("lstm_mod2.best.hdf5")
    pred = lstm_model.predict(x_test, batch_size=batch_size)
    score = lstm_model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    print(score)


    attention = Model(inp, flstm)
    profiles = attention.predict(x_test, batch_size = batch_size)


    for i in range(10):
        print(pred[i])
        plt.plot(list(profiles[i].flatten()))
        plt.xlabel('Base positions')
        plt.ylabel('Output value of LSTM layer')
        plt.axvline(int(tes[i]), color = 'black')
        plt.savefig("profiles_{}.png".format(i))
        plt.close('all')