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
from collections import Counter
from Bio import SeqIO
import itertools
from sklearn import manifold


#https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
tf.set_random_seed(42)
rn.seed(42)


batch_size = 10#50
epochs = 100
VOCAB = ['a','c','g','t']
split_ratio = [0.8, 0.1, 0.1]#[0.7, 0.15, 0.15]
N_LSTM = 10
filters = 100

##########################################################
# CLASSES
##########################################################
class Data:
    def __init__(self, records):
        self.bases = ['a', 'c', 'g', 't']
        self.numBases = len(self.bases)

        self.nucleotides = self.__getNucleotides()
        self.diNucleotides = self.__getDiNucleotides()
        self.triNucleotides = self.__getTriNucleotides()

        self.baseIndices = dict((c, i) for i, c in enumerate(self.bases))
        self.indicesBase = dict((i, c) for i, c in enumerate(self.bases))

        self.dataRatio = [0.7, 0.15, 0.15]
        self.sequences = [str(record.seq) for record in records]
        self.numSequences = len(self.sequences)
        self.sequenceLength = len(self.sequences[0])

        self.distribution = self.__calculateDistribution()

        self.xTrain, self.yTrain = self.__getTrainData()
        self.xValidation, self.yValidation = self.__getValidationData()
        self.xTest, self.yTest = self.__getTestDate()

        self.trainMotifIndices, self.validationMotifIndices, self.testMotifIndices = self.__getMotifIndices(records)

    # Make the methods for nucleotides returning dict where value is the 3d point.
    def __getNucleotides(self):
        dissimilarities = np.zeros((self.numBases, self.numBases))
        dissimilarities.fill(255)
        np.fill_diagonal(dissimilarities, 0)
        mds = manifold.MDS(3, dissimilarity='precomputed')
        pos = mds.fit(dissimilarities).embedding_
        # Transformed points in the 3d space
        transformedPos = mds.fit_transform(dissimilarities, init=pos)
        nucleotidesPositionDict = dict((c, list(transformedPos[i])) for i, c in enumerate(self.bases))
        return nucleotidesPositionDict

    def __getDiNucleotides(self):
        tuples = [a + b for (a, b) in list(itertools.product(self.bases, self.bases))]
        dissimilarities = np.zeros((len(tuples), len(tuples)))
        dissimilarities.fill(255)
        np.fill_diagonal(dissimilarities, 0)
        mds = manifold.MDS(3, dissimilarity='precomputed')
        pos = mds.fit(dissimilarities).embedding_
        # Transformed points in the 3d space
        transformedPos = mds.fit_transform(dissimilarities, init=pos)
        diNuclotidesPositionDict = dict((c, list(transformedPos[i])) for i, c in enumerate(tuples))

        return diNuclotidesPositionDict

    def __getTriNucleotides(self):
        triplets = [a + b + c for (a, b, c) in list(itertools.product(self.bases, self.bases, self.bases))]
        dissimilarities = np.zeros((len(triplets), len(triplets)))
        dissimilarities.fill(255)
        np.fill_diagonal(dissimilarities, 0)
        mds = manifold.MDS(3, dissimilarity='precomputed')
        pos = mds.fit(dissimilarities).embedding_
        # Transformed points in the 3d space
        transformedPos = mds.fit_transform(dissimilarities, init=pos)
        triNuclotidesPositionDict = dict((c, list(transformedPos[i])) for i, c in enumerate(triplets))
        return triNuclotidesPositionDict

    def __getMotifIndices(self, records):
        train = []
        validation = []
        test = []
        for i, record in enumerate(records):
            id = record.id
            id = id[::-1] # reverse
            id = id[:id.index('_')]
            id = id[::-1]
            if i < round(self.dataRatio[0] * self.numSequences):
                train.append(id)
            elif i < round(self.dataRatio[0] * self.numSequences) + round(self.dataRatio[1] * self.numSequences):
                validation.append(id)
            else:
                test.append(id)
        return train, validation, test

    def __calculateDistribution(self):
        totalOccurrences = dict((base, 0) for base in self.bases)
        distribution = []
        totalBases = len(self.sequences) * len(self.sequences[0])
        for seq in self.sequences:
            occurrences = Counter(seq)
            for k in occurrences.keys():
                totalOccurrences[str(k).lower()] += occurrences.get(k)

        for base in self.bases:
            probability = totalOccurrences[base] / totalBases
            distribution.append(probability)
        return distribution

    def __generateBackground(self):
        return [self.__generateSequenceFromDistribution() for i in range(self.numSequences)]

    def __generateSequenceFromDistribution(self):
        sequence = ""
        for i in range(self.sequenceLength):
            base = np.random.choice(self.bases, p = self.distribution)
            sequence += base
        return sequence

    def __encodeSingleSequenceToOneHot(self, sequence):
        oneHotEncoding = np.zeros((len(sequence), self.numBases), dtype=np.uint8)
        for i, base in enumerate(sequence):
            oneHotEncoding[i, self.baseIndices[base.lower()]] = 1
        return oneHotEncoding.flatten()

    def __encodeSequencesToOneHot(self, sequences):
        oneHotEncoding = np.zeros((len(sequences), len(sequences[0]) * self.numBases), dtype=np.uint8)
        for i, seq in enumerate(sequences):
            oneHotEncoding[i,] = self.__encodeSingleSequenceToOneHot(seq)
        return oneHotEncoding

    def __getTrainData(self):
        endIndex = round(self.dataRatio[0] * self.numSequences)
        sequencesMotif = self.sequences[ : endIndex]
        sequencesNoMotif = [self.__generateSequenceFromDistribution() for i in range(len(sequencesMotif))]
        xTrain = self.__encodeSequencesToOneHot(sequencesMotif + sequencesNoMotif)
        yTrain = np.asarray([1 if i < len(sequencesMotif) else 0 for i in range(len(xTrain))])
        return xTrain, yTrain

    def __getValidationData(self):
        startIndex = round(self.dataRatio[0] * self.numSequences)
        endIndex = startIndex + round(self.dataRatio[1] * self.numSequences)
        sequencesMotif = self.sequences[startIndex : endIndex]
        sequencesNoMotif = [self.__generateSequenceFromDistribution() for i in range(len(sequencesMotif))]
        xValidation = self.__encodeSequencesToOneHot(sequencesMotif + sequencesNoMotif)
        yValidation = np.asarray([1 if i < len(sequencesMotif) else 0 for i in range(len(xValidation))])
        return xValidation, yValidation

    def __getTestDate(self):
        startIndex = round(self.dataRatio[0] * self.numSequences) + round(self.dataRatio[1] * self.numSequences)
        sequencesMotif = self.sequences[startIndex : ]
        sequencesNoMotif = [self.__generateSequenceFromDistribution() for i in range(len(sequencesMotif))]
        xTest = self.__encodeSequencesToOneHot(sequencesMotif + sequencesNoMotif)
        yTest = np.asarray([1 if i < len(sequencesMotif) else 0 for i in range(len(xTest))])
        return xTest, yTest

##########################################################
# DEFINING FUNCTION USED FOR PRE-PROCESSING OF INPUT DATA
##########################################################

def getRecords(filePath):
    return list(SeqIO.parse(filePath, "fasta"))

def wta_last_ax(args):
    return K.max(args, axis=-1, keepdims=True)

data = Data(getRecords("dna_TCCCACAAAC_2000_100.fa"))
print("xTrain shape:", data.xTrain.shape)
print("yTrain shape:", data.yTrain.shape)
print("xValidation shape:", data.xValidation.shape)
print("yValidation shape:", data.yValidation.shape)
print("xTest shape:", data.xTest.shape)
print("yTest shape:", data.yTest.shape)

##########################################################
# DEFINING THE NEURAL NETWORK
##########################################################

input = Input(batch_shape=(batch_size,) + data.xTrain.shape[1:])
print("input shape:", input.shape)
print("input shape:", input.shape[-1])
inputRes = Reshape((data.xTrain.shape[-1], 1))(input)
print("data.xTrain.shape[-1]:", data.xTrain.shape[-1])
print("inputRes shape:", inputRes.shape)

con = Conv1D(2, kernel_size=10*4, use_bias=False,
             padding='valid', activation='relu', strides=data.numBases)(inputRes)
ccc = Lambda(wta_last_ax)(con)
flstm = Flatten()(ccc)

output = Dense(1, activation='sigmoid', trainable=True,
               use_bias=True)(flstm)

model = Model(input, output)
optimizer = optimizers.Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
plot_model(model, to_file='model.png')

# if __name__ == "__main__":
#     # mcp = ModelCheckpoint('cnn.best.hdf5', monitor="val_acc",
#     #                           save_best_only=True, save_weights_only=False)
#     #
#     # lstmModel.fit(data.xTrain, data.yTrain,
#     #            shuffle=True,
#     #            epochs=epochs,
#     #            batch_size=batch_size,
#     #            validation_data=(data.xValidation, data.yValidation),
#     #            callbacks=[mcp])
#
#     print('Predicting...')
#     print("Loading the model")
#     model.load_weights("cnn.best.hdf5")
#     pred = model.predict(data.xTest, batch_size=batch_size)
#     score = model.evaluate(data.xTest, data.yTest, batch_size=batch_size, verbose=1)
#     print(score)
#
#     attention = Model(input, flstm)
#     profiles = attention.predict(data.xTest, batch_size=batch_size)
#     print("profiles length:", len(profiles))
#
#     for i in range(10):
#         print(pred[i])
#         plt.plot(list(profiles[i].flatten()))
#         plt.xlabel('Base positions')
#         plt.ylabel('Output value of LSTM layer')
#         plt.axvline(int(data.testMotifIndices[i]), color='black')
#         plt.savefig("testProfiles_{}.png".format(i))
#         plt.close('all')
