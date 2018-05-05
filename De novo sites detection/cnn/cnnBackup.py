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
from keras import activations
from keras import metrics
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Conv1D, Conv2D, Multiply, MaxPool2D, UpSampling2D
from keras.layers import LSTM, Add
from keras.models import Model
from keras import backend
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
        self.combinedNucleotides = self.__getCombinedNucleotides()

        self.baseIndices = dict((c, i) for i, c in enumerate(self.bases))
        self.indicesBase = dict((i, c) for i, c in enumerate(self.bases))

        self.dataRatio = [0.7, 0.15, 0.15]
        self.sequences = [str(record.seq) for record in records] # Sequences from file | Has capital letters
        self.numSequences = len(self.sequences)
        self.sequenceLength = len(self.sequences[0])
        self.distribution = self.__calculateDistribution()

        # Sequences as images
        self.images = self.__encodeSequenceMatrixToImages(self.sequences) # Sequences from file as images
        # self.images = self.images - np.min(self.images)
        # Make sure all data is set as np.array
        self.xTrainImages, self.yTrainImages = self.__getTrainImages()
        self.xValidationImages, self.yValidationImages = self.__getValidationImages()
        self.xTestImages, self.yTestImages = self.__getTestImages()

        # Sequences
        # self.xTrain, self.yTrain = self.__getTrainData()
        # self.xValidation, self.yValidation = self.__getValidationData()
        # self.xTest, self.yTest = self.__getTestDate()

        self.trainMotifIndices, self.validationMotifIndices, self.testMotifIndices = self.__getMotifIndices(records)

    def __getTrainImages(self):
        endIndex = round(self.dataRatio[0] * self.numSequences)
        sequencesMotif = self.sequences[: endIndex]
        sequencesNoMotif = [self.__generateSequenceFromDistribution() for i in range(len(sequencesMotif))]
        xTrainImages = self.__encodeSequenceMatrixToImages(sequencesMotif + sequencesNoMotif)
        yTrainImages = np.asarray([1 if i < len(sequencesMotif) else 0 for i in range(len(xTrainImages))])
        return xTrainImages, yTrainImages

    def __getValidationImages(self):
        startIndex = round(self.dataRatio[0] * self.numSequences)
        endIndex = startIndex + round(self.dataRatio[1] * self.numSequences)
        sequencesMotif = self.sequences[startIndex: endIndex]
        sequencesNoMotif = [self.__generateSequenceFromDistribution() for i in range(len(sequencesMotif))]
        xValidationImages = self.__encodeSequenceMatrixToImages(sequencesMotif + sequencesNoMotif)
        yValidationImages = np.asarray([1 if i < len(sequencesMotif) else 0 for i in range(len(xValidationImages))])
        return xValidationImages, yValidationImages

    def __getTestImages(self):
        startIndex = round(self.dataRatio[0] * self.numSequences) + round(self.dataRatio[1] * self.numSequences)
        sequencesMotif = self.sequences[startIndex:]
        sequencesNoMotif = [self.__generateSequenceFromDistribution() for i in range(len(sequencesMotif))]
        xTestImages = self.__encodeSequenceMatrixToImages(sequencesMotif + sequencesNoMotif)
        yTestImages = np.asarray([1 if i < len(sequencesMotif) else 0 for i in range(len(xTestImages))])
        return xTestImages, yTestImages

    def __encodeSequenceToImage(self, sequence):
        # TODO better approach than to .lower() on every subseq?
        # image size should be (6, 288) as we remove 2 nucleotides in both start and end
        # image = np.zeros(shape=(6, self.sequenceLength * 3))
        ##### Dimensions of 3
        image = np.zeros(shape=(6, 288))

        row0 = []
        for i, c in enumerate(sequence):
            point = self.nucleotides[c.lower()]
            row0.extend(point)

        row1 = []
        for i in range(0, self.sequenceLength - 1, 2):
            subseq = sequence[i : i+2].lower()
            point = self.diNucleotides[subseq]
            row1.extend(point * 2)

        row2 = [0]*3 # Padding as we start from index 1
        for i in range(1, self.sequenceLength - 1, 2):
            subseq = sequence[i : i+2].lower()
            point = self.diNucleotides[subseq]
            row2.extend(point * 2)
        row2.extend([0]*3) # Padding as there are 1 nucleotide left

        row3 = []
        for i in range(0, self.sequenceLength - 2, 3):
            subseq = sequence[i : i+3].lower()
            point = self.triNucleotides[subseq]
            row3.extend(point * 3)
        row3.extend([0]*3) # Padding as there are 1 nucleotide left

        row4 = [0]*3 # Padding as we start from index 1
        for i in range(1, self.sequenceLength - 2, 3):
            subseq = sequence[i : i+3].lower()
            point = self.triNucleotides[subseq]
            row4.extend(point * 3)
        # No padding on row4, as there are no nucleotides left

        # 1 nucleotide is 9 elements. 9 * 33 = 297.
        # Starting at index 2, we destroy a trinucleotide -> only 32 trinucleotides can be represented
        row5 = [0]*6 # Padding as we start from index 2
        for i in range(2, self.sequenceLength - 2, 3):
            subseq = sequence[i : i+3].lower()
            point = self.triNucleotides[subseq]
            row5.extend(point * 3)
        row5.extend([0]*6) # Padding as there are 2 nucleotides left

        image[0] = row0[6:-6]
        image[1] = row1[6:-6]
        image[2] = row2[6:-6]
        image[3] = row3[6:-6]
        image[4] = row4[6:-6]
        image[5] = row5[6:-6]

        ##### Dimensions of 85
        # image = np.zeros(shape=(6, (100 * 85) - (85 * 4)))
        # row0 = []
        # for i, c in enumerate(sequence):
        #     point = self.combinedNucleotides[c.lower()]
        #     row0.extend(point)
        #
        # row1 = []
        # for i in range(0, self.sequenceLength - 1, 2):
        #     subseq = sequence[i : i+2].lower()
        #     point = self.combinedNucleotides[subseq]
        #     row1.extend(point * 2)
        #
        # row2 = [0]*85 # Padding as we start from index 1
        # for i in range(1, self.sequenceLength - 1, 2):
        #     subseq = sequence[i : i+2].lower()
        #     point = self.combinedNucleotides[subseq]
        #     row2.extend(point * 2)
        # row2.extend([0]*85) # Padding as there are 1 nucleotide left
        #
        # row3 = []
        # for i in range(0, self.sequenceLength - 2, 3):
        #     subseq = sequence[i : i+3].lower()
        #     point = self.combinedNucleotides[subseq]
        #     row3.extend(point * 3)
        # row3.extend([0]*85) # Padding as there are 1 nucleotide left
        #
        # row4 = [0]*85 # Padding as we start from index 1
        # for i in range(1, self.sequenceLength - 2, 3):
        #     subseq = sequence[i : i+3].lower()
        #     point = self.combinedNucleotides[subseq]
        #     row4.extend(point * 3)
        # # No padding on row4, as there are no nucleotides left
        #
        # # 1 nucleotide is 9 elements. 9 * 33 = 297.
        # # Starting at index 2, we destroy a trinucleotide -> only 32 trinucleotides can be represented
        # row5 = [0]*(2*85) # Padding as we start from index 2
        # for i in range(2, self.sequenceLength - 2, 3):
        #     subseq = sequence[i : i+3].lower()
        #     point = self.combinedNucleotides[subseq]
        #     row5.extend(point * 3)
        # row5.extend([0]*(2*85)) # Padding as there are 2 nucleotides left
        #
        # image[0] = row0[170:-170]
        # image[1] = row1[170:-170]
        # image[2] = row2[170:-170]
        # image[3] = row3[170:-170]
        # image[4] = row4[170:-170]
        # image[5] = row5[170:-170]

        return image

    def __encodeSequenceMatrixToImages(self, sequenceMatrix):
        return np.asarray([self.__encodeSequenceToImage(seq) for seq in sequenceMatrix])

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

    def __getCombinedNucleotides(self):
        combined = self.nucleotides.copy()
        combined.update(self.diNucleotides)
        combined.update(self.triNucleotides)
        dissimilarities = np.zeros((len(combined), len(combined)))
        dissimilarities.fill(255)
        np.fill_diagonal(dissimilarities, 0)
        mds = manifold.MDS(85, dissimilarity='precomputed')
        pos = mds.fit(dissimilarities).embedding_
        # Transformed points in the 3d space
        transformedPos = mds.fit_transform(dissimilarities, init=pos)
        combinedDict = dict((c, list(transformedPos[i])) for i, c in enumerate(combined.keys()))
        return combinedDict


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

    # TODO convert getData methods into -> getInputData, getOutputData ?
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
    return backend.max(args, axis=-1, keepdims=True)

seedsDict = dict()
#https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
os.environ['PYTHONHASHSEED'] = '0'
seed = 91
np.random.seed(seed)
rn.seed(seed)
tf.set_random_seed(seed)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
backend.set_session(sess)

data = Data(getRecords("dna_TCCCACAAAC_2000_100.fa"))
# print("xTrain shape:", data.xTrain.shape)
# print("yTrain shape:", data.yTrain.shape)
# print("xValidation shape:", data.xValidation.shape)
# print("yValidation shape:", data.yValidation.shape)
# print("xTest shape:", data.xTest.shape)
# print("yTest shape:", data.yTest.shape)
# print("images shape:", data.images.shape)
# print("xTrainImages shape:", data.xTrainImages.shape)
# print("yTrainImages shape:", data.yTrainImages.shape)
# print("xValidationImages shape:", data.xValidationImages.shape)
# print("yValidationImages shape:", data.yValidationImages.shape)
# print("xTestImages shape:", data.xTestImages.shape)
# print("yTestImages shape:", data.yTestImages.shape)

##########################################################
# DEFINING THE NEURAL NETWORK
##########################################################

batch_size = 10#50
epochs = 100
dimensions = 3

input = Input(batch_shape=(batch_size,) + data.images.shape[1:])
print("input shape:", input.shape)
inputRes = Reshape((data.images.shape[1], data.images.shape[2], 1))(input)
print("inputRes shape:", inputRes.shape)

##### Regular CNN
# conv = Conv2D(3, kernel_size=(6, 10*3), use_bias=False,
#              padding='valid', activation='relu', strides=(1,3))(inputRes)
# maxLambda = Lambda(wta_last_ax)(conv)
# # maxPool = MaxPool2D((6, 3), strides=(1,1), padding='valid')(conv)
# # maxLambda = Lambda(wta_last_ax)(maxPool)
# fl = Flatten()(maxLambda)
# output = Dense(1, activation='sigmoid', trainable=True,
#                use_bias=True)(fl)
#
# model = Model(input, output)
# optimizer = optimizers.Adam(lr=0.001)
# model.compile(optimizer=optimizer, loss='binary_crossentropy',
#               metrics=['accuracy'])
# model.summary()
# plot_model(model, to_file='modelImages.png')


##### CNN
## Current best prediction
# x = Conv2D(6, kernel_size=(6, 3), activation='relu', padding='valid', strides=(6,3))(inputRes)
# x = Conv2D(3, kernel_size=(1, 1), activation='relu', padding='valid', strides=(1,1))(x)
# maxLambda = Lambda(wta_last_ax)(x)
# fl = Flatten()(maxLambda)
# output = Dense(1, activation='sigmoid', trainable=True,
#                use_bias=True)(fl)
##
x = Conv2D(6, kernel_size=(6, 3), activation='relu', padding='valid', strides=(6,3))(inputRes)
x = Conv2D(3, kernel_size=(1, 1), activation='relu', padding='valid', strides=(1,1))(x)
maxLambda = Lambda(wta_last_ax)(x)
fl = Flatten()(maxLambda)
output = Dense(1, activation='sigmoid', trainable=True,
               use_bias=True)(fl)

model = Model(input, output)
optimizer = optimizers.Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
plot_model(model, to_file='modelImages.png')

if __name__ == "__main__":
    mcp = ModelCheckpoint('cnnImages.best.hdf5', monitor="val_acc",
                              save_best_only=True, save_weights_only=False)

    model.fit(data.xTrainImages, data.yTrainImages,
               shuffle=True,
               epochs=epochs,
               batch_size=batch_size,
               validation_data=(data.xValidationImages, data.yValidationImages),
               callbacks=[mcp])

    print('Predicting...')
    print("Loading the model")
    model.load_weights("cnnImages.best.hdf5")
    pred = model.predict(data.xTestImages, batch_size=batch_size)
    score = model.evaluate(data.xTestImages, data.yTestImages, batch_size=batch_size, verbose=1)
    print(score)

    attention = Model(input, fl)
    profiles = attention.predict(data.xTestImages, batch_size=batch_size)
    print("profiles length:", len(profiles))

    yAxisMaxValue = max(profiles.flatten())
    for i in range(0, 600):
        print(pred[i])
        plt.plot(list(profiles[i].flatten()))
        plt.xlabel('Base positions')
        plt.ylabel('Output value of maxLambda layer')
        plt.ylim(0, yAxisMaxValue)
        if i < 300:
            plt.axvline(int(data.testMotifIndices[i]), color='black')
        plt.savefig("imageProfilesConvLayer_{}.png".format(i))
        plt.close('all')

    # Stats for motif
    countHitsSequence = 0
    countHitsMotif = 0
    sumDiff = 0
    listMaxPos = []
    listStartPos = []
    listMotifIndexWrongClass = []
    mapDiff = dict((k, 0) for k in range(0, 11))
    for i in range(0, 300):
        maxPos = np.argmax(profiles[i].flatten())
        seqStartPos = int(data.testMotifIndices[i])
        listMaxPos.append(maxPos)
        listStartPos.append(seqStartPos)
        if abs(seqStartPos-maxPos) <= 10:
            mapDiff[abs(seqStartPos-maxPos)] += 1
        if seqStartPos <= maxPos < seqStartPos+10:
            countHitsSequence += 1
            sumDiff += abs(maxPos - seqStartPos)

        if pred[i] >= 0.5 and data.yTestImages[i] == 1:
            countHitsMotif += 1
        elif pred[i] < 0.5 and data.yTestImages[i] == 0:
            countHitsMotif += 1
        else:
            listMotifIndexWrongClass.append(i)

    avgDiff = sumDiff / 300
    print("Hit correct on", countHitsSequence, "sequences out of", len(data.testMotifIndices))
    print("Average maxValue index after sequence start index:", avgDiff)
    print("Correct motif classification: ", str(countHitsMotif) + "/300")
    print("Motif sequences with wrong classificaton: ", listMotifIndexWrongClass)

    # Stats for background
    countHitsBackground = 0
    listBackgroundIndexWrongClass = []
    for i in range(300, 600):
        if pred[i] >= 0.5 and data.yTestImages[i] == 1:
            countHitsBackground += 1
        elif pred[i] < 0.5 and data.yTestImages[i] == 0:
            countHitsBackground += 1
        else:
            listBackgroundIndexWrongClass.append(i)
    print("Correct background classification: ", str(countHitsBackground) + "/300")
    print("Background sequences with wrong classification: ", listBackgroundIndexWrongClass)

    # print(listMaxPos)
    # print(listStartPos)
    # print(mapDiff)

    # Count correct classification on both motif and background sequences
    # Look at plots for background sequences. Any peaks ?
    # Check the difference between motif and background plots
    # seedsDict[s] = (countHits, avgDiff, mapDiff)
    # print(seedsDict)
    # print("############### Done with seed", s)

# f = open("seedsOutput.txt", "w")
# for key, val in seedsDict.items():
#     f.write(str(key) + " | " + str(val) + "\n")
# f.close()