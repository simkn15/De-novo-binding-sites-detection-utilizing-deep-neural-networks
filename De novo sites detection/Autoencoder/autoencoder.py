import numpy as np
import itertools
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv1D, Conv2D, MaxPooling2D, UpSampling2D, MaxPooling1D, UpSampling1D
from keras.models import Model, load_model
from Bio import SeqIO
import os.path

# TODO: Generate the datasets by randomization. Atm it is the same distribution of rows every time. (Same indices)
# TODO: How does seed affect shuffle in Model().fit(shuffle=true): Seems like there are no difference. Every run has a distinct output.
"""
TODO:
    - Convert sequence into image:
        - Row 1: Single nucleotides:
        - Row 2: Pairs of nucleotides: start index 0
        - Row 3: Pairs of nucleotides: start index 1
        - Row 4: Triplets of nucleotides: start index 0
        - Row 5: Triplets of nucleotides: start index 1
        - Row 6: Triplets of nucleotides: start index 2
        * 4^1 = 4 singles
            - Get RGB colour where all singles have equal distances
        * 4^2 = 16 pairs
            - Get RGB colour where all pairs have equal distances
        * 4^3 = 64 triplets
            - Get RGB colour where all triplets have equal distances
        * Use an equivalent method to isoMDS() to convert to lower/higher dimensions: scipy ?
    - Move class CharacterTable() into own file ?
    - Move class Sequence() into own file ??
"""

class CharacterTable:
    def __init__(self):
        self.basesBackground = ["a", "c", "g", "t"]
        self.basesForeground = ["A", "C", "G", "T"]
        self.numBases = len(self.basesBackground)
        self.baseIndices = dict((c, i) for i, c in enumerate(self.basesBackground))
        self.indicesBase = dict((i, c) for i, c in enumerate(self.basesBackground))

    def encodeSequenceToOneHot(self, sequence):
        """
        :param sequence: (string)
        :return: Hot encoding of sequence
        """
        oneHotEncoding = np.zeros((len(sequence), self.numBases), dtype=np.uint8)
        for i, base in enumerate(sequence):
            oneHotEncoding[i, self.baseIndices[base.lower()]] = 1
        return oneHotEncoding.flatten()

    def encodeMatrixToOneHot(self, sequenceMatrix):
        oneHotEncoding = np.zeros((len(sequenceMatrix), len(sequenceMatrix[0]) * self.numBases), dtype=np.uint8)
        for i, seq in enumerate(sequenceMatrix):
            oneHotEncoding[i,] = self.encodeSequenceToOneHot(seq)
        return oneHotEncoding

    def decodeOneHot(self, sequence):
        """
        :param sequence: Hot encoded sequence
        :return: Decoded sequence (string)
        """
        seq = ''
        for i in range(0, len(sequence), self.numBases):
            for j in range(0, self.numBases):
                if sequence[i + j] == 1:
                    seq += self.indicesBase[j]
                    break
        return seq

    def decodeMaxFloat(self, sequence):
        """
        :param sequence: encoded sequence with elements as floats
            (Every 4 elements is one base, and max float is the actual base)
        :return: Decoded sequence (string)
        """
        seq = ''
        for i in range(0, len(sequence), self.numBases):
            maxFloat = 0.0
            maxIndex = 0
            for j in range(0, self.numBases):
                if sequence[i + j] > maxFloat:
                    maxFloat = sequence[i + j]
                    maxIndex = j
            # print("maxFloat = {}, maxIndex = {}, base = {}, atBaseIndex = {}".format(maxFloat, maxIndex, self.indicesBase[maxIndex], i))
            seq += self.indicesBase[maxIndex]
        return seq

    def compareTwoSequences(self, sequence, predicted, decode=False):
        """
        :param sequence: Sequence with foreground and background: E.g. Date().testSequences
        :param predicted: predicted sequence
        :param decode: Flag for decoding predicted
        :return:
        """
        if decode:
            predicted = self.decodeMaxFloat(predicted)
        backgroundHits = 0
        backgroundTotal = 0
        foregroundHits = 0
        foregroundTotal = 0
        for i, base in enumerate(sequence):
            if base in self.basesBackground:
                backgroundTotal += 1
                if base == predicted[i]:
                    backgroundHits += 1

            elif base in self.basesForeground:
                foregroundTotal += 1
                if base.lower() == predicted[i]:
                    foregroundHits += 1

        return [backgroundHits, backgroundTotal, foregroundHits, foregroundTotal]

class Sequence:
    def __init__(self, sequence, hasMotif):
        self.sequence = sequence
        self.hasMotif = 1 if hasMotif else 0

class Data:
    def __init__(self, sequenceMatrix):
        self.bases = ["a", "c", "g", "t"]
        self.tuples = self.getTuples()
        self.triplets = self.getTriples()

        self.sequencesWithMotif = self.initSequences(sequenceMatrix, hasMotif=True)

        self.baseDistribution = self.calculateDistributionOfBases()

        self.sequenceLength = len(self.sequencesWithMotif[0].sequence)

        self.sequencesNoMotif = self.generateSequencesFromBaseDistribution()

        self.trainSize = int((len(sequenceMatrix) * 2) * 0.70)
        self.validationSize = int((len(sequenceMatrix) * 2) * 0.15)
        self.testSize = int((len(sequenceMatrix) * 2) * 0.15)

        self.trainSequences = self.generateTrainSequences()
        self.validationSequences = self.generateValidationSequences()
        self.testSequences = self.generateTestSequences()

    def calculateDistributionOfBases(self):
        occurrenceDict = dict((c, 0) for i, c in enumerate(self.bases))
        distribution = []
        totalBases = 0
        for seq in self.sequencesWithMotif:
            for base in seq.sequence:
                occurrenceDict[base.lower()] += 1
                totalBases += 1

        for base in self.bases:
            probability = occurrenceDict[base] / totalBases
            distribution.append(probability)

        return distribution

    def generateSequencesFromBaseDistribution(self):
        amountOfSequences = len(self.sequencesWithMotif)
        generatedSequences = []
        for i in range(amountOfSequences):
            generatedSequences.append(self.generateSingleSequenceFromDistribution())
        return generatedSequences

    def generateSingleSequenceFromDistribution(self):
        sequence = ""
        for i in range(self.sequenceLength):
            base = np.random.choice(self.bases, p=self.baseDistribution)
            sequence += base
        return Sequence(sequence, hasMotif=False)

    def initSequences(self, records, hasMotif):
        return [Sequence(record.seq, hasMotif=hasMotif) for record in records]

    def generateTrainSequences(self):
        endIndex = int(self.trainSize / 2)
        sequences = np.append(self.sequencesWithMotif[ : endIndex], self.sequencesNoMotif[ : endIndex])
        return sequences

    def generateValidationSequences(self):
        startIndex = int(self.trainSize / 2)
        endIndex = startIndex + int(self.validationSize / 2)
        sequences = np.append(self.sequencesWithMotif[startIndex : endIndex],self.sequencesNoMotif[startIndex : endIndex])
        return sequences

    def generateTestSequences(self):
        startIndex = int(self.trainSize / 2) + int(self.validationSize / 2)
        sequences = np.append(self.sequencesWithMotif[startIndex : ], self.sequencesNoMotif[startIndex : ])
        return sequences

    def getClassVectorValidationSequences(self):
        classVector = np.zeros(len(self.validationSequences), dtype=np.uint8)
        for i, seq in enumerate(self.validationSequences):
            classVector[i] = 1 if seq.hasMotif else 0
        return classVector

    def getSequencesWithMotif(self):
        return [seq.sequence for seq in self.sequencesWithMotif]

    def getSequencesNoMotif(self):
        return [seq.sequence for seq in self.sequencesNoMotif]

    def getTrainSequences(self):
        return [seq.sequence for seq in self.trainSequences]

    def getValidationSequences(self):
        return [seq.sequence for seq in self.validationSequences]

    def getTestSequences(self):
        return [seq.sequence for seq in self.testSequences]

    # section for deriving an image of a sequence
    # How to build and view an actual image ?
    # Need to derive distinct colors for 4 + 16 + 64 = 84
    def getTuples(self):
        tuples = list(itertools.product(self.bases, self.bases))
        return [a + b for (a, b) in tuples]

    def getTriples(self):
        triplets = list(itertools.product(self.bases, self.bases, self.bases))
        return [a + b + c for (a, b, c) in triplets]

    # Build dict with key seq and value colour
    # keys are 'base', 'basebase', 'basebasebase'

class Stats:
    def __init__(self, avgHitsNoMotif, avgHitsBackgroundWithMotif, avgHitsForegroundWithMotif, avgTotalHits):
        self.avgHitsNoMotif = avgHitsNoMotif
        self.avgHitsBackgroundWithMotif = avgHitsBackgroundWithMotif
        self.avgHitsForegroundWithMotif = avgHitsForegroundWithMotif
        self.avgTotalHits = avgTotalHits

def printHits(hitsDict):
    for k, v in hitsDict.items():
        print("Hits of ", k, " occurred in ", len(v), " sequences")

def countSequencesWithMotif(sequences):
    counter = 0
    for seq in sequences:
        counter += 1 if seq.hasMotif else 0

    return counter

def getStatsPredicted(sequenceObjects, predicted, printStats=False):
    sumBackgroundHits = 0
    sumForegroundHits = 0
    sumHitsNoMotif = 0 # hits in sequences with no motif
    hitsDict = dict((i, []) for i in range(11))
    amountSequences = len(sequenceObjects)
    amountSequencesWithMotif = countSequencesWithMotif(sequenceObjects)

    for i, seq in enumerate(sequenceObjects):
        result = characterTable.compareTwoSequences(seq.sequence, predicted[i], decode=True)
        if seq.hasMotif:
            sumBackgroundHits += result[0]
            sumForegroundHits += result[2]
            hitsDict.get(result[2]).append(i)
        else:
            sumHitsNoMotif += result[0]

    printHits(hitsDict)
    avgHitsNoMotif = sumHitsNoMotif / (amountSequences - amountSequencesWithMotif)
    avgBackground = sumBackgroundHits / amountSequencesWithMotif
    avgForeground = sumForegroundHits / amountSequencesWithMotif
    avgTotalHits = (sumHitsNoMotif + sumBackgroundHits + sumForegroundHits) / amountSequences
    stats = Stats(avgHitsNoMotif, avgBackground, avgForeground, avgTotalHits)
    percentForegroundHits = (stats.avgHitsForegroundWithMotif * 100) / 10
    percentBackgroundHits = (stats.avgHitsBackgroundWithMotif * 100) / 90
    percentHitsNoMotif = (stats.avgHitsNoMotif * 100) / 100
    if printStats:
        print("Average hits in sequences with no motif: {:.3f} | {:.1f} %".format(stats.avgHitsNoMotif, percentHitsNoMotif))
        print("Average background: {:.3f} | {:.1f} %".format(stats.avgHitsBackgroundWithMotif, percentBackgroundHits))
        print("Average foreground: {:.3f} | {:.1f} %".format(stats.avgHitsForegroundWithMotif, percentForegroundHits))
        print("Average total withMotif: {:.3f}".format((stats.avgHitsForegroundWithMotif + stats.avgHitsBackgroundWithMotif)))
        print("Average foreground hit per background hit: {:.3f}".format(avgForeground/avgBackground))
        print("Total hits: {:.3f}".format(stats.avgTotalHits))

    return stats

def getRecords(filePath):
    return list(SeqIO.parse(filePath, "fasta"))

def getAutoencoder(inputOutputLayerSize, layers=[], activation='relu', optimizer='adadelta', loss='binary_crossentropy'):
    """
    :param:
    :return:
    """
    inputLayer = Input(shape=(inputOutputLayerSize,))

    if len(layers) >= 1:
        hiddenLayer = Dense(layers[0], activation=activation)(inputLayer)
        for i in range(1, len(layers)):
            hiddenLayer = Dense(layers[i], activation=activation)(hiddenLayer)
        outputLayer = Dense(inputOutputLayerSize, activation='sigmoid')(hiddenLayer)
    else:
        outputLayer = Dense(inputOutputLayerSize, activation='sigmoid')(inputLayer)

    autoencoder = Model(inputLayer, outputLayer)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    autoencoder.summary()

    return autoencoder

def getCAE(): # possibly make stride to the length of a sequence
    input = Input(shape=(400, 1))
    x = Conv1D(5, 40, strides=4, activation='relu', padding='same')(input)
    x = MaxPooling1D(2, padding='valid')(x)
    # x = Conv1D(4, 40, activation='relu', padding='same')(x)
    # x = MaxPooling1D(2, padding='valid')(x)
    # x = Conv1D(3, 40, activation='relu', padding='same')(x)
    # x = MaxPooling1D(2, padding='valid')(x)
    # x = Conv1D(2, 40, activation='relu', padding='same')(x)
    # x = MaxPooling1D(2, padding='same')(x)
    #
    x = Conv1D(2, 40, strides=4, activation='relu', padding='same')(x)
    x = UpSampling1D(4)(x)
    x = Conv1D(3, 40, strides=4, activation='relu', padding='same')(x)
    x = UpSampling1D(4)(x)
    x = Conv1D(4, 40, strides=4, activation='relu', padding='same')(x)
    x = UpSampling1D(4)(x)
    # x = Conv1D(5, 40, activation='relu', padding='same')(x)
    # x = UpSampling1D(2)(x)
    output = Conv1D(1, 40, activation='sigmoid', padding='same')(x)

    # x = Conv1D(2, 10, activation='relu', padding='same')(input)
    # x = MaxPooling1D(2, padding='valid')(x)
    # x = Conv1D(1, 10, activation='relu', padding='same')(x)
    # x = MaxPooling1D(2, padding='valid')(x)
    # x = Conv1D(1, 10, activation='relu', padding='same')(x)
    # x = UpSampling1D(2)(x)
    # x = Conv1D(2, 10, activation='relu', padding='same')(x)
    # x = UpSampling1D(2)(x)
    # output = Conv1D(1, 10, activation='sigmoid', padding='same')(x)

    cae = Model(input, output)
    cae.compile(optimizer='adadelta', loss='binary_crossentropy')
    cae.summary()

    return cae

characterTable = CharacterTable()
# filePath = "../data/dna_['GAACTACTTA']_100_100_1_nch_0.fa"
# filePath = "../data/dna_['TCATCACAGT']_300_100_1_nch_0.fa"
# filePath = "../data/dna_['GTCCTGTTTT']_500_100_1_nch_0.fa"
filePath = "../data/dna_['CAGTCATTCC']_1000_100_1_nch_0.fa"
data = Data(getRecords(filePath))
print(data.baseDistribution)

# encodedTrainSequences = characterTable.encodeMatrixToOneHot(data.getTrainSequences())
# encodedValidationSequences = characterTable.encodeMatrixToOneHot(data.getValidationSequences())
# encodedTestSequences = characterTable.encodeMatrixToOneHot(data.getTestSequences())
#
# trainNewModel = True
# saveToFile = False
# predictModel = not trainNewModel
# seed = 42
# np.random.seed(seed)
#
# if trainNewModel:
#     inputOutputLayer = 400
#     layers = [200, 100, 50, 100, 200, 400]
#     epochs = 5
#     batchSize = 25
#
#     for i in range(1):
#         # autoencoder = getAutoencoder(inputOutputLayer, layers)
#         encodedTrainSequences = encodedTrainSequences.reshape(len(encodedTrainSequences), len(encodedTrainSequences[0]), 1)
#         encodedTestSequences = encodedTestSequences.reshape(len(encodedTestSequences), len(encodedTestSequences[0]), 1)
#         encodedValidationSequences = encodedValidationSequences.reshape(len(encodedValidationSequences), len(encodedValidationSequences[0]), 1)
#         autoencoder = getCAE()
#         # autoencoder.fit(encodedTrainSequences, encodedTrainSequences,
#         #                 epochs=epochs,
#         #                 batch_size=batchSize,
#         #                 validation_data=(encodedValidationSequences, encodedValidationSequences),
#         #                 shuffle=True)
#         #
#         # predicted = autoencoder.predict(encodedTestSequences)
#         #
#         # stats = getStatsPredicted(data.testSequences, predicted, printStats=True)
#         #
#         # if saveToFile:
#         #     # fileName = "autoencoder({:.1f}#{:.0f})({}).h5".format(stats.avgHitsForegroundWithMotif, stats.avgHitsBackgroundWithMotif, seed)
#         #     fileName = "cae({:.1f}#{:.0f})({}).h5".format(stats.avgHitsForegroundWithMotif, stats.avgHitsBackgroundWithMotif, seed)
#         #     filePath = "./models/" + fileName
#         #     if not os.path.isfile(filePath):
#         #         autoencoder.save(filePath)
#
# if predictModel:
#     autoencoder = load_model('./models/cae(9.9#42)(42).h5')
#     # encodedTrainSequences = encodedTrainSequences.reshape(len(encodedTrainSequences), len(encodedTrainSequences[0]), 1)
#     encodedTestSequences = encodedTestSequences.reshape(len(encodedTestSequences), len(encodedTestSequences[0]), 1)
#     # encodedValidationSequences = encodedValidationSequences.reshape(len(encodedValidationSequences), len(encodedValidationSequences[0]), 1)
#
#     predicted = autoencoder.predict(encodedTestSequences)
#     getStatsPredicted(data.testSequences, predicted, printStats=True)
#     autoencoder.summary()
#
#     # predicted = autoencoder.predict(encodedTrainSequences)
#     # getStatsPredicted(data.trainSequences, predicted, printStats=True)
#
#     # predicted = autoencoder.predict(encodedValidationSequences)
#     # getStatsPredicted(data.validationSequences, predicted, printStats=True)

