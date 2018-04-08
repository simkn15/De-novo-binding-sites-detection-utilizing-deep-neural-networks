import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model, load_model
from Bio import SeqIO


'''
- TODO: Redo everything
'''
#####################
#       Start       #
#      Classes      #
#####################
class CharacterTable:
    def __init__(self):
        self.chars = 'acgtACGT'
        self.charIndices = dict((c, i) for i, c in enumerate(self.chars))
        self.indicesChar = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, sequence):
        """
        :param sequence: (string)
        :return: Hot encoding of sequence
        """
        X = np.zeros((len(sequence), len(self.chars)), dtype=np.bool)
        for i, char in enumerate(sequence):
            X[i, self.charIndices[char]] = 1
        return X.flatten()

    def decodeOneHot(self, sequence):
        """
        :param sequence: Hot encoded sequence
        :return: Decoded sequence (string)
        """
        seq = ''
        for i in range(0, len(sequence), len(self.chars)):
            for j in range(0, len(self.chars)):
                if sequence[i + j] == 1:
                    seq += self.indicesChar[j]
                    break
        return seq

    def decodeMaxFloat(self, sequence):
        """
        :param sequence: Hot encoded sequence
        :return: Decoded sequence (string)
        """
        seq = ''
        for i in range(0, len(sequence), len(self.chars)):
            maxFloat = 0.0
            maxIndex = 0
            for j in range(0, len(self.chars)):
                if sequence[i + j] > maxFloat:
                    maxFloat = sequence[i + j]
                    maxIndex = j
            seq += self.indicesChar[maxIndex]
        return seq

    def compareTwoSequences(self, original, predicted, decode=False):
        if decode:
            original = self.decodeMaxFloat(original)
            predicted = self.decodeMaxFloat(predicted)
        backgroundHits = 0
        backgroundTotal = 0
        foregroundHits = 0
        foregroundTotal = 0
        for i in range(len(original)):
            char = original[i]
            if char in self.chars[0:4]:
                backgroundTotal += 1
                if original[i] == predicted[i]:
                    backgroundHits += 1

            elif char in self.chars[4:]:
                foregroundTotal += 1
                if original[i] == predicted[i]:
                    foregroundHits += 1

        return [backgroundHits, backgroundTotal, foregroundHits, foregroundTotal]

#####################
#        End        #
#      Classes      #
#####################

#####################
#       Start       #
#      Methods      #
#####################


def printFastaFile(path):
    with open(path, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            print(record.id)
            print(record.seq)

def makeSequenceMatrix(records):
    sequences = []
    for i in range(len(records)):
        sequences.append(records[i].seq)

    return np.asarray(sequences)

def encodeSequenceMatrix(sequenceMatrix):
    encodedMatrix = np.zeros((sequenceMatrix.shape[0], sequenceMatrix.shape[1] * len(characterTable.chars)), dtype=np.bool)
    for i, seq in enumerate(sequenceMatrix):
        encodedMatrix[i] = characterTable.encode(seq)

    return encodedMatrix

def getAutoencoder(inputLayerSize, encodeLayers=[], decodeLayers=[], activation='relu', optimizer='adadelta', loss='binary_crossentropy'):
    """
    :param:
    :return:
    """
    inputLayer = Input(shape=(inputLayerSize,))

    encoding = Dense(encodeLayers[0], activation=activation)(inputLayer)
    for i in range(1, len(encodeLayers)):
        encoding = Dense(encodeLayers[i], activation=activation)(encoding)

    decoding = Dense(decodeLayers[0], activation=activation)(encoding)
    for i in range(1, len(decodeLayers) - 1):
        decoding = Dense(decodeLayers[i], activation=activation)(decoding)

    decoding = Dense(decodeLayers[-1], activation='sigmoid')(decoding)

    autoencoder = Model(inputLayer, decoding)
    autoencoder.compile(optimizer=optimizer, loss=loss)

    return autoencoder

def initAndTrainAutoencoder(xTrain, xTest,
                     inputLayer=800, encodeLayers=[], decodeLayers=[],
                     epochs=100,
                     batchSize=25):

    autoencoder = getAutoencoder(inputLayerSize=inputLayer, encodeLayers=encodeLayers, decodeLayers=decodeLayers)
    autoencoder.fit(xTrain, xTrain,
                    epochs=epochs,
                    batch_size=batchSize,
                    shuffle=True)

    predicted = autoencoder.predict(xTest)

    sumForegroundHits = 0
    sumBackgroundHits = 0
    amountSequences = 0
    for i in range(len(predicted)):
        res = characterTable.compareTwoSequences(xTest[i], predicted[i], decode=True)
        sumBackgroundHits += res[0]
        sumForegroundHits += res[2]
        amountSequences += 1

    print("Avg background: ", sumBackgroundHits / amountSequences)
    print("Avg foreground: ", sumForegroundHits / amountSequences)

def printStatsPredicted(xTest, predicted):
    sumForegroundHits = 0
    sumBackgroundHits = 0
    hitsDict = dict((i, []) for i in range(11))
    for i in range(len(predicted)):
        res = characterTable.compareTwoSequences(xTest[i], predicted[i], decode=True)
        sumBackgroundHits += res[0]
        sumForegroundHits += res[2]
        hitsDict.get(res[2]).append(i)

    printHits(hitsDict)
    print("Avg background: ", sumBackgroundHits / len(predicted))
    print("Avg foreground: ", sumForegroundHits / len(predicted))

    return hitsDict

def printHits(hitsDict):
    for k, v in hitsDict.items():
        print("Hits of ", k, " occurred in ", len(v), " sequences")

#####################
#        End        #
#      Methods      #
#####################

characterTable = CharacterTable()

################################
#   Load data
################################
# filePath = "./data/dna_['GAACTACTTA']_100_100_1_nch_0.fa"
# filePath = "./data/dna_['TCATCACAGT']_300_100_1_nch_0.fa"
# filePath = "./data/dna_['GTCCTGTTTT']_500_100_1_nch_0.fa"
filePath = "./data/dna_['CAGTCATTCC']_1000_100_1_nch_0.fa"
records = list(SeqIO.parse(filePath, "fasta"))

trainSize = int(len(records) * 0.9)
testSize = len(records) - trainSize
sequenceMatrix = makeSequenceMatrix(records)
encodedMatrix = encodeSequenceMatrix(sequenceMatrix)
# encodedMatrix = np.zeros((sequenceMatrix.shape[0], sequenceMatrix.shape[1] * len(characterTable.chars)), dtype=np.bool)
# for i, seq in enumerate(sequenceMatrix):
#     encodedMatrix[i] = characterTable.encode(seq)

xTrain = encodedMatrix[0:trainSize, :]
xTest = encodedMatrix[trainSize:, :]
xTrain = xTrain.astype('float32')
xTest = xTest.astype('float32')

################################
#   Training
################################
# trainSize = 0.9

# Activation function : All relu
# initAndTrainAutoencoder(xTrain, xTest, 800, [400, 200, 100, 50, 25], [50, 100, 200, 400, 800], 200, 25)  # 24.145 | 6.24
# initAndTrainAutoencoder(xTrain, xTest, 800, [400, 200, 100, 50, 25], [50, 100, 200, 400, 800], 500, 25)  # 24.42 | 5.65
# initAndTrainAutoencoder(xTrain, xTest, 800, [400, 100, 25], [200, 400, 800], 500, 25)  # 27.16 | 5.63
# initAndTrainAutoencoder(xTrain, xTest, 800, [200, 50, 25], [200, 800], 500, 25)  # 27.55 | 4.82
# initAndTrainAutoencoder(xTrain, xTest, 800, [100, 25], [100, 200, 800], 500, 25)  # 27.9 | 5.76

# Activation function : output layer with sigmoid, rest is relu
# initAndTrainAutoencoder(xTrain, xTest, 800, [400, 200, 100, 50, 25], [50, 100, 200, 400, 800], 200, 25) # 23.78 | 7.7
# initAndTrainAutoencoder(xTrain, xTest, 800, [400, 200, 100, 50, 25], [50, 100, 200, 400, 800], 1000, 25) # 23.25 | 8.39
# initAndTrainAutoencoder(xTrain, xTest, 800, [400, 200, 100, 50, 25], [50, 100, 200, 400, 800], 5000, 25) # 23.38 | 8.76
trainNewModel = False
predictModel = True
if trainNewModel:
    inputLayer = 800
    encodeLayers = [400, 200, 100, 50, 25]
    decodeLayers = [50, 100, 200, 400, 800]
    epochs = 5000
    batchSize = 25

    for i in range(20):
        autoencoder = getAutoencoder(inputLayerSize=inputLayer, encodeLayers=encodeLayers, decodeLayers=decodeLayers)
        autoencoder.fit(xTrain, xTrain,
                        epochs=epochs,
                        batch_size=batchSize,
                        shuffle=True)

        predicted = autoencoder.predict(xTest)

        sumForegroundHits = 0
        sumBackgroundHits = 0
        amountSequences = 0
        for i in range(len(predicted)):
            res = characterTable.compareTwoSequences(xTest[i], predicted[i], decode=True)
            sumBackgroundHits += res[0]
            sumForegroundHits += res[2]
            amountSequences += 1

        print("Avg background: ", sumBackgroundHits / amountSequences)
        print("Avg foreground: ", sumForegroundHits / amountSequences)

        # if (sumForegroundHits / amountSequences) > 8.50:
        #     autoencoder.save('savedModel850.h5')
        if (sumForegroundHits / amountSequences) > 8.55:
            autoencoder.save('savedModel855.h5')
        if (sumForegroundHits / amountSequences) > 8.60:
            autoencoder.save('savedModel860.h5')
        if (sumForegroundHits / amountSequences) > 8.65:
            autoencoder.save('savedModel865.h5')
        if (sumForegroundHits / amountSequences) > 8.70:
            autoencoder.save('savedModel870.h5')
        if (sumForegroundHits / amountSequences) > 8.75:
            autoencoder.save('savedModel875.h5')
        if (sumForegroundHits / amountSequences) > 8.80:
            autoencoder.save('savedModel880.h5')
        if (sumForegroundHits / amountSequences) > 8.85:
            autoencoder.save('savedModel885.h5')
        if (sumForegroundHits / amountSequences) > 8.90:
            autoencoder.save('savedModel890.h5')

if predictModel:
    autoencoder = load_model('savedModel850.h5')
    predicted = autoencoder.predict(xTest)
    printStatsPredicted(xTest, predicted)

# Make a stats function

# class Network(object):
#     def __init__(self):
#         amountOfHitsDict = dict((i, []) for in range(11))
#
#
# class StatsNeuralNetwork(object):
