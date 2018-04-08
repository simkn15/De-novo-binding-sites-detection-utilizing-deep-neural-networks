cmake_minimum_required(VERSION 3.9)
project(autoencoder)

set(CMAKE_CXX_STANDARD 11)

add_executable(autoencoder
        autoencoder/autoencoder.py
        autoencoder/autoencoderMnist.py
        autoencoder/summation.py
        data/dna_['CAGTCATTCC']_1000_100_1_nch_0.fa
        data/dna_['GAACTACTTA']_100_100_1_nch_0.fa
        data/dna_['GTCCTGTTTT']_500_100_1_nch_0.fa
        data/dna_['TCATCACAGT']_300_100_1_nch_0.fa)
