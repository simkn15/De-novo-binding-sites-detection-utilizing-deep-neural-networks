General:
    - Alex had the most success with a convolutional autoencoder with maxpooling

Autoencoder:
    - Autoencoders tries to reconstruct the input to the output
        - How does this help us determine if we have found a motif ????
        - How does this help us know where the motif is, if there is one ????
    - How does the output look like when adding the following ?:
        - Convolutional layers
        - Maxpooling

Output:
    How should the output be represented?
        - Undercomplete
        - Overcomplete
        - One-to-one
    How do we read and learn from the output?
        - If one-to-one, how do we extract the learned features?
            - Presuming lossy output. Non-lossy output does now reveal anything, as it is just a copy of the input.

Decide on:
- Number of layers
    - Size of the layers
- Activation function: https://keras.io/activations/
    - Probably just use 'relu'.
    - It is difficult to say if any activation function is better than the other in any given situation
- Biases and its initialization
- Weights and its initialization
- Optimizer (Stochastic Gradient Descent) : autoencoderMnist.py = adadelta
    - Should most likely be SGD : Unless testing of adadelta and its paper indicate as better choice.
        - The paper apparently states that it is better as of testing with MNIST
- Loss function : autoencoderMnist.py = binary_crossentropy

- Later considerations:
    - Type of layers : https://keras.io/layers/about-keras-layers/
    - Regularizers
    - Dropout
    - Spatial

TODO:
- Try other loss functions
- Try with sigmoid on last encoding layer
- Implement a convolutional autoencoder with maxpooling