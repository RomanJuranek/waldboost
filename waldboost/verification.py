""" CNN Verification

Verification model inputs are image features and detector response. Based on
the image features, model predicts the value that is added.

p(X,H) = sigmoid(model(X) + H)

The model is trained to minimize binary cross entropy of p(X,H) which is 0 for
negative samples and 1 for positive samples.

Example:
M = verification.model(input_shape=X0.shape[1:])
verification.train(M, X0, H0, X1, H1, batch_size=64, epochs=10)

bbs, h, p = detect(image, model, M)
"""


import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, Concatenate, Add, Dropout, Activation, BatchNormalization, Flatten
from tensorflow.keras.models import Model


def model(input_shape):
    img = Input(shape = input_shape)
    score = Input([1])

    x = Conv2D(32, 3, padding="same")(img)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(32, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Conv2D(32, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(32, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(1, activation="linear")(x)
    x = Add()([x, score])
    x = Activation("sigmoid")(x)

    return Model(inputs=[img, score], outputs=x)


def train(M, X0, H0, X1, H1, epochs=10, batch_size=64):
    b = batch_size // 2
    M.compile(loss="binary_crossentropy", optimizer="adam")
    N0,N1 = X0.shape[0], X1.shape[0]
    Y = np.array([0]*b + [1]*b, dtype="f")
    for e in range(1,epochs+1):
        print(f"Epoch {e}/{epochs}")
        loss = []
        for _ in range(1000):
            i0 = np.random.choice(N0, b)
            i1 = np.random.choice(N1, b)
            X = [
                np.concatenate([X0[i0,...], X1[i1,...]]).astype("f"),
                np.concatenate([H0[i0], H1[i1]]).astype("f"),
            ]
            l = M.train_on_batch(X, Y)
            loss.append(l)
        print(np.mean(loss))
    print("Done")
