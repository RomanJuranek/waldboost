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
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, Concatenate, Add, Dropout, Activation, BatchNormalization, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from .samples import gather_samples


def model_cnn(input_shape):
    img = Input(shape = input_shape)
    score = Input([1])

    x = Conv2D(8, 3, padding="same")(img)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(8, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    #x = Conv2D(8, 1, padding="same", activation="relu")(x)

    x = Conv2D(16, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(16, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    #x = Conv2D(8, 1, padding="same", activation="relu")(x)

    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation="linear")(x)
    x = Add()([x, score])

    return Model(inputs=[img, score], outputs=x)


def exploss(y_true, y_pred):
    return K.maximum(K.minimum(K.exp(-y_true * y_pred), 1e3), 1e-6)


def train(M, X0, H0, X1, H1, epochs=10, batch_size=64, steps=1000):
    b = batch_size // 2
    M.compile(loss=exploss, optimizer=Adam(lr=1e-4))
    N0,N1 = X0.shape[0], X1.shape[0]
    Y = np.array([-1]*b + [1]*b, dtype="f")
    for e in range(1,epochs+1):
        print(f"Epoch {e}/{epochs}")
        loss = []
        for _ in range(steps):
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



def detect_and_verify(image, model, verifier):
    bbs = []
    scores = []
    samples = []
    for chns, scale in model.channels(image):
        r,c,h = model.predict_on_image(chns)
        bbs.append( model.get_bbs(r, c, scale) )
        scores.append( h )
        samples.append( gather_samples(chns, r, c, model.shape) )
    bbs = [x for x in bbs if x.size]
    scores = [x for x in scores if x.size]
    samples = [x for x in samples if x.size]

    if not bbs: return [],[]

    bbs = np.concatenate(bbs)
    scores = np.concatenate(scores)
    samples = np.concatenate(samples)
    scores = verifier.predict( [samples.astype("f"), scores.astype("f")], batch_size=256 )

    return bbs, scores.flatten()
