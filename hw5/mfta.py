import pandas as pd
import numpy as np
from keras.layers import Input, Embedding, Flatten, Dot, Add
from keras.callbacks import ModelCheckpoint
import keras

pd.options.display.max_columns = 10 
pd.options.display.width = 134
pd.options.display.max_rows = 20

def get_model(n_users, n_items, latent_dim=500):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_vec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    user_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(user_input)
    user_bias = Flatten()(user_bias)
    item_bias = Embedding(n_items, 1, embeddings_initializer='zeros')(item_input)
    item_bias = Flatten()(item_bias)
    r_hat = Dot(axes=1)([user_vec, item_vec])
    r_hat = Add()([r_hat, user_bias, item_bias])
    model = keras.models.Model([user_input, item_input], r_hat)
    model.compile(loss='mse', optimizer='Adam')
    return model

train = pd.read_csv('train.csv')
train = train.sample(frac=1).reset_index(drop=True)
test = pd.read_csv('test.csv')
matrix = pd.concat([train,test]).pivot('UserID','MovieID','Rating')
filepath="mfta-{epoch:02d}-{val_loss:.4f}.h5"
checkpoint = [ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')]
model = get_model(len(matrix.index), len(matrix.columns))
history = model.fit(x=[train['UserID'], train['MovieID']], y=train['Rating'], validation_split=0.1, epochs=8, verbose=1, batch_size=1024, callbacks=checkpoint)