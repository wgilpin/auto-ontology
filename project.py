# %%
from wandb.keras import WandbCallback
import wandb
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model
from tensorflow import convert_to_tensor
import numpy as np
from transformers import TFDistilBertModel
from extract_bert_features import embed, make_pipe
from model_k import create_model
from scipy.spatial.distance import cosine
from tensorflow.keras.callbacks import EarlyStopping
from progress.bar import Bar
import pandas as pd

from timer import timer

import matplotlib.pyplot as plt

from spacyNER import training_data_per_entity_spacy


# %%



# %% [markdown]
# # IMPORT import df from file
# 

# %%
import pandas as pd
from data import get_training_data

# train_x/y has only the embeddings
train_x, test_x, train_y, test_y = get_training_data(
    save_dir="results",
    radius=10,
    fraction=0.99,
    count=-1,
    force_recreate=True,
    entity_filter=['PER','ORG','LOC'])


# %%
# plot loss
# summarize history for loss
def plot_loss(history) -> None:
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('validation')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# %% [markdown]
# grid search

# %%
default = {
    "ae": [
        {"n": 768, "act": "relu"},
        {"n": 500, "act": "relu"},
        {"n": 2000, "act": "relu"},
    ],
    "repr": [
        {"n": 2000, "act": "relu"},
        {"n": 768, "act": "relu"},
        {"n": 768, "act": "relu"},
    ],
    "latent": {"n": 160, "act": "relu"},
    "output": "sigmoid",
    "opt": "adam",
    "loss": "mse",
    "lr": 0.001,
    "batch": 256
}

grid = [
    # big
    {
        "name": "4 deep big - 160 latent",
        **default,

        "output": "tanh",
    },
    # sigmoid
    # {
    #     "name": "1 deep - 160 latent",
    #     **default,
    #     "layers": [
    #         {"n": 768, "act": "relu"},
    #     ],
    #     "latent": {"n": 160, "act": "relu"},
    # },
    # {
    #     "name": "2 deep - 160 latent",
    #     **default,
    #     "layers": [
    #         {"n": 768, "act": "relu"},
    #         {"n": 400, "act": "relu"},
    #     ],
    #     "latent": {"n": 160, "act": "relu"},
    # },
    # {
    #     "name": "3 deep - 160 latent",
    #     **default,
    #     "layers": [
    #         {"n": 768, "act": "relu"},
    #         {"n": 512, "act": "relu"},
    #         {"n": 256, "act": "relu"},
    #     ],
    #     "latent": {"n": 160, "act": "relu"},
    # },
    # tanh
    # {
    #     "name": "1 deep - 160 latent - tanh",
    #     **default,
    #     "layers": [
    #         {"n": 768, "act": "relu"},
    #     ],
    #     "latent": {"n": 160, "act": "relu"},
    #     "output": "tanh",
    # },
    # {
    #     "name": "2 deep - 160 latent - tanh",
    #     **default,
    #     "layers": [
    #         {"n": 768, "act": "relu"},
    #         {"n": 400, "act": "relu"},
    #     ],
    #     "latent": {"n": 160, "act": "relu"},
    #     "output": "tanh",
    # },
    # {
    #     "name": "3 deep - 160 latent - tanh",
    #     **default,
    #     "layers": [
    #         {"n": 768, "act": "relu"},
    #         {"n": 512, "act": "relu"},
    #         {"n": 256, "act": "relu"},
    #     ],
    #     "latent": {"n": 160, "act": "relu"},
    #     "output": "tanh",
    # },


]


# %%
model = create_model(grid[0])
plot_model(model, show_dtype=True, 
            show_layer_names=True, show_shapes=True)

# %%


# %%

early_stopping_cb = EarlyStopping(
    monitor='loss', patience=15, verbose=1, min_delta=0.0001)

for config in grid:
    train_x, test_x = test_train_split()
    model = create_model(config)
    print(f"Training {config['name']}")
    plot_model(model, show_dtype=True, 
                       show_layer_names=True, show_shapes=True)
    history = model.fit(
        train_x,
        train_x,
        validation_data=(test_x, test_x),
        epochs=2000,
        batch_size=config["batch"],
        shuffle=True,
        verbose=0,
        callbacks=[early_stopping_cb],
    )
    print(f"...Loss => {history.history['loss'][-1]}")
    plot_loss(history)

# %%
# find best latent dims
early_stopping_cb = EarlyStopping(
    monitor='loss', patience=10, verbose=1, min_delta=0.00001)

best_loss = {"dims": 0, "loss": 100}
test_dims = [16, 32, 48, 64, 96, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 256, 512, 768]
# test_dims = range(140, 180, 1)
# test_dims = [168]
test_batches = [1024]
test_history = []
for latent_dim in test_dims:
    for batch_size in test_batches:
        model = create_model(
            layers=[
                {"n": 768, "act": "relu"},
                # {"n": 512, "act": "relu"},
                # {"n": 256, "act": "relu"},
            ],
            latent_layer={"n": latent_dim, "act": "relu"},
            output_fn="tanh",
            optimizer_fn="adam",
            loss_fn="mse",
            verbose=0)
        print(f"Training {latent_dim} latent dims, batch: {batch_size}")
        history = model.fit(
            train_x,
            train_x,
            validation_data=(test_x, test_x),
            epochs=2000,
            batch_size=1024,
            validation_split=0.1,
            shuffle=True,
            verbose=0,
            callbacks=[early_stopping_cb],
        )
        loss = history.history['val_loss'][-1]
        print(f"...Loss => {loss}")
        test_history.append(loss)
        if loss < best_loss["loss"]:
            best_loss = {
                "dims": latent_dim, "loss": loss, "batch_size": batch_size}

# results
plt.plot(test_dims, test_history)
plt.title('autoencoder loss vs latent dims')
plt.ylabel('loss')
plt.xlabel('dimensions')
plt.show()
print(
    f"Best latent dims: {best_loss['dims']}, "
    f"batch:{best_loss['batch_size']} => {best_loss['loss']}")

# %% [markdown]
# # DEC approach

# %%
from train_DEC import run_model

# %% [markdown]
# ## Spacy fit on CoNLL

# %%
run_model("conll")


# %%
run_model("conll")

# %% [markdown]
# 
# # fewNERD data

# %%
run_model("fewNERD")

# %%
from data import pre_embed


df = pre_embed("fewNERD")

# %%
df.head()


# %%
from spacyNER import training_data_from_embeds_spacy
    
training_data_from_embeds_spacy(df[0:5],0)[1]


# %% [markdown]
# # fewNERD with Spacy NER

# %%
run_model("fewNERD_spacy", 10000, force_recreate=True, radius=0)

# %% [markdown]
# # DEC+Spacy

# %%
%reload_ext autoreload
%autoreload 2

# compare
import numpy as np
import pandas as pd 
import os
from spacyNER import get_training_data_spacy
from data_conll import get_sample_conll_hf
from train_DEC import entity_types

size = 10000
filename = f"./data/conll_spacy_{size}.pkl"
if os.path.exists(filename):
    print(f"Loading {filename}")
    trg = pd.read_pickle(filename)
else:
    sample_conll = get_sample_conll_hf(size)

    trg = get_training_data_spacy(sample_conll, 0, entity_filter=entity_types)
    trg.to_pickle(filename)

print(f'Done: {trg.shape}')


# %%
from data import test_train_split

x, _, y, _ = test_train_split(trg)
print(f"x: {x.shape}, y: {y.shape}")

# %%
from train_DEC import train_DEC
from metrics import acc

# %%timeit 
dec = train_DEC(x, y)
update_interval = 30
y_pred = dec.fit(x, y=y, tol=0.001, maxiter=2e4, batch_size=512,
    update_interval=update_interval, save_dir='./data')

print('acc:', acc(y, y_pred))

# %%
import matplotlib.pyplot as plt
def plot_loss(history) -> None:
    plt.plot(history['loss'])
    plt.plot(history['acc'])
    plt.plot(history['delta_label'])
    plt.plot(history['correct_label'])

    plt.title('model loss')
    plt.ylabel('validation')
    plt.xlabel('epoch')
    plt.legend(['loss', 'accuracy', 'delta', 'correct'], loc='upper left')
    plt.show()

# %%

plot_loss(dec.history)

# %%
n = 1000
comp = y[0:n]==y_pred[0:n]
print (comp.sum(), comp.sum()/n)

# %% [markdown]
# # Deep Clustering

# %%
%reload_ext autoreload
%autoreload 2

# compare
import numpy as np
import pandas as pd 
import os
from spacyNER import get_training_data_spacy
from data_conll import get_sample_conll_hf
from train_DEC import entity_types

size = 100
filename = f"./data/conll_spacy_{size}.pkl"
if os.path.exists(filename):
    print(f"Loading {filename}")
    trg = pd.read_pickle(filename)
else:
    sample_conll = get_sample_conll_hf(size)

    trg = get_training_data_spacy(sample_conll, 0, entity_filter=entity_types)
    trg.to_pickle(filename)

print(f'Done: {trg.shape}')

# %%
from data import test_train_split

x, _, y, _ = test_train_split(trg)
print(f"x: {x.shape}, y: {y.shape}")

# %%
%reload_ext autoreload
%autoreload 2
from model_tf2 import ClusterNetwork

reuters = ClusterNetwork(
    latent_dim = 10,
    latent_weight = 0.001,
    noise_factor = 0.4,
    keep_prob = 1.0,
    alpha1=20,
    alpha2=1,
    optimizer='adam',
    learning_rate=0.001,
    n_clusters=4,
).train(x, y, train_batch_size=100, pretrain_epochs=5, train_epochs=50)


# %% [markdown]
# # TF2 from scratch

# %%
from tensorflow import keras as k
import tensorflow as tf
import numpy as np

# %%
from random import shuffle
from data import test_train_split

def load_data():
    size = 100
    filename = f"./data/conll_spacy_{size}.pkl"
    if os.path.exists(filename):
        print(f"Loading {filename}")
        trg = pd.read_pickle(filename)
    else:
        sample_conll = get_sample_conll_hf(size)

        trg = get_training_data_spacy(sample_conll, 0, entity_filter=entity_types)
        trg.to_pickle(filename)

    # shuffle
    trg = trg.sample(frac=1)

    x, _, y, _ = test_train_split(trg)
    print(f"x: {x.shape}, y: {y.shape}")



# %%


# %%
embeddings_dims = 768
latent_dims = 256

input_layer = k.Input(shape=(embeddings_dims,))

enc1 = tf.keras.layers.Dense(500)(input_layer)
enc2 = tf.keras.layers.Dense(500)(enc1)
enc3 = tf.keras.layers.Dense(2000)(enc2)

enc_X_out_logits = tf.keras.layers.Dense(latent_dims)(enc3)
encoder = tf.nn.sigmoid(enc_X_out_logits)
encoder_model = tf.keras.Model(inputs=input_layer, outputs=encoder)
encoder_model.summary()


# %%

dec1 = tf.keras.layers.Dense(2000)(enc_X_out_logits)
dec2 = tf.keras.layers.Dense(500)(dec1)
dec4 = tf.keras.layers.Dense(500)(dec2)

dec_X_out_logits = tf.keras.layers.Dense(embeddings_dims)(dec4)
decoder = tf.nn.sigmoid(dec_X_out_logits)
ae_model = k.models.Model(inputs=input_layer, outputs=decoder)
ae_model.summary()


# %%

lat1 = tf.keras.layers.Dense(2000)(enc_X_out_logits)
lat2 = tf.keras.layers.Dense(500)(lat1)
lat3 = tf.keras.layers.Dense(500)(lat2)
lat4 = tf.keras.layers.Dense(500)(lat3)
latent_network = tf.keras.layers.Dense(latent_dims)(lat4)
latent_model = k.models.Model(inputs=input_layer, outputs=latent_network)
latent_model.summary()

model = k.models.Model(inputs=input_layer, outputs=[decoder, latent_network])

# %%
@tf.function
def pairwise_sqd_distance(X, batch_size):

    tiled = tf.tile(tf.expand_dims(X, axis=1),
                    tf.stack([1, x.shape[0], 1]))
    tiled_trans = tf.transpose(tiled, perm=[1, 0, 2])
    diffs = tiled - tiled_trans
    sqd_dist_mat = tf.reduce_sum(tf.square(diffs), axis=2)
    tf.print(sqd_dist_mat, [sqd_dist_mat])

    return sqd_dist_mat

# %%
@tf.function
def make_q(z, batch_size, alpha):

    sqd_dist_mat = pairwise_sqd_distance(z, batch_size)
    q = tf.pow((1 + sqd_dist_mat/alpha), -(alpha+1)/2)
    q = tf.linalg.set_diag(q, tf.zeros(shape=[batch_size]))
    q = q / tf.reduce_sum(q, axis=0, keepdims=True)
    #q = 0.5*(q + tf.transpose(q))
    q = tf.clip_by_value(q, 1e-10, 1.0)

    return q

# %%
@tf.function
def latent_model(z_enc, training):
    z = latent_network(z_enc, training)

    p = make_q(z_enc, batch_size, alpha=alpha1)
    q = make_q(z, batch_size, alpha=alpha2)

    latent_loss = tf.reduce_sum(-(tf.multiply(p, tf.math.log(q))))


# %%
@tf.function
def ae_model_iter(X, training, rec_loss="mse"):

        # Add noise to the input to feed to Denoising encoder model.
        X_noisy = X + noise_factor * \
            tf.random.normal(shape=tf.shape(X), mean=0.0,
                             stddev=1.0, dtype=tf.float64)
        X_noisy = tf.clip_by_value(X, 0.0, 1.0)

        # Pass through encoder and decoder.
        z = encoder(X_noisy, training)
        X_out_logits, X_out = decoder(z, training)

        # Calculate Reconstruction loss.
        if rec_loss == 'mse':
            reconstr_loss = tf.reduce_mean(
                tf.math.squared_difference(X, X_out), axis=1)
        else:
            reconstr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=X, logits=X_out_logits), axis=1)
        reconstr_loss = tf.reduce_mean(reconstr_loss)

        return reconstr_loss, z
        

# %%
def batches(x, y, n):
    print(f"batches: {n} of {x.shape}, {y.shape}")
    for i in range(0, (y.shape[0] // n)-1):
        print(f"batch {i*n},{(i*n)+n}")
        yield x[i*n:(i*n)+n], y[i*n:(i*n)+n]

# %%
losses = []
alpha1 = 20
alpha2 = 1
rec_weight = 1.0
latent_weight = 0.01
batch_size = 32
for batch_idx, (x_batch, y_batch) in enumerate(batches(x, y, n=batch_size)):
    with tf.GradientTape() as tape:
        y_reconst = ae_model(x_batch, training=True)
        z_enc = encoder_model(x_batch, training=True)
        l = latent_model(x_batch, training=True)

        p = make_q(l, batch_size, alpha=alpha1)
        q = make_q(z_enc, batch_size, alpha=alpha2)
        latent_loss = tf.reduce_sum(-(tf.multiply(p, tf.compat.v1.log(q))))
        reconstr_loss = tf.reduce_mean(
            tf.math.squared_difference(x_batch.astype('float32'), y_reconst), axis=1)

        # Joint loss.
        joint_loss = rec_weight * reconstr_loss +\
                     latent_weight * latent_loss
        losses.append(joint_loss)
        tf.py_function(some_function, losses, [tf.float32]) # where the last argument
        
    gradients = tape.gradient(joint_loss, tape.watched_variables())
    # apply gradients for all variables watched by the tape
    opt.apply_gradients((grad, var)
                        for (grad, var) in zip(gradients, tape.watched_variables())
                        if grad is not None)

print(losses[-1][0])

# %%



