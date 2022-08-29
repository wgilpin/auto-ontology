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
from extract_bert_features import embed, get_pipe
from model_k import create_model
from scipy.spatial.distance import cosine
from tensorflow.keras.callbacks import EarlyStopping
from progress.bar import Bar
import pandas as pd

from timer import timer

import matplotlib.pyplot as plt

from spacyNER import training_data_per_entity_spacy


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
import pickle

from data import load_data


# %%
x, y, mapping = load_data(1000)

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
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

# %%
from random import shuffle
from data import test_train_split

def load_data(size: int):
    filename_data = f"./data/conll_spacy_{size}.pkl"
    filename_mapping = f"./data/conll_spacy_{size}_map.pkl"
    if os.path.exists(filename_data) and os.path.exists(filename_mapping):
        print(f"Loading {filename_data}")
        trg = pd.read_pickle(filename_data)
        with open(filename_mapping, 'rb') as handle:
            mapping = pickle.load(handle)
    else:
        sample_conll = get_sample_conll_hf(size)

        trg, mapping = get_training_data_spacy(sample_conll, 0, entity_filter=entity_types)
        trg.to_pickle(filename_data)
        with open(filename_mapping, 'wb') as handle:
            pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print(f'Done: {trg.shape}')
    print(mapping)


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


# %%

dec1 = tf.keras.layers.Dense(2000)(enc_X_out_logits)
dec2 = tf.keras.layers.Dense(500)(dec1)
dec4 = tf.keras.layers.Dense(500)(dec2)

dec_X_out_logits = tf.keras.layers.Dense(embeddings_dims)(dec4)
decoder = tf.nn.sigmoid(dec_X_out_logits)
ae_model = k.models.Model(inputs=input_layer, outputs=decoder)


# %%

lat1 = tf.keras.layers.Dense(2000)(enc_X_out_logits)
lat2 = tf.keras.layers.Dense(500)(lat1)
lat3 = tf.keras.layers.Dense(500)(lat2)
lat4 = tf.keras.layers.Dense(500)(lat3)
latent_network = tf.keras.layers.Dense(latent_dims)(lat4)
latent_model = k.models.Model(inputs=input_layer, outputs=latent_network)

model = k.models.Model(inputs=input_layer, outputs=[decoder, latent_network])

opt = tf.keras.optimizers.Adam(learning_rate=0.001)

# %%
# @tf.function
# def latent_model(z_enc, training):
#     z = latent_network(z_enc, training)

#     p = make_q(z_enc, batch_size, alpha=alpha1)
#     q = make_q(z, batch_size, alpha=alpha2)

#     latent_loss = tf.reduce_sum(-(tf.multiply(p, tf.math.log(q))))


# %%
# @tf.function
# def ae_model_iter(X, training, rec_loss="mse"):

#         # Add noise to the input to feed to Denoising encoder model.
#         X_noisy = X + noise_factor * \
#             tf.random.normal(shape=tf.shape(X), mean=0.0,
#                              stddev=1.0, dtype=tf.float64)
#         X_noisy = tf.clip_by_value(X, 0.0, 1.0)

#         # Pass through encoder and decoder.
#         z = encoder(X_noisy, training)
#         X_out_logits, X_out = decoder(z, training)

#         # Calculate Reconstruction loss.
#         if rec_loss == 'mse':
#             reconstr_loss = tf.reduce_mean(
#                 tf.math.squared_difference(X, X_out), axis=1)
#         else:
#             reconstr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
#                 labels=X, logits=X_out_logits), axis=1)
#         reconstr_loss = tf.reduce_mean(reconstr_loss)

#         return reconstr_loss, z
        

# %%
@tf.function
def pairwise_sqd_distance(X, batch_size):
    tiled = tf.tile(tf.expand_dims(X, axis=1),
                    tf.stack([1, batch_size, 1]))
    tiled_trans = tf.transpose(tiled, perm=[1, 0, 2])
    diffs = tiled - tiled_trans
    sqd_dist_mat = tf.reduce_sum(tf.square(diffs), axis=2)

    return sqd_dist_mat

# %%
@tf.function
def make_q(z, batch_size, alpha):

    sqd_dist_mat = pairwise_sqd_distance(z, batch_size)
    q = tf.pow((1 + sqd_dist_mat/alpha), -(alpha+1)/2)
    q = tf.linalg.set_diag(q, tf.zeros(shape=[batch_size]))
    q = q / (tf.reduce_sum(q, axis=0, keepdims=True)+1e-9)
    #q = 0.5*(q + tf.transpose(q))
    q = tf.clip_by_value(q, 1e-10, 1.0)

    return q

# %%
def batches(x, y, n):
    for i in range(0, (y.shape[0] // n)-1):
        yield x[i*n:(i*n)+n], y[i*n:(i*n)+n]

# %%
x, y = load_data(100)

# %%
%matplotlib inline

losses = []
alpha1 = 20
alpha2 = 1
rec_weight = 1.0
latent_weight = 0.01
batch_size = 32
num_epochs = 2

# pretrain ae
print("Pretraining")
pretrain_batches = batches(x, y, batch_size)
for batch_idx, (x_batch, y_batch) in enumerate(pretrain_batches):
    if batch_idx > 20:
        break
    with tf.GradientTape() as tape:
        y_reconst = ae_model(x_batch, training=True)

        reconstr_loss = tf.reduce_mean(
            tf.math.squared_difference(x_batch.astype('float32'), y_reconst), axis=1)

    gradients = tape.gradient(reconstr_loss, tape.watched_variables())
    # apply gradients for all variables watched by the tape
    opt.apply_gradients((grad, var)
                        for (grad, var) in zip(gradients, tape.watched_variables())
                        if grad is not None)
print(f"Pretraining final loss {reconstr_loss}")

# train full model
print("Full training")
# Prepare the metrics.
train_acc_metric = k.metrics.SparseCategoricalAccuracy()
val_acc_metric = k.metrics.SparseCategoricalAccuracy()
for epoch in tqdm(range(num_epochs)):
    full_batches = batches(x, y, batch_size)
    for batch_idx, (x_batch, y_batch) in enumerate(full_batches):
        with tf.GradientTape() as tape:
            y_reconst = ae_model(x_batch, training=True)
            z_enc = encoder_model(x_batch, training=True)
            l = latent_model(x_batch, training=True)

            p = make_q(l, batch_size, alpha=alpha1)
            q = make_q(z_enc, batch_size, alpha=alpha2)
            latent_loss = tf.reduce_sum(-(tf.multiply(p, tf.math.log(q))))
            reconstr_loss = tf.reduce_mean(
                tf.math.squared_difference(x_batch.astype('float32'), y_reconst), axis=1)

            # Joint loss.
            joint_loss = rec_weight * reconstr_loss +\
                        latent_weight * latent_loss
            losses.append(joint_loss)
            # print(f"LOSS {losses[-1][0]}")
            
            gradients = tape.gradient(joint_loss, tape.watched_variables())
            # apply gradients for all variables watched by the tape
            opt.apply_gradients((grad, var)
                                for (grad, var) in zip(gradients, tape.watched_variables())
                                if grad is not None)
            # Update training metric.
            train_acc_metric.update_state(y_batch, l)

    # print(losses[-1][0])
    plt.plot(losses)

# %% [markdown]
# # Subclassed

# %%

import stat
import warnings 
warnings.filterwarnings("ignore")

from tensorflow import keras as k
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn import metrics
from sklearn import mixture
from linear_assignment import linear_assignment
from data import load_data

clustering = "DBSCAN"
n_clusters = 10
dbscan_eps = 0.2
dbscan_min_samples = 5

def cluster_metric(y, z_state):
    """
    y is labels, not one hot encoded vector.
    """

    if clustering == 'GMM':
        gmix = mixture.GaussianMixture(
            n_components=n_clusters, covariance_type='full')
        gmix.fit(z_state)
        y_pred = gmix.predict(z_state)
    elif clustering == 'Kmeans':
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        y_pred = kmeans.fit_predict(z_state)
    elif clustering == 'DBSCAN':
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        y_pred = dbscan.fit_predict(z_state)
    elif clustering == 'OPTICS':
        optics = OPTICS(min_samples=dbscan_min_samples)
        y_pred = optics.fit_predict(z_state)
    else:
        raise ValueError('Clustering algorithm not specified/unknown.')

    def cluster_acc(y_true, y_pred):
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)

    print(f'\n{clustering} Accuracy: {acc}, NMI: {nmi}, ARI: {ari}')

    # res = np.ndarray([acc], np.float32)
    return np.ndarray([], np.float32)

@tf.function
def tf_cluster_metric(y, z_state):
  y = tf.numpy_function(cluster_metric, [y, z_state], tf.float32)
  return y * y

@tf.function
def pairwise_sqd_distance(X, batch_size):
    tiled = tf.tile(tf.expand_dims(X, axis=1),
                    tf.stack([1, batch_size, 1]))
    tiled_trans = tf.transpose(tiled, perm=[1, 0, 2])
    diffs = tiled - tiled_trans
    sqd_dist_mat = tf.reduce_sum(tf.square(diffs), axis=2)

    return sqd_dist_mat


@tf.function
def make_q(z, batch_size, alpha):

    sqd_dist_mat = pairwise_sqd_distance(z, batch_size)
    q = tf.pow((1 + sqd_dist_mat/alpha), -(alpha+1)/2)
    q = tf.linalg.set_diag(q, tf.zeros(shape=[batch_size]))
    q = q / (tf.reduce_sum(q, axis=0, keepdims=True)+1e-9)
    #q = 0.5*(q + tf.transpose(q))
    q = tf.clip_by_value(q, 1e-10, 1.0)

    return q
    
loss_tracker = k.metrics.Mean(name="loss")

class DeepCluster(k.Model):

    def __init__(self, latent_dim, n_clusters, batch_size: int=256):
        super(DeepCluster, self).__init__()
        self.latent_dim = latent_dim
        self.embeddings_dims = 768
        self.latent_dims = 256
        self.alpha1 = 20
        self.alpha2 = 1
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.batch_size = batch_size
        self.noise_factor = 0.5
        self.rec_weight = 1.0
        self.latent_weight = 0.01
        self.clustering = 'Kmeans'
        self.n_clusters = n_clusters
        self.build_model()


    def build_model(self):
        
        input_layer = k.Input(shape=(self.embeddings_dims,))

        # encoder
        enc1 = tf.keras.layers.Dense(500, name="enc1")(input_layer)
        enc2 = tf.keras.layers.Dense(500, name="enc2")(enc1)
        enc3 = tf.keras.layers.Dense(2000, name="enc3")(enc2)
        enc_X_out_logits = tf.keras.layers.Dense(self.latent_dims, name="enc_logits")(enc3)
        self.encoder = tf.nn.sigmoid(enc_X_out_logits, name="enc_sigmoid")
        self.encoder_model = tf.keras.Model(inputs=input_layer, outputs=self.encoder, name="encoder")

        # decoder
        dec1 = tf.keras.layers.Dense(2000, name="dec1")(enc_X_out_logits)
        dec2 = tf.keras.layers.Dense(500, name="dec2")(dec1)
        dec4 = tf.keras.layers.Dense(500, name="dec3")(dec2)
        dec_X_out_logits = tf.keras.layers.Dense(self.embeddings_dims, name="dec_logits")(dec4)
        self.decoder = tf.nn.sigmoid(dec_X_out_logits, name="dec_sigmoid")
        self.ae_model = k.models.Model(inputs=input_layer, outputs=self.decoder, name="decoder")


        # latent net
        lat1 = tf.keras.layers.Dense(2000, name="lat1")(enc_X_out_logits)
        lat2 = tf.keras.layers.Dense(500, name="lat2")(lat1)
        lat3 = tf.keras.layers.Dense(500, name="lat3")(lat2)
        lat4 = tf.keras.layers.Dense(500, name="lat4")(lat3)
        latent_network = tf.keras.layers.Dense(self.latent_dims, name="lat_logits")(lat4)
        latent_sig = tf.nn.sigmoid(latent_network, name="lat_sigmoid")
        self.latent_model = k.models.Model(inputs=input_layer, outputs=latent_sig, name="latent_network")

    def compile(self):
        super(DeepCluster, self).compile(
                                    optimizer=self.opt, 
                                    # metrics=[self.cluster_metrics],
                                    )
    
    @staticmethod
    def squared_dist(A): 
        expanded_a = tf.expand_dims(A, 1)
        expanded_b = tf.expand_dims(A, 0)
        distances = tf.reduce_sum(tf.math.squared_difference(expanded_a, expanded_b), 2)
        return distances

    @staticmethod
    def pairwise_sqd_distance(X, batch_size):
        tiled = tf.tile(tf.expand_dims(X, axis=1),
                        tf.stack([1, batch_size, 1]))
        tiled_trans = tf.transpose(tiled, perm=[1, 0, 2])
        print(f"sub_1: {tiled.shape}, {tiled_trans.shape}")
        diffs = tiled - tiled_trans
        sqd_dist_mat = tf.reduce_sum(tf.square(diffs), axis=2)

        return sqd_dist_mat


    @staticmethod
    def make_q(z, batch_size, alpha):

        sqd_dist_mat = tf.reduce_sum((tf.expand_dims(z, 1)-tf.expand_dims(z, 0))**2,2)
        q = tf.pow((1 + sqd_dist_mat/alpha), -(alpha+1)/2)
        q = tf.linalg.set_diag(q, tf.zeros(shape=[batch_size]))
        q = q / (tf.reduce_sum(q, axis=0, keepdims=True)+1e-9)
        #q = 0.5*(q + tf.transpose(q))
        q = tf.clip_by_value(q, 1e-10, 1.0)

        return q

    def train_step(self, data):
        # Unpack the data passed to `fit()`.
        x, y = data

        # Add noise to the input to feed to Denoising encoder model.
        X_noisy = x + self.noise_factor * \
            tf.random.normal(shape=tf.shape(x), mean=0.0,
                             stddev=1.0, dtype=tf.float32)
        X_noisy = tf.clip_by_value(x, 0.0, 1.0)

        with tf.GradientTape() as tape:
            y_reconst = self.ae_model(x, training=True)
            z_enc = self.encoder_model(x, training=True)
            l = self.latent_model(x, training=True)

            p = self.make_q(z=l, batch_size=self.batch_size, alpha=self.alpha1)
            q = self.make_q(z=z_enc, batch_size=self.batch_size, alpha=self.alpha2)
            latent_loss = tf.reduce_sum(-(tf.multiply(p, tf.math.log(q))))


            reconstr_loss = tf.reduce_mean(
                tf.math.squared_difference(x, y_reconst))

            # Joint loss.
            # print(tf_cluster_metric(y, l))

            # print(f"latent {latent_loss}")
            # print(f"reconstr {reconstr_loss}")
            loss = tf.constant(self.rec_weight)*latent_loss
            # tf.constant(self.rec_weight)*reconstr_loss# +\
                # tf.constant(self.latent_weight)*latent_loss
            # loss = self.rec_weight * reconstr_loss +\
            #        self.latent_weight * latent_loss

        print(f"loss: {loss}")
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, l)
        # Return a dict mapping metric names to current value
        return {"loss": loss_tracker.result()}

    

# %%
x, y, mapping = load_data(10000)


# %%

!del /s /q .\\logs\\*

# Load the TensorBoard notebook extension.
%load_ext tensorboard

from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping


clustering = "GMM"
n_clusters = len(mapping)
dbscan_eps = 0.03
dbscan_min_samples = 10
# Define the Keras TensorBoard callback.
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = k.callbacks.TensorBoard(log_dir=logdir)
early_stopping_cb = EarlyStopping(
    monitor='loss', patience=15, verbose=1, min_delta=0.001)

dc = DeepCluster(latent_dim=256, n_clusters=n_clusters, batch_size=256)
dc.compile()
dc.fit(x, y, epochs=10, callbacks=[tensorboard_callback, early_stopping_cb])



# %%
dc.history.history

# %%
from metrics import acc, nmi, ari, plot_confusion

q = dc.predict(x, verbose=0)

# evaluate the clustering performance
y_pred = q.argmax(1)
if y is not None:
    acc = np.round(acc(y, y_pred), 5)
    nmi = np.round(nmi(y, y_pred), 5)
    ari = np.round(ari(y, y_pred), 5)
    print(f'Acc = {acc:.5f}, nmi = {nmi:.5f}, ari = {ari:.5f}')

# %%
from data import load_data


df = load_data(10000)

# %% [markdown]
# # Joint

# %%
%reload_ext autoreload
%autoreload 2

import numpy as np
from data import load_data


# %%

size = 10000
max_iter = 140
entities = ['ORG', 'NORP', 'PERSON', 'GPE', 'PRODUCT', 'EVENT' ]
x, y, mapping = load_data(size, entity_filter=entities)
n_clusters = len(np.unique(y))
print(x.shape)
assert(n_clusters <= len(entities))


# %%
from dec_model import (
    autoencoder_model, train_model, target_distribution,
    make_model, ClusteringLayer, pre_train)
filename = f"DEC_model_final_{size}_{max_iter}.h5"

# %%
autoencoder, encoder = make_model(x.shape[-1])

# %%
model, y_pred = pre_train(x, y, autoencoder, encoder)

# %%
train_model(x, y, model, y_pred, filename, max_iter=8000)

# %%
model.load_weights(os.path.join('results',filename))

# %%
from metrics import acc, nmi, ari, plot_confusion

q = model.predict(x, verbose=0)
p = target_distribution(q)  # update the auxiliary target distribution p

# evaluate the clustering performance
y_pred = q.argmax(1)
if y is not None:
    acc = np.round(acc(y, y_pred), 5)
    nmi = np.round(nmi(y, y_pred), 5)
    ari = np.round(ari(y, y_pred), 5)
    print(f'Acc = {acc:.5f}, nmi = {nmi:.5f}, ari = {ari:.5f}')

# %%
plot_confusion(y, y_pred, mapping, 8)

# %% [markdown]
# # TF with custom losses for P

# %%
import pickle
from tensorflow import keras as k
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

# %%
from random import shuffle
from data import test_train_split, load_data


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


dec1 = tf.keras.layers.Dense(2000)(enc_X_out_logits)
dec2 = tf.keras.layers.Dense(500)(dec1)
dec4 = tf.keras.layers.Dense(500)(dec2)

dec_X_out_logits = tf.keras.layers.Dense(embeddings_dims)(dec4)
decoder = tf.nn.sigmoid(dec_X_out_logits)
ae_model = k.models.Model(inputs=input_layer, outputs=decoder)


lat1 = tf.keras.layers.Dense(2000)(enc_X_out_logits)
lat2 = tf.keras.layers.Dense(500)(lat1)
lat3 = tf.keras.layers.Dense(500)(lat2)
lat4 = tf.keras.layers.Dense(500)(lat3)
latent_logits = tf.keras.layers.Dense(latent_dims)(lat4)
latent_network = tf.nn.sigmoid(latent_logits)
latent_model = k.models.Model(inputs=input_layer, outputs=latent_network)

model = k.models.Model(inputs=input_layer, outputs=[decoder, latent_network])

opt = tf.keras.optimizers.Adam(learning_rate=0.001)

# %%
@tf.function
def pairwise_sqd_distance(X, batch_size):
    tiled = tf.tile(tf.expand_dims(X, axis=1),
                    tf.stack([1, batch_size, 1]))
    tiled_trans = tf.transpose(tiled, perm=[1, 0, 2])
    diffs = tiled - tiled_trans
    sqd_dist_mat = tf.reduce_sum(tf.square(diffs), axis=2)

    return sqd_dist_mat

@tf.function
def make_q(z, batch_size, alpha):

    sqd_dist_mat = pairwise_sqd_distance(z, batch_size)
    q = tf.pow((1 + sqd_dist_mat/alpha), -(alpha+1)/2)
    q = tf.linalg.set_diag(q, tf.zeros(shape=[batch_size]))
    q = q / (tf.reduce_sum(q, axis=0, keepdims=True)+1e-9)
    #q = 0.5*(q + tf.transpose(q))
    q = tf.clip_by_value(q, 1e-10, 1.0)

    return q
    
def batches(x, y, n):
    for i in range(0, (y.shape[0] // n)-1):
        yield x[i*n:(i*n)+n], y[i*n:(i*n)+n]

# %%
entities = ['ORG', 'PERSON', 'GPE', 'NORP', 'PRODUCT', 'EVENT' ]
x, y, mapping = load_data(10000, entity_filter=entities)

# %%
%matplotlib inline

losses = []
alpha1 = 20
alpha2 = 1
rec_weight = 1
latent_weight = 1/600
batch_size = 128
num_epochs = 5

# pretrain ae
print("Pretraining")
pretrain_batches = batches(x, y, batch_size)
for batch_idx, (x_batch, y_batch) in enumerate(pretrain_batches):
    if batch_idx > 20:
        break
    with tf.GradientTape() as tape:
        y_reconst = ae_model(x_batch, training=True)

        reconstr_loss = tf.reduce_mean(
            tf.math.squared_difference(x_batch.astype('float32'), y_reconst))

    gradients = tape.gradient(reconstr_loss, tape.watched_variables())
    # apply gradients for all variables watched by the tape
    opt.apply_gradients((grad, var)
                        for (grad, var) in zip(gradients, tape.watched_variables())
                        if grad is not None)
print(f"Pretraining final loss {reconstr_loss}")

# train full model
print("Full training")
# Prepare the metrics.
train_acc_metric = k.metrics.SparseCategoricalAccuracy()
val_acc_metric = k.metrics.SparseCategoricalAccuracy()
for epoch in tqdm(range(num_epochs)):
    full_batches = batches(x, y, batch_size)
    for batch_idx, (x_batch, y_batch) in enumerate(full_batches):
        with tf.GradientTape() as tape:
            y_reconst = ae_model(x_batch, training=True)
            z_enc = encoder_model(x_batch, training=True)
            l = latent_model(x_batch, training=True)

            p = make_q(l, batch_size, alpha=alpha1)
            q = make_q(z_enc, batch_size, alpha=alpha2)
            latent_loss = tf.reduce_sum(-(tf.multiply(p, tf.math.log(q))))
            reconstr_loss = tf.reduce_mean(
                tf.math.squared_difference(x_batch.astype('float32'), y_reconst))

            # Joint loss.
            joint_loss = rec_weight * reconstr_loss +\
                        latent_weight * latent_loss
            # print(f"LOSS {losses[-1][0]}")
            
            gradients = tape.gradient(joint_loss, tape.watched_variables())
            # apply gradients for all variables watched by the tape
            opt.apply_gradients((grad, var)
                                for (grad, var) in zip(gradients, tape.watched_variables())
                                if grad is not None)
            # Update training metric.
        train_acc_metric.update_state(y_batch, l)
        losses.append(joint_loss)

    # print(losses[-1][0])
    plt.plot(losses)

# %%
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def cluster_metric(y, z_state, clustering):
    """
    y is labels, not one hot encoded vector.
    """

    if clustering == 'GMM':
        gmix = mixture.GaussianMixture(
            n_components=n_clusters, covariance_type='full')
        gmix.fit(z_state)
        y_pred = gmix.predict(z_state)
    elif clustering == 'Kmeans':
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        y_pred = kmeans.fit_predict(z_state)
    elif clustering == 'DBSCAN':
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        y_pred = dbscan.fit_predict(z_state)
    elif clustering == 'OPTICS':
        optics = OPTICS(min_samples=dbscan_min_samples)
        y_pred = optics.fit_predict(z_state)
    else:
        raise ValueError('Clustering algorithm not specified/unknown.')

    def cluster_acc(y_true, y_pred):
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(adjusted_rand_score(y, y_pred), 5)

    print(f'\n{clustering} Accuracy: {acc}, NMI: {nmi}, ARI: {ari}')

    # res = np.ndarray([acc], np.float32)
    return np.ndarray([], np.float32)

# %%
from metrics import plot_confusion
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn import mixture
from linear_assignment import linear_assignment

dbscan_eps = 0.2
dbscan_min_samples = 5
n_clusters = np.unique(y).shape[0]
q = latent_model.predict(x, verbose=0)


# evaluate the clustering performance
for cluster_algo in ["GMM", "Kmeans", "DBSCAN", "OPTICS"]:
    cluster_metric(y, q, cluster_algo)


