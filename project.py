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

from spacyNER import create_training_data_per_entity_spacy


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
import os
import numpy as np
from keras.initializers import VarianceScaling
from tensorflow.keras.optimizers import SGD
from DEC import DEC
from metrics import acc
from data import get_training_data

entity_types=['PER','ORG','LOC', 'MISC']

save_dir = "./results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# load dataset
dataset = "conll"
if dataset == "reuters":
    from datasets_deep import load_reuters
    x, y = load_reuters()
else:
    x, test_x, y, test_y = get_training_data(save_dir="results",
                                             count=10000,
                                             source="fewNERD",
                                             radius=5,
                                             fraction=0.99,
                                             entity_filter=entity_types,
                                             force_recreate=False)
x.shape

# %%
def train_DEC(x, y):
    print(f"{dataset} dataset: {x.shape} x:{type(x)}, y:{type(y)}")
    assert(len(np.unique(y)) <= len(entity_types))
    n_clusters = len(np.unique(y))
    print(f"{n_clusters} clusters")

    init = 'glorot_uniform'
    pretrain_optimizer = 'adam'
    # setting parameters

    pretrain_epochs = 50
    init = VarianceScaling(scale=1. / 3., mode='fan_in',
                        distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
    pretrain_optimizer = SGD(learning_rate=1, momentum=0.9)

    # prepare the DEC model
    dec = DEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, init=init)

    if os.path.exists(os.path.join(save_dir, 'ae_weights.h5')):
        print("Loading weights")
        dec.autoencoder.load_weights(os.path.join(save_dir, 'ae_weights.h5'))
    else:
        print("Training weights")
        dec.pretrain(x=x, y=y, optimizer=pretrain_optimizer,
                        epochs=pretrain_epochs, batch_size=256,
                        save_dir='./results')

    dec.model.summary()
    dec.compile(optimizer=SGD(0.01, 0.9), loss='kld')

    return dec

dec = train_DEC(x, y)

# %%
%%timeit 
update_interval = 30
y_pred = dec.fit(x, y=y, tol=0.001, maxiter=2e4, batch_size=256,
    update_interval=update_interval, save_dir=save_dir)
print('acc:', acc(y, y_pred))


# %% [markdown]
# 
# # fewNERD data

# %%
import numpy as np
from timer import timer
from fewNERD_data import create_training_data_per_entity_fewNERD


# %%
training_data = create_training_data_per_entity_fewNERD(length=50, radius=10, entity_filter=[1,2,3,4])
print("Training data created")

# %%



