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

from spacyNER import get_spacy_NER_data


# %%

emb_pipe = make_pipe('distilbert-base-uncased', TFDistilBertModel)

early_stopping_cb = EarlyStopping(
    monitor='loss', patience=3, verbose=1, min_delta=0.001)

# %%

# test
def test():
    z = ['The brown fox jumped over the dog',
         'The ship sank in the Atlantic Ocean',
         'Urban foxes sometimes fight with dogs']
    print("Start")
    embedding_features1 = embed(emb_pipe, z[0])
    print(
        f"emb shape is {embedding_features1.shape} for {len(z[0].split(' '))} words")
    embedding_features2 = embed(emb_pipe, z[1])
    embedding_features3 = embed(emb_pipe, z[2])
    distance12 = 1-cosine(embedding_features1[0], embedding_features2[0])
    distance23 = 1-cosine(embedding_features2[0], embedding_features3[0])
    distance13 = 1-cosine(embedding_features1[0], embedding_features3[0])
    print(f"distance 1-2 {distance12}")
    print(f"distance 2-3 {distance23}")
    print(f"distance 1-3 {distance13}")

# %%
# create 1 training item per entity, given a token radius

@timer
def create_training_data_per_entity(
        length: int = 10,
        radius: int = 7) -> list:
    """
    Creates training data for the autoencoder.
    """
    from tqdm import tqdm

    print(f"Create Training Data for {length} items, radius {radius}")
    # get spacy data
    spacy_data = get_spacy_NER_data(length)
    print(f"Created NER Data for {len(spacy_data)} items")

    # get embedding data
    res = []
    with tqdm(total=len(spacy_data)) as pbar:
        for s in spacy_data:
            for chunk in s["chunks"]:
                start_token = max(chunk["entity"].start-radius, 0)
                end_token = min(chunk["entity"].end+radius, len(s["nlp"]))
                short_sentence = s["nlp"][start_token:end_token]
                res.append({
                    "sentence": short_sentence,
                    "chunk": str(chunk["chunk"]),
                    "label": chunk["entity"].label,
                    "label_id": chunk["entity"].label_,
                    "embedding": embed(emb_pipe, str(short_sentence))
                })
                pbar.update(1)
    # average all the embeddings in a sample, dropping 1st (CLS)
    for r in res:
        r["embedding"] = np.mean(r["embedding"][1:], axis=0)
    return res



# %%
# 
# CREATE TRAINING DATA
merged = create_training_data_per_entity(-1, 5)
print(len(merged))


# %%
# create full DF

x_train_np = np.stack([m["embedding"] for m in merged])

merged_df = pd.DataFrame(merged)
emb_df = pd.DataFrame(x_train_np)

merged_df.drop(columns=['embedding'], inplace=True)
df = pd.concat([merged_df, emb_df], axis=1)

df['label'] = np.unique(df['label'], return_inverse = True)[1]


print(df.shape)
# save data to csv
df.to_csv("./data/training_df.csv", index=False)
df.head()

# %% [markdown]
# # IMPORT import df from file
# 

# %%
import pandas as pd

df = pd.read_csv("./data/training_df.csv")
# train df has only the embeddings

def test_train_split(frac=0.8):
    training_data = df.sample(frac=frac, random_state=42)
    testing_data = df.drop(training_data.index)
    train_y_df = np.array(training_data["label"])
    test_y_df = np.array(testing_data["label"])
    train_x_df = np.array(training_data.drop(columns=['chunk', 'label', 'label_id', 'sentence']))
    test_x_df = np.array(testing_data.drop(columns=['chunk', 'label', 'label_id', 'sentence']))
    print(f"Full df: {df.shape}")
    print(f"Train df: {train_x_df.shape}")
    print(f"Test df: {test_x_df.shape}")
    return train_x_df, test_x_df, train_y_df, test_y_df


# %%
train_x, test_x, train_y, test_y = test_train_split(0.8)

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

save_dir = "./results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# load dataset
dataset = "conll"
if dataset == "reuters":
    from datasets import load_reuters
    x, y = load_reuters()
else:
    x, test_x, y, test_y = test_train_split(0.99)
print(f"{dataset} dataset: {x.shape} x:{type(x)}, y:{type(y)}")
n_clusters = len(np.unique(y))
print(f"{n_clusters} clusters")

init = 'glorot_uniform'
pretrain_optimizer = 'adam'
# setting parameters

update_interval = 30
pretrain_epochs = 50
init = VarianceScaling(scale=1. / 3., mode='fan_in',
                        distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
pretrain_optimizer = SGD(lr=1, momentum=0.9)

# %%
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


# %%
%%timeit 
y_pred = dec.fit(x, y=y, tol=0.001, maxiter=2e4, batch_size=256,
    update_interval=update_interval, save_dir=save_dir)
print('acc:', acc(y, y_pred))



