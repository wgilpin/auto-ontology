# %%
import numpy as np
from transformers import TFDistilBertModel
from extract_bert_features import embed, make_pipe
from model_k import compile_ae_model, create_models_from_params
from scipy.spatial.distance import cosine
from timer import timer

from spacyNER import get_spacy_NER_data

emb_pipe = make_pipe('distilbert-base-uncased', TFDistilBertModel)

# %%
ae_model = compile_ae_model()

# # some fake data
# fake_train_x = np.random.rand(1000, 768)
# fake_test_x = np.random.rand(500, 768)
# fake_train_y = np.random.rand(1000, 1)
# fake_test_y = np.random.rand(500, 1)
# ae_model.fit(
#     x=fake_train_x,
#     y=fake_train_y,
#     epochs=5,
#     batch_size=100,
#     shuffle=True,
#     validation_data=(fake_test_x, fake_test_y))

# %%
def test():
    z = ['The brown fox jumped over the dog',
        'The ship sank in the Atlantic Ocean',
        'Urban foxes sometimes fight with dogs']
    print("Start")
    embedding_features1 = embed(emb_pipe, z[0])
    print(f"emb shape is {embedding_features1.shape} for {len(z[0].split(' '))} words")
    embedding_features2 = embed(emb_pipe, z[1])
    embedding_features3 = embed(emb_pipe, z[2])
    distance12 = 1-cosine(embedding_features1[0], embedding_features2[0])
    distance23 = 1-cosine(embedding_features2[0], embedding_features3[0])
    distance13 = 1-cosine(embedding_features1[0], embedding_features3[0])
    print(f"distance 1-2 {distance12}")
    print(f"distance 2-3 {distance23}")
    print(f"distance 1-3 {distance13}")
# %%

# %%
# create 1 training item per entity, given a token radius
@timer
def create_training_data_per_entity(
        length: int = 10,
        radius: int=7) -> list:
    """
    Creates training data for the autoencoder.
    """
    # get spacy data
    spacy_data = get_spacy_NER_data(length)
    # get embedding data
    res = []
    for s in spacy_data:
        for chunk in s["chunks"]:
            start_token = max(chunk["entity"].start-radius, 0)
            end_token = min(chunk["entity"].end+radius, len(s["nlp"]))
            short_sentence = s["nlp"][start_token:end_token]
            res.append({
                "sentence": short_sentence,
                "chunk": str(chunk["chunk"]),
                "label": chunk["entity"].label_,
                "embedding": embed(emb_pipe, str(short_sentence))
                })
    # average all the embeddings in a sample, dropping 1st (CLS)
    for r in res:
        r["embedding"] = np.mean(r["embedding"][1:], axis=0)
    return res
# %%
merged = create_training_data_per_entity(20, 5)
print(len(merged))

# for m in merged:
#     # pad the embedding to the longest one
#     add_zeros = longest_embedding - len(m["embedding"])
#     m["embedding"] = np.pad(m["embedding"], (0,add_zeros))
for m in merged[0:5]:
    print(f"{m['sentence']} [{m['chunk']}:{m['label']}] ({len(m['embedding'])})")
# %%

# training the autoencoder
model = compile_ae_model()
x_train = np.stack([m["embedding"] for m in merged])

print(x_train.shape)
print(x_train[0].shape)
# %%
model.fit(x_train, x_train, epochs=100, batch_size=100, shuffle=True)

# %%
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import wandb
from wandb.keras import WandbCallback

wandb.init(project="bert-ae")

input_dims = 768
latent_dims = 256
input_layer = Input(shape=(input_dims,))
e1 = Dense(input_dims*3/4, name="encoder_input", activation='relu')(input_layer)
e2 = Dense(input_dims/2, name="encoder_hidden", activation='relu')(e1)
encoder = Dense(latent_dims, name="encoder_output", activation='relu')(e2)
d1 = Dense(input_dims/2, name="decoder_input", activation='relu')(encoder)
d2 = Dense(input_dims*3/4, name="decoder_hidden", activation='relu')(d1)
decoder_logits = Dense(768, activation='tanh')(d2)
model = Model(inputs=input_layer, outputs=[decoder_logits])
optimizer = Adam(lr=0.001, decay=1e-6)

model.compile(optimizer="adam",loss='mse')

callback = EarlyStopping(monitor='loss', patience=10, verbose=1)

print(model.summary())

wandb.config = {
    "project": "MSc Project",
    "learning_rate": 0.001,
    "epochs": 2000,
    "batch_size": 256
}

# ... Define a model

history = model.fit(
                x_train,
                x_train,
                epochs=wandb.config["epochs"],
                batch_size=wandb.config["batch_size"],
                validation_split = 0.1,
                shuffle=True,
                callbacks=[callback, WandbCallback()])
wandb.finish()
# %%
layers1 = [
    {"n": 768, "act": "relu"},
    {"n": 400, "act": "relu"},
]
latent_layer1 = {"n": 128, "act": "relu"}
output_fn1 = "sigmoid"
optimizer_fn1 = "adam"
loss_fn1 = "mse"

model1 = create_models_from_params(layers1, latent_layer1, output_fn1, optimizer_fn1, loss_fn1)
wandb.init(project="bert-ae")
wandb.config = {
    "project": "MSc Project",
    "learning_rate": 0.001,
    "epochs": 2000,
    "batch_size": 256
}
model1.fit(
            x_train,
            x_train,
            epochs=wandb.config["epochs"],
            batch_size=wandb.config["batch_size"],
            validation_split = 0.1,
            shuffle=True,
            callbacks=[callback, WandbCallback()])
wandb.finish()
# %%
