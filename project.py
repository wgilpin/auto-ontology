# %%
import numpy as np
from transformers import TFDistilBertModel
from extract_bert_features import embed, make_pipe
from model_k import compile_ae_model
from scipy.spatial.distance import cosine


# %%
ae_model = compile_ae_model()

# some fake data
fake_train_x = np.random.rand(1000, 768)
fake_test_x = np.random.rand(500, 768)
fake_train_y = np.random.rand(1000, 1)
fake_test_y = np.random.rand(500, 1)
ae_model.fit(
    x=fake_train_x,
    y=fake_train_y,
    epochs=5,
    batch_size=100,
    shuffle=True,
    validation_data=(fake_test_x, fake_test_y))

# %%
z = ['The brown fox jumped over the dog',
     'The ship sank in the Atlantic Ocean',
     'Urban foxes sometimes fight with dogs']
emb_pipe = make_pipe('distilbert-base-uncased', TFDistilBertModel)
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
