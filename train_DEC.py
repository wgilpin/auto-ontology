import os
import numpy as np
from keras.initializers import VarianceScaling
from tensorflow.keras.optimizers import SGD
from DEC import DEC
from metrics import acc
from data import get_training_data

entity_types=['PER', 'PERSON','ORG','LOC', 'GPE', 'MISC']

save_dir = "./results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# load dataset
datasets = ["conll", "reuters", "fewNERD"][0]

def train_DEC(x, y):
    print(f"dataset x:{x.shape} y:{y.shape}")
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

def load_dataset(dataset, length:int=0, force_recreate:bool=False, radius:int=0):
    if dataset == "reuters":
        from datasets_deep import load_reuters
        x, y = load_reuters()
    elif dataset in ["conll", "fewNERD", "fewNERD_spacy"]:
        x, _, y, _ = get_training_data(save_dir="results",
                                                count=length,
                                                source=dataset,
                                                radius=radius,
                                                fraction=0.99,
                                                entity_filter=entity_types,
                                                force_recreate=force_recreate)
    else:
        raise ValueError("Unknown dataset")

    return x, y

# %%timeit 
def run_model(dataset: str, length: int=0, force_recreate=False, radius: int=0) -> None:
    x, y = load_dataset(dataset, length, force_recreate=force_recreate, radius=radius)
    dec = train_DEC(x, y)
    update_interval = 30
    y_pred = dec.fit(x, y=y, tol=0.001, maxiter=2e4, batch_size=512,
        update_interval=update_interval, save_dir=save_dir)
    print('acc:', acc(y, y_pred))