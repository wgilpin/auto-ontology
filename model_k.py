# %%
from typing import Any
import tensorflow as tf

from tensorflow import keras as k
from tensorflow.keras.layers import Dense, Input


# %%

# create a Dense layer from a config
def create_layer(config: dict, name: str) -> Dense:
    """
    Create a layer from a config dictionary.
    - config: dictionary of layer parameters
        - n: number of units
        - act: activation function
    """
    return Dense(
        name=name,
        units=config["n"],
        activation=config["act"],
        kernel_regularizer='l1')

# %%


# %%

# build the autoencoder


def create_ae_encoder_model(
        emb_size: int,
        layers: list[dict],
        latent_layer: int) -> k.Model:
    """
    Create a model with the given parameters.
    - emb_size: the number of input dimensions
    - latent_layer: latent layer config: dict with keys:
          - n: int number of units
          - act: str activation function
    - layers: list of layers in the encoder. Decoder is mirrored.
        array of dict with keys:
          - n: int number of units
          - act: str activation function
    """

    # build ml, the array of layers
    ml = [Input(shape=(emb_size,))]
    for n, l in enumerate(layers):
        ml.append(create_layer(l, f"encoder_{n}")(ml[-1]))
    ml.append(create_layer(latent_layer, "ae_latent")(ml[-1]))

    return ml


def create_ae_decoder_model(
        input_layer: Dense,
        layers: list[dict],
        output_fn) -> k.Model:
    """
    Create a model with the given parameters.
    - layers: list of layers in the encoder. Reverse order.
        array of dict with keys:
          - n: int number of units
          - act: str activation function
    """

    # build ml, the array of layers
    ml = [create_layer(layers[0], "decoder_in")(input_layer)]
    for n, l in enumerate(reversed(layers[1:]), 1):
        ml.append(create_layer(l, f"decoder_{n}")(ml[-1]))

    out_config = {**layers[0], "act": output_fn}
    ml.append(create_layer(out_config, "decoder_out")(ml[-1]))
    return ml

# %%
# create the latent_space model


def create_latent_space_model(
        input_layer: Dense,
        layers: list[dict],
        output_fn: Any) -> k.Model:
    """
    Create a model with the given parameters.
    - input_layer: previous model output layer
    - layers: list of layers in the encoder. Decoder is mirrored.
        array of dict with keys:
          - n: int number of units
          - act: str activation function
    - output_fn: the output function to use
    """
    # build ml, the array of layers
    ml = [input_layer]
    for n, l in enumerate(layers):
        ml.append(create_layer(l, f"latent_{n}")(ml[-1]))

    out_config = {**layers[0], "act": output_fn}
    ml.append(create_layer(out_config, "latent_out")(ml[-1]))
    return ml

# %%
# CREATE THE MODEL


def create_model(cfg: dict, emb_size: int = 768, verbose: int = 0) -> k.Model:
    """
    Create a model with the given parameters.
    -cfg: the model configuration. Dict with keys:
        - latent: latent layer
        - ae: autoencoder layers
            array of dict with keys:
            - n: int number of units
            - act: str activation function
        - emb_size: the number of input dimensions
    """
    enc = create_ae_encoder_model(
        emb_size=emb_size,
        layers=cfg['ae'],
        latent_layer=cfg['latent'])
    dec = create_ae_decoder_model(
        input_layer=enc[-1],
        layers=cfg['ae'],
        output_fn=cfg['output'])
    latent_space = create_latent_space_model(
        input_layer=enc[-1],
        layers=cfg['ae'],
        output_fn='sigmoid')
    full_model = k.Model(
        inputs=enc[0],
        outputs=[
            dec[-1],
            latent_space[-1],
            ])

    full_model.compile(cfg['opt'], cfg['loss'])
    if verbose:
        print(full_model.summary())
    return full_model


# %%
