# %%
import tensorflow as tf

from tensorflow import keras as k
from tensorflow.keras.layers import Dense, Input

# %%
def build_model(input_dims, latent_dims):


    input_layer = k.layers.Input(shape=(input_dims,))

    # Encoder
    e1 = k.layers.Dense(input_dims*3/4, name="encoder_input")(input_layer)
    e2 = k.layers.Dense(input_dims/2, name="encoder_hidden")(e1)
    encoder = k.layers.Dense(latent_dims, name="encoder_output")(e2)

    # Decoder
    d1 = k.layers.Dense(input_dims/2, name="decoder_input")(encoder)
    d2 = k.layers.Dense(input_dims*3/4, name="decoder_hidden")(d1)
    decoder_logits = k.layers.Dense(input_dims, name="decoder_output")(d2)
    decoder = tf.nn.sigmoid(decoder_logits, name="decoder_sigmoid")

    # Latent network
    # l1 = k.layers.Dense(input_dims*3/2)(decoder)
    # l2 = k.layers.Dense(input_dims/2)(l1)
    # latent = k.layers.Dense(latent_dims)(l2)

    model = k.Model(inputs=input_layer, outputs=[decoder])

    return model


def compile_ae_model():
    optimizer = k.optimizers.Adam(lr=0.001, decay=1e-6)
    ae_model = build_model(input_dims=768, latent_dims=128)
    ae_model.compile(optimizer, loss="mse")
    print(ae_model.summary())
    return ae_model

# %%
def create_layer(config: dict) -> Dense:
    """
    Create a layer from a config dictionary.
    - config: dictionary of layer parameters
        - n: number of units
        - act: activation function
    """
    return Dense(units=config["n"], activation=config["act"])
    
# %%
def create_models_from_params(layers, latent_layer, output_fn, optimizer_fn, loss_fn):
    """
    Create a model with the given parameters.
    - input_dims: the number of input dimensions
    - latent_layer: latent layer config: dict with keys:
          - n: int number of units
          - act: str activation function
    - layers: list of layers in the encoder. Decoder is mirrored.
        array of dict with keys:
          - n: int number of units
          - act: str activation function
    - optimizer: the optimizer to use
    - loss: the loss function to use
    """

    # build ml, the array of layers
    ml = [Input(shape=(layers[0]["n"],))]
    for l in layers:
        ml.append(create_layer(l)(ml[-1]))
    ml.append(create_layer(latent_layer)(ml[-1]))
    for _, l in enumerate(reversed(layers[1:]), 1):
        ml.append(create_layer(l)(ml[-1]))

    out_config = {**layers[0], "act": output_fn}
    ml.append(create_layer((out_config))(ml[-1]))

    model = k.Model(inputs=ml[0], outputs=[ml[-1]])

    model.compile(optimizer_fn, loss_fn)
    print(model.summary())
    return model

# %%

# layers1 = [
#     {"n": 768, "act": "relu"},
#     {"n": 400, "act": "relu"},
# ]
# latent_layer1 = {"n": 128, "act": "relu"}
# output_fn1 = "sigmoid"
# optimizer_fn1 = k.optimizers.Adam(lr=0.001, decay=1e-6)
# loss_fn1 = "mse"

# model1 = create_models_from_params(layers1, latent_layer1, output_fn1, optimizer_fn1, loss_fn1)
# model1.summary()
# model1.compile(optimizer_fn1, loss_fn1)


# %%
