import tensorflow as tf

from tensorflow import keras as k


def build_model(input_dims, latent_dims, output_dims):

    input_layer = k.layers.Input(shape=(input_dims,))

    # Encoder
    e1 = k.layers.Dense(input_dims*3/4, name="encoder_input")(input_layer)
    e2 = k.layers.Dense(input_dims/2, name="encoder_hidden")(e1)
    encoder = k.layers.Dense(latent_dims, name="encoder_output")(e2)

    # Decoder
    d1 = k.layers.Dense(input_dims/2, name="decoder_input")(encoder)
    d2 = k.layers.Dense(input_dims*3/4, name="decoder_hidden")(d1)
    decoder_logits = k.layers.Dense(output_dims, name="decoder_output")(d2)
    decoder = tf.nn.sigmoid(decoder_logits, name="decoder_sigmoid")

    # Latent network
    # l1 = k.layers.Dense(input_dims*3/2)(decoder)
    # l2 = k.layers.Dense(input_dims/2)(l1)
    # latent = k.layers.Dense(latent_dims)(l2)

    model = k.Model(inputs=input_layer, outputs=[decoder])

    return model


def compile_ae_model():
    optimizer = k.optimizers.Adam(lr=0.001, decay=1e-6)
    ae_model = build_model(input_dims=768, latent_dims=128, output_dims=256)
    ae_model.compile(optimizer, loss="mse")
    print(ae_model.summary())
    return ae_model
