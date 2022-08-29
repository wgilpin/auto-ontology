import os
from scipy.optimize import linear_sum_assignment
import src.metrics as metrics
from sklearn.cluster import KMeans
from keras.initializers import VarianceScaling
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Layer, InputSpec
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
np.random.seed(10)

MAXITER = 8000
UPDATE_INTERVAL = 140
TOL = 0.001 # tolerance threshold to stop training
PRETRAIN_EPOCHS = 1000
BATCH_SIZE = 256
SAVE_DIR = './results'



def build_autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is
            input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the
            auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    input_img = Input(shape=(dims[0],), name='input')
    x = input_img
    # internal layers in encoder
    for i in range(n_stacks-1):
        x = Dense(dims[i + 1], activation=act,
                  kernel_initializer=init, name=f'encoder_{i}')(x)

    # hidden layer
    encoded = Dense(
        dims[-1],
        kernel_initializer=init,
        name=f"encoder_{n_stacks - 1}")(x)

    x = encoded
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        x = Dense(dims[i], activation=act,
                  kernel_initializer=init, name=f'decoder_{i}')(x)

    # output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
    decoded = x
    return input_img, decoded, encoded

def autoencoder_model(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is
            input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the
            auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    input_img, decoded, encoded = build_autoencoder(dims, act=act, init=init)
    return Model(inputs=input_img, outputs=decoded, name='AE'),\
        Model(inputs=input_img, outputs=encoded, name='encoder')


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a
    vector that represents the probability of the sample belonging to each 
    cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` 
            which represents the initial cluster centers. 
        alpha: degrees of freedom parameter in Student's t-distribution. 
            Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)
        self.clusters = None

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(
            shape=(self.n_clusters, input_dim),
            initializer='glorot_uniform',
            name='clusters',
        )
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning 
                    sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample.
                shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs,
                   axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        # Make sure each sample's 10 values add up to 1.
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def clustering_autoencoder_model(dims, act='relu', init='glorot_uniform', n_clusters=10):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is
            input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the
            auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    autoencoder, encoder = autoencoder_model(dims, act=act, init=init)
    autoencoder.load_weights(SAVE_DIR+'/ae_weights.h5')
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
    return Model(inputs=encoder.input,
            outputs=[clustering_layer, autoencoder.output])


def target_distribution(q):
    """
    computing an auxiliary target distribution

    """
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def make_model(input_shape):
    """
    do it
    """

    dims = [input_shape, 500, 500, 2000, 50]
    init = VarianceScaling(scale=1. / 3.,
                           mode='fan_in',
                           distribution='uniform')

    autoencoder, encoder = autoencoder_model(dims, init=init)
    return autoencoder, encoder

def pre_train(x, y, autoencoder, encoder):
    # pretrain AE
    n_clusters = len(np.unique(y))
    print(f"X: {x.shape}, clusters: {n_clusters}")
    size = x.shape[0]
    pretrain_optimizer = SGD(lr=1, momentum=0.9)
    ae_weights_file = f'{SAVE_DIR}/ae_weights_{size}.h5'
    if os.path.exists(ae_weights_file):
        print("loading ae weights")
        autoencoder.load_weights(ae_weights_file)
    else:
        print("pretraining autoencoder")
        es_callback = keras.callbacks.EarlyStopping(
            monitor='loss', patience=5, min_delta=1e-5)
        autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
        autoencoder.fit(
            x, x,
            batch_size=BATCH_SIZE,
            epochs=PRETRAIN_EPOCHS,
            callbacks=[es_callback])
        autoencoder.save_weights(ae_weights_file)

    clustering_layer = ClusteringLayer(
        n_clusters, name='clustering')(encoder.output)
    model = Model(inputs=encoder.input, outputs=clustering_layer)
    model.compile(optimizer=SGD(0.01, 0.9), loss='kld')

    # init cluster centres
    print("Initialising cluster centres")
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(encoder.predict(x))
    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

    return model, y_pred



def train_model(x: np.ndarray,
                y: np.ndarray,
                model: Model,
                y_pred: np.ndarray,
                filename: str,
                max_iter: int = None):
    """
    train
    """
    y_pred_last = np.copy(y_pred)
    size = x.shape[0]

    if max_iter is None:
        max_iter = MAXITER
    loss = 0
    index = 0
    index_array = np.arange(x.shape[0])

    print(f"Train iterations: {max_iter}")
    for ite in range(int(max_iter)):
        if ite % UPDATE_INTERVAL == 0:
            q = model.predict(x, verbose=0)
            p = target_distribution(q)  # update the auxiliary target distribution p

            # evaluate the clustering performance
            y_pred = q.argmax(1)
            if y is not None:
                accu = np.round(metrics.acc(y, y_pred), 5)
                nmi = np.round(metrics.nmi(y, y_pred), 5)
                ari = np.round(metrics.ari(y, y_pred), 5)
                loss = np.round(loss, 5)
                print(f"Iter {ite}: acc = {accu:.5f}, "
                      f"nmi = {nmi:.5f}, ari = {ari:.5f}"
                      f" ; loss={loss}")

            # check stop criterion - model convergence
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / size
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label < TOL:
                print('delta_label ', delta_label, '< tol ', TOL)
                print('Reached tolerance threshold. Stopping training.')
                break
        idx = index_array[index * BATCH_SIZE: min((index+1) * BATCH_SIZE, size)]
        loss = model.train_on_batch(x=x[idx], y=p[idx])
        index = index + 1 if (index + 1) * BATCH_SIZE <= size else 0

    save_file = os.path.join(SAVE_DIR, filename)
    model.save_weights(save_file)
    print(f"saved model to {save_file}")
    return loss

def train_joint_model(x: np.ndarray,
                y: np.ndarray,
                model: Model,
                y_pred: np.ndarray,
                filename: str,
                max_iter: int = None):
    """
    train the two-headed model
    """
    y_pred_last = np.copy(y_pred)
    size = x.shape[0]

    if max_iter is None:
        max_iter = MAXITER
    loss = 0
    index = 0
    index_array = np.arange(x.shape[0])

    print(f"Train iterations: {max_iter}")
    for ite in range(int(max_iter)):
        if ite % UPDATE_INTERVAL == 0:
            q = model.predict(x, verbose=0)
            p = target_distribution(q)  # update the auxiliary target distribution p

            # evaluate the clustering performance
            y_pred = q.argmax(1)
            if y is not None:
                accu = np.round(metrics.acc(y, y_pred), 5)
                nmi = np.round(metrics.nmi(y, y_pred), 5)
                ari = np.round(metrics.ari(y, y_pred), 5)
                loss = np.round(loss, 5)
                print(f"Iter {ite}: acc = {accu:.5f}, "
                      f"nmi = {nmi:.5f}, ari = {ari:.5f}"
                      f" ; loss={loss}")

            # check stop criterion - model convergence
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / size
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label < TOL:
                print('delta_label ', delta_label, '< tol ', TOL)
                print('Reached tolerance threshold. Stopping training.')
                break
        idx = index_array[index * BATCH_SIZE: min((index+1) * BATCH_SIZE, size)]
        loss = model.train_on_batch(x=x[idx], y=p[idx])
        index = index + 1 if (index + 1) * BATCH_SIZE <= size else 0

    save_file = os.path.join(SAVE_DIR, filename)
    model.save_weights(save_file)
    print(f"saved model to {save_file}")
    return loss
