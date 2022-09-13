from dataclasses import dataclass
import fractions
import os
import math
import glob
from typing import Optional
import warnings
from pathlib import Path
import umap
import umap.plot as plt_u
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
import matplotlib.pyplot as plt
from pandas import DataFrame, crosstab
from IPython.display import Image
from keras.utils import plot_model
from tqdm.notebook import trange
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score)
from sklearn.metrics import pairwise
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering
from sklearn import mixture
from scipy.stats import multivariate_normal
from wordcloud import WordCloud
from jinja2 import Environment, FileSystemLoader
from metrics import plot_confusion

from data import load_data
from timer import timer

warnings.filterwarnings("ignore")

ENTITY_FILTER_LIST = ['GPE', 'PERSON', 'ORG', 'DATE', 'NORP',
                      'TIME', 'PERCENT', 'LOC', 'QUANTITY', 'MONEY', 'FAC', 'CARDINAL',
                      'EVENT', 'PRODUCT', 'WORK_OF_ART', 'ORDINAL', 'LANGUAGE']


@dataclass
class Cluster:
    """
    Describes a single cluster of entities.
    """
    freqs: dict
    freqs_unknown: dict
    class_freqs: np.ndarray
    frac: float
    n: int
    label: str
    entity_id: int
    clus_no: int
    name: str

ClusterList = dict[str, Cluster]

def write_results_page(clusters, new_clusters, save_dir, test_name, scores):
    """
    Write the results page out to the folder, with index.html.
    """
    # pass strings to the template for formatting
    str_scores = scores.copy()
    for key in str_scores:
        str_scores[key] = f"{str_scores[key]:.4f}"

    environment = Environment(loader=FileSystemLoader("templates/"))

    results_filename = os.path.join(save_dir, "index.html")
    results_template = environment.get_template("index.jinja")
    context = {
        "clusters": clusters,
        "new_clusters": new_clusters,
        "test_name": test_name,
        "metrics": str_scores,
    }
    with open(results_filename, mode="w", encoding="utf-8") as results:
        results.write(results_template.render(context))
        full_filename = Path(results_filename).absolute()
        print(f'... wrote results to {full_filename}')


def freqs_descending(df, col):
    """
    Return a list of words and their frequencies, sorted by frequency.
    """
    uniques, counts = np.unique(df[col], return_counts=True)
    freq_list = np.asarray((uniques, counts)).T
    freq_list2 = np.asarray(sorted(freq_list, key=lambda x: -x[1]))
    # purity
    y_true_this_cluster = len(
        df[df[col] == freq_list2[0][0]])
    frac = y_true_this_cluster/len(df)
    return freq_list2, frac


def do_clustering(
        clustering: str,
        n_clusters: int,
        z_state: DataFrame,
        params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform clustering on the data.
        -clustering: the clustering algorithm to use
        -n_clusters: the number of clusters to use
        -z_state: the data to cluster
        -params: dict, optional
            'eps' or 'min_samples' values for DBSCAN/OPTICS
    Returns:
        - the cluster assignments
        - cluster centers
    """
    dbscan_eps = 1
    dbscan_min_samples = 5
    min_cluster_size = 15
    if 'eps' in params:
        dbscan_eps = params['eps']
    if 'min_samples' in params:
        dbscan_min_samples = params['min_samples']
    if 'min_cluster_size' in params:
        min_cluster_size = params['min_cluster_size']

    if clustering == 'GMM':
        gmix = mixture.GaussianMixture(
            n_components=n_clusters,
            covariance_type='full',
            verbose=params['verbose'])
        gmix.fit(z_state)
        y_pred = gmix.predict(z_state)
        # get centres
        centers = np.empty(shape=(gmix.n_components, z_state.shape[1]))
        for i in range(gmix.n_components):
            density = multivariate_normal(
                cov=gmix.covariances_[i],
                mean=gmix.means_[i]).logpdf(z_state)
            centers[i, :] = z_state[np.argmax(density)]
    elif clustering == 'Kmeans':
        kmeans = KMeans(n_clusters=n_clusters, n_init=10,
                        verbose=params['verbose'])
        y_pred = kmeans.fit_predict(z_state)
        centers = kmeans.cluster_centers_
    elif clustering == 'DBSCAN':
        dbscan = DBSCAN(
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
            metric='manhattan')
        y_pred = dbscan.fit_predict(z_state)
        centers = np.zeros((len(np.unique(y_pred))))
    elif clustering == 'OPTICS':
        optics = OPTICS(
            min_samples=dbscan_min_samples,
            min_cluster_size=min_cluster_size,
            metric='cosine')
        y_pred = optics.fit_predict(z_state)
        centers = np.zeros((len(np.unique(y_pred))))
    elif clustering == "agg":
        agg = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity='cosine',
            linkage='complete')
        y_pred = agg.fit_predict(z_state)
        centers = None
    else:
        raise ValueError('Clustering algorithm not specified/unknown.')

    return y_pred, centers  


def show_wordcloud(
        freqs: np.ndarray,
        name: str,
        filepath: str,
        width: int = 16,
        save_only: bool = False) -> None:
    """
    Show wordcloud for a cluster.
    """
    if len(freqs) > 0:
        wc = WordCloud(width=800, height=500).generate_from_frequencies(freqs)
        if not save_only:
            plt.figure(figsize=(width, width-1))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            plt.show()
        wc.to_file(filepath)
    else:
        print(f"No words for cluster {name}")

def make_q(z, batch_size, alpha):
    """
    Calculate the probability distribution of z
    """
    sqd_dist_mat = np.float32(
                        pairwise.pairwise_distances(z, metric='sqeuclidean'))
    q = tf.pow((1 + sqd_dist_mat/alpha), -(alpha+1)/2)
    q = tf.linalg.set_diag(q, tf.zeros(shape=[batch_size]))
    q = q / tf.reduce_sum(q, axis=0, keepdims=True)
    # q = 0.5*(q + tf.transpose(q))
    q = tf.clip_by_value(q, 1e-10, 1.0)

    return q


class DeepLatentCluster():
    """
    Deep Latent Clustering
    Dahal, P. (2018). Learning Embedding Space for Clustering From Deep
    Representations. 2018 IEEE International Conference on Big Data (Big Data),
    3747–3755.
    """

    def __init__(
        self,
        run_name: str,
        config: dict,
        verbose: int = 1,
    ):

        self.x: Optional[DataFrame] = None
        self.y: Optional[DataFrame] = None
        self.x_sample = None
        self.y_sample = None
        self.str_sample = None
        self.mapping: Optional[dict] = None
        self.strings: list[str] = []
        self.y_pred_last = None
        self.input_dim = 768
        self.batch_size = 256

        self.run_name = run_name
        self.model: Optional[Model] = None
        self.encoder: Optional[Model] = None
        self.autoencoder: Optional[Model] = None
        self.save_dir = ""
        self.verbose = verbose
        self.train_acc_metric = None

        # latent model config
        self.config = {
            "layers_ae": [
                {"n": 500, "act": None},
                {"n": 500, "act": None},
                {"n": 2000, "act": None},
            ],
            "layer_ae_latent":
                {"n": 40, "act": None},
            "layers_latent_network": [
                {"n": 2000, "act": None},
                {"n": 500, "act": None},
                {"n": 500, "act": None},
            ],
            "output_fn": 'sigmoid',
            "opt": k.optimizers.Adam(
                lr=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=None,
                decay=0.0,
                amsgrad=False),
            "latent_loss": "cross_entropy",
            "noise_factor": 0.0,
            "loss": k.losses.binary_crossentropy,
            "epochs": 1,
            "train_size": 10000,
            "num_clusters": 25,
            "cluster": "GMM",
            "radius": 0,
            "entities": None,
            "entity_count": 10,
            "loss_weights": None,
            "latent_weight": 0.0001,
            "reconstr_weight": 1.0,
            "max_iter": 8000,
            "pretrain_epochs": 1000,
            "emb_size": 768,
            "alpha1": 20,
            "alpha2": 1,
            "ae_init_fn": VarianceScaling(
                mode='fan_in',
                scale=1. / 3.,
                distribution='uniform'),
            "base_dir": "results",
        }
        if config is not None:
            if any([k for k in config.keys() if k not in self.config.keys()]):
                print(f"ERROR: Invalid config key {k}")
            self.config = {**self.config, **config}

        if self.config['entities'] is not None and self.config['entity_count'] > 0:
            raise ValueError(
                'entities and entity_count cannot both be specified')
        if self.config['entities'] is None:
            if self.config['entity_count'] == 0:
                self.config['entities'] = ENTITY_FILTER_LIST
            else:
                self.config['entities'] = ENTITY_FILTER_LIST[
                    :self.config['entity_count']]

        self.save_dir = f'./{self.config["base_dir"]}/{self.run_name}'
        if not os.path.exists(self.save_dir):
            # create save dir
            os.makedirs(self.save_dir)

    def output(self, s: str) -> None:
        """
        Print output if verbose is set.
        """
        if self.verbose > 0:
            print(s)

    def make_data(self, oversample: bool = True, train: bool = True) -> None:
        """
        Make data for training.
        - oversample: oversample the data to balance the classes
        - train: if True, make training data, otherwise make test data
        """
        self.output("Load Data")
        self.x, self.y, self.mapping, self.strings = load_data(
            self.config['train_size'],
            entity_filter=self.config['entities'],
            get_text=True,
            oversample=oversample,
            verbose=self.verbose,
            radius=self.config['radius'],
            train=train)
        self.input_dim = self.x.shape[1]
        self.output("Data Loaded")

    @staticmethod
    def create_layer(config: dict, name: str, init_fn: str = 'glorot_uniform') -> Dense:
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
            kernel_initializer=init_fn,
            kernel_regularizer='l1')

    def autoencoder_model(self,
                          layer_specs: list,
                          act: str = 'tanh',
                          init_fn: str = 'glorot_uniform'):
        """
        Creates the autoencoder given
        -layer_specs: list of layer sizes.
            Model is symmetrical so only need to specify the first half.
        -act: activation function for hidden layers
        -init_fn: initializer for weights

        returns:
            - the full autoencoder
            - the encoder only
        """
        layers = len(layer_specs) - 1
        # input
        input_img = Input(shape=(layer_specs[0],), name='input')
        encoder = [input_img]
        # hidden layers in encoder
        for i in range(layers-1):
            layer = Dense(
                layer_specs[i + 1],
                activation=act,
                kernel_initializer=init_fn,
                name=f'encoder_{i}')(encoder[-1])
            encoder += [layer]
            if self.verbose >= 2:
                print(f'encoder_{i}: {layer_specs[i+1]} '
                      f'activation={act}')

        # latent layer
        latent = Dense(
            layer_specs[-1],
            kernel_initializer=init_fn,
            name=f'encoder_{layers - 1}')(encoder[-1])
        encoder += [latent]
        if self.verbose >= 2:
            print(f'encoder_{layers - 1}: {layer_specs[-1]}')

        autoencoder = [encoder[-1]]
        # hidden layers in decoder
        for i in range(layers-1, 0, -1):
            layer = Dense(
                layer_specs[i],
                activation=act,
                kernel_initializer=init_fn,
                name=f'decoder_{i}')(autoencoder[-1])
            autoencoder += [layer]
            if self.verbose >= 2:
                print(f'encoder_{i}: {layer_specs[i]}'
                      f' activation={act}')

        # output
        layer = Dense(
            layer_specs[0],
            kernel_initializer=init_fn,
            name='decoder_0')(autoencoder[-1])
        autoencoder += [layer]
        if self.verbose >= 2:
            print(f'output: {layer_specs[0]}'
                  f'')
        return (encoder, autoencoder)

    def create_latent_space_model(self, input_layer: Dense) -> Model:
        """
        Create the model for the latent space
        - input_layer: previous model output layer
        """
        # build ml, the array of layers
        ml = [input_layer]
        layers = self.config['layers_latent_network']
        for n, l in enumerate(layers):
            ml.append(self.create_layer(l, f"latent_{n}")(ml[-1]))

        out_config = {**layers[0], "act": self.config['output_fn']}
        ml.append(self.create_layer(out_config, "latent_out")(ml[-1]))
        return ml

    def create_model(self) -> Model:
        """
        Create the entire model.

        """
        self.output("Autoencoder")
        enc, dec = self.autoencoder_model(
            [768, 500, 500, 2000, 40],
            init_fn=self.config['ae_init_fn'],
            act='relu',
        )

        self.output("Latent Model")
        latent_space = self.create_latent_space_model(
            input_layer=enc[-1])
        self.model = k.Model(
            name="full-model",
            inputs=enc[0],
            outputs=[
                dec[-1],
                latent_space[-1],
            ])

        self.autoencoder = k.Model(
            name="ae-model",
            inputs=enc[0],
            outputs=[
                dec[-1],
            ])

        self.encoder = k.Model(
            name="encoder",
            inputs=enc[0],
            outputs=[
                enc[-1],
            ])

    def make_model(self, verbose: Optional[int]=1) -> None:
        """
        Make the model and saves the image of it
        """
        if verbose is not None:
            self.verbose = verbose

        self.create_model()

        self.model.compile(# type: ignore
            loss=[
                self.latent_loss(
                    self.model.get_layer("encoder_2").get_weights()[0])],
            # loss_weights=self.config['loss_weights'],
            optimizer=SGD(learning_rate=0.5, momentum=0.9))
        self.output("model compiled")

        self.autoencoder.compile(optimizer='adam', loss='mse')

        img_file = os.path.join(self.save_dir, f'{self.run_name}_model.png')
        plot_model(self.model, to_file=img_file, show_shapes=True)
        Image(filename=img_file)

    def reconstr_loss(self, x, x_pred):
        """
        Reconstruction loss from autoencoder
        Mean squared error
        """
        return tf.reduce_mean(tf.square(tf.subtract(x, x_pred)))

    def latent_loss(self, z_enc):
        """
        Latent loss from latent space based on KL divergence of p & q
        """
        def loss(_, y_pred):
            """
            A loss function is of the form loss(y_true, y_pred), but
            we don't need y_true, so we ignore it.

            """
            p = make_q(y_pred, self.batch_size, alpha=self.config['alpha1'])
            q = make_q(z_enc, self.batch_size, alpha=self.config['alpha2'])
            if self.config['latent_loss'] == 'kl':
                return tf.reduce_sum(-(tf.multiply(p, tf.math.log(tf.divide(p,q)))))
            elif self.config['latent_loss'] == 'cross_entropy':
                return tf.reduce_sum(-(tf.multiply(p, tf.math.log(q))))
            else:
                raise ValueError(f"Unknown loss type {self.config['latent_loss']}")
        return loss

    def train_model(self, verbose: int = 1) -> None:
        """
        Run the model.
        """
        self.verbose = verbose

        if self.x is None:
            self.make_data(oversample=True)
            self.output("Data Loaded")

        if self.model is None:
            self.make_model(verbose)

        self.output("Training autoencoder")
        early_stopping_cb = EarlyStopping(
            monitor='loss', patience=10, verbose=1, min_delta=0.00001)
        history = self.autoencoder.fit(
            self.x,
            self.x,
            batch_size=self.batch_size,
            epochs=self.config["pretrain_epochs"],
            verbose=self.verbose,
            callbacks=[early_stopping_cb],
        )
        self.autoencoder.save_weights(
            os.path.join(self.save_dir, 'ae_weights.h5'))
        self.output("Trained autoencoder")

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.title('Autoencoder pretraining loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
        plt.savefig(os.path.join(self.save_dir, 'AE loss.png'))

        # # init cluster centres before train
        # self.init_cluster_centers()

        # train full model
        losses = self.train()
        # summarize history for loss
        plt.plot(losses)
        plt.title('Full model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
        plt.savefig(os.path.join(self.save_dir, 'Full model loss.png'))


        print("Training Done")

    def aux_target_distribution(self, q):
        """
        Calculate the target distribution p for the auxiliary loss term
        """
        # sum column wise
        row_sum = q.sum(axis=0)
        # q_ij ^2 / row sum
        top = q ** 2 / row_sum
        # then / column sum
        return (top.T / top.sum(axis=1)).T

    def train_step(self, x, y):
        """
        A single training step
        """
        if self.config['noise_factor'] > 0.0:
            x = x + self.config['noise_factor'] * tf.random.normal(
                shape=tf.shape(x),
                mean=0.0,
                stddev=1.0,
                dtype=tf.float32)
            # x = tf.clip_by_value(x, 0.0, 1.0)
        with tf.GradientTape() as tape:
            dec, lat = self.model(x, training=True)
            loss_l = self.latent_loss(self.encoder(x))(y, lat)
            loss_r = self.reconstr_loss(x, dec)
            loss_value = self.config['latent_weight'] * loss_l +\
                self.config['reconstr_weight'] * loss_r

        grads = tape.gradient(loss_value, self.model.trainable_weights)
        optimizer = self.config['opt']
        optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_acc_metric.update_state(y, lat)
        return loss_value

    def train(self):
        """
        Train the model - full training loop
        """
        self.train_acc_metric = k.metrics.SparseCategoricalAccuracy()

        update_interval = 140
        index_array = np.arange(self.x.shape[0])
        tol = 1e-6  # tolerance threshold to stop training
        loss = 0
        index = 0
        update_interval = 140
        losses = []
        for epoch in trange(self.config['epochs']):
            for ite in trange(int(self.config['max_iter']), leave=False):
                if ite % update_interval == 0:
                    self.output(f"Iter:{ite} -> loss:{loss}")
                idx = index_array[
                    index * self.batch_size:
                    min((index+1) * self.batch_size, self.x.shape[0])]
                loss = self.train_step(self.x[idx], self.y[idx])
                losses += [loss]
                # stop training if tolerance threshold is reached
                patience = 5
                max_delta = np.max(np.abs(losses[-patience:] - losses[-1]))
                if len(losses) > 3 and max_delta < tol:
                    print("Stop traing for tolerance threshold reached")
                    break
            if self.verbose == 0:
                # final values
                print(f'epoch: {epoch} loss={loss}')
        self.model.save_weights(os.path.join(
            self.save_dir, 'lat_model_final.h5'))
        return losses

    def make_load_model(self, load_dir):
        """
        Load a model from a directory
        """
        self.make_model(self.verbose)

        if load_dir is None:
            load_dir = self.save_dir
        else:
            load_dir = f'./{self.config["base_dir"]}/{load_dir}'

        ae_weights_file = os.path.join(load_dir, 'ae_weights.h5')
        self.output(f"Loading AE weights from {ae_weights_file}")
        self.autoencoder.load_weights(ae_weights_file)

        model_weights_file = os.path.join(load_dir, 'lat_model_final.h5')
        self.output(f"Loading model weights from {model_weights_file}")
        self.model.load_weights(model_weights_file)

    @staticmethod
    def get_freqs(word_list):
        """
        Get the frequencies of words in a list
        """
        unique, counts = np.unique(word_list, return_counts=True)
        freq_list = np.asarray((unique, counts)).T
        freq_list = sorted(freq_list, key=lambda x: -x[1])[0:50]
        freqs = {w: f for w, f in freq_list}
        return freqs

    def get_sample(self, sample_size: int) -> None:
        """
        Get a sample of the data
        """
        if self.config['train_size'] != sample_size or self.x_sample is None:
            self.output("Load Data ")
            self.x, self.y, self.mapping, self.strings = load_data(
                0,
                get_text=True,
                verbose=self.verbose,
                train=False)
            self.output(f"Test Data Loaded {self.x.shape}")

        # sample
        if sample_size > 0 and self.x.shape[0] > sample_size:
            sample_idx = np.random.choice(
                self.x.shape[0], sample_size, replace=False)
            y_sample = self.y[sample_idx]
            str_sample = self.strings[sample_idx]
            x_sample = self.x[sample_idx]
        else:
            x_sample = self.x
            y_sample = self.y
            str_sample = self.strings

        self.x_sample = x_sample
        self.y_sample = y_sample
        self.str_sample = str_sample

    def predict(self, sample_size: int, head: str):
        """
        Make predictions for the given sample size.
        Sample will be from Test dataset not Train data
        returns
            - latent space predictions
            - actual labels
            - the textual repr of each item
        """
        if head is None:
            head = 'z'
        self.output(f"Using [{head}] head for predictions")

        self.get_sample(sample_size)

        # predict z space
        assert self.x_sample is not None
        num_batches = math.ceil((1.0 * len(self.x_sample)) / self.batch_size)
        self.output(f"Predicting...{num_batches} batches of "
                    f"{self.batch_size} x {self.x_sample.shape[1]}")

        # run the model on sampled x in batches
        z_sample = []
        for i in trange(num_batches, disable=self.verbose == 0):
            idx = np.arange(
                i * self.batch_size,
                min(len(self.x_sample), (i + 1) * self.batch_size))
            x = self.x_sample[idx]
            if head == 'z':
                _, z_batch = self.model.predict(x, verbose=0)
            elif head == 'ae':
                z_batch = self.autoencoder.predict(x, verbose=0)
            elif head == 'enc':
                z_batch = self.encoder.predict(x, verbose=0)

            z_sample += [z_batch] # type: ignore

        z_space = np.vstack(np.array(z_sample)) # type: ignore

        return z_space

    def visualise_umap(
                    self,
                    z_sample,
                    y_sample,
                    to_dir: bool=True,
                    name: Optional[str] = None) -> np.ndarray:
        """
        Visualise the latent space using UMAP
        """
        self.output("Visualising")
        if name is None:
            name = "UMAP"
        else:
            print(f"Saving to {name}")
        y_label = np.asarray(
            [(self.mapping[l] if l in self.mapping else l) for l in y_sample])
        mapper = umap.UMAP(metric='manhattan',
                           verbose=self.verbose > 0).fit(z_sample)
        plt_u.points(mapper, labels=y_label, height=1200, width=1200)

        if to_dir:
            save_file = os.path.join(self.save_dir, f"{name}.png")
        else:
            save_file = f"{name} Benchmark.png"
        plt_u.plt.savefig(save_file)
        if self.verbose > 0:
            plt_u.plt.show()
        plt_u.plt.close()
        return y_label

    def apply_cluster_algo(self, z_sample):
        """
        Apply the clustering algorithm to the latent space
        """
        self.output(f"Clustering {z_sample.shape[0]} points "
                    f"using {self.config['cluster']}")
        y_pred_sample, c = do_clustering(
            clustering=self.config['cluster'],
            n_clusters=self.config['num_clusters'],
            z_state=z_sample,
            params={'verbose': self.verbose})
        if self.config['cluster'] in ['DBSCAN', 'OPTICS']:
            self.config['num_clusters'] = len(c)

        y_label = self.visualise_umap(z_sample, self.y_sample)

        return y_pred_sample, y_label

    def calc_metrics(self, TP, FP, FN):
        """
        Calculate f1, precision and recall
        """
        if TP + FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)
        if TP + FN == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return f1, precision, recall

    def rearrange_clusters(
                        self,
                        sample,
                        y_pred_sample,
                        do_rearrange:bool=True
                        ) -> tuple[DataFrame, ClusterList]:
        """
        Rearrange the clusters so that the most common label is in the
        eponymoous cluster
        """
        # placeholder for revised predictions
        sample['y_pred_new'] = 0

        # make a n x m array
        y_tru_per_clus = crosstab(
            index=sample['y_true'], columns=sample['y_pred'])
        y_tru_counts = y_tru_per_clus.sum()
        y_tru_frac_by_clus = y_tru_per_clus / y_tru_counts

        clusters: ClusterList = {}

        for clus_no in np.unique(y_pred_sample):
            if clus_no < 0:
                continue
            cluster = sample[sample['y_pred'] == clus_no]
            prob_ent = int(np.argmax(y_tru_per_clus[clus_no]))
            prob_lbl = self.mapping[prob_ent]
            frac = y_tru_frac_by_clus[clus_no][prob_ent]

            # wordcloud
            freqs = self.get_freqs(cluster['text'].values)
            unknown_cluster = cluster[cluster['y_true'] == 0]
            freqs_unknown = self.get_freqs(unknown_cluster['text'].values)
            class_freqs, _ = freqs_descending(cluster, 'y_true')
            entry = Cluster(
                freqs=freqs,
                freqs_unknown=freqs_unknown,
                class_freqs=class_freqs,
                frac = frac,
                n = len(cluster),
                label = prob_lbl,
                entity_id = prob_ent,
                clus_no = clus_no,
                name = prob_lbl)

            if do_rearrange:
                # filling in the dict {name: entry}
                # where the best PERSON entry is eponymous and less likely entries
                # are named "UNK-PERSON-X" for cluster X
                cluster_name = prob_lbl
                unk_cluster_name = f"UNK-{prob_lbl}-{clus_no}"

                if prob_lbl == 'UNKNOWN':
                    cluster_name = unk_cluster_name
                elif prob_lbl in clusters:
                    if frac > clusters[prob_lbl].frac:
                        # we found a better cluster for this label
                        clusters[unk_cluster_name] = clusters[prob_lbl]
                    else:
                        # this cluster is worse than this one, so it's unknown
                        cluster_name = unk_cluster_name
            else:
                cluster_name = f"c-{prob_lbl}-{clus_no}"

            clusters[cluster_name] = entry

            # write the cluster label back into the sample
            sample.loc[
                (sample['y_pred'] == clus_no) &
                (sample['y_true'] == prob_ent),
                'y_pred_new'] = prob_ent

        return sample, clusters

    def eval_cluster(self,
                    y_pred_sample,
                    y_label,
                    rearrange: bool=True):
        """
        show wordclouds for each cluster
        """
        self.output("CLUSTERS")

        raw_sample = DataFrame({
            'text': self.str_sample,
            'y_true': self.y_sample,
            'y_pred': y_pred_sample,
            'y_label': y_label,
        })

        sample, clusters = self.rearrange_clusters(
                                        raw_sample, y_pred_sample, rearrange)

        # confusion
        f1_list = []
        size_list = []
        cluster_scores = {}
        for cluster_name, ce in clusters.items():
            c = sample[sample.y_pred == ce.clus_no]
            c_ent = ce.entity_id

            # the right entity class in the right cluster
            TP = c[
                # in this cluster
                (c.y_pred_new == c_ent) &\
                # and this is the right class
                (c.y_true == c_ent)].shape[0]

            # this cluster, we think it's right entity but not the right entity
            FP = c[
                # in this cluster
                (c.y_pred_new == c_ent) &\
                # but not the right entity class
                (c.y_true != c_ent)].shape[0]

            # it's the right entity in wrong cluster
            FN = sample[
                # not in this cluster
                (sample.y_pred_new != c_ent) &\
                # but should be
                (sample.y_true == c_ent)].shape[0]

            f1, prec, rec = self.calc_metrics(TP, FP, FN)

            cluster_scores[cluster_name] = {
                'F1': f1,
                'precision': prec,
                'recall': rec,
                'TP': TP,
                'FP': FP,
                'FN': FN,
            }

            if cluster_name[0:3] == 'UNK':
                f1_list.append(f1)
                size_list.append(ce.n/sample.shape[0])

            self.output(f"#{cluster_name}:{ce.clus_no} size:{len(c)} "
                        f"prec:{prec:.4f} rec:{rec:.4f} f1:{f1:.4f}")

        # full cluster
        f1 = np.dot(f1_list, size_list)
        print(f"\nF1 by Known Clusters: {f1:.4f}")

        return sample, clusters, f1

    def show_core_metrics(self, y_pred_sample, all_clusters, new_labels):
        """
        show the core metrics for the clustering
        """
        assert self.mapping is not None
        assert self.y_sample is not None

        # confusion matrix
        num_predicted = len(np.unique(y_pred_sample))
        cm_width = max(8, num_predicted * 2)
        cm_width = min(16, cm_width)
        plot_confusion(self.y_sample,
                       y_pred_sample,
                       self.mapping,
                       self.save_dir,
                       size=cm_width,
                       new_labels=new_labels)

        # how big are the predicted clusters
        cluster_counts = np.unique(y_pred_sample, return_counts=True)[1]
        # how big are the new ones?
        
        num_new = cluster_counts[len(self.mapping):]
        # how many of the new ones are more than 2.5% of the sample?
        num_new_large = len(num_new[num_new > (len(self.y_sample) * 0.025)])

        # metrics
        y = all_clusters['y_true']
        y_pred = all_clusters['y_pred_new']
        f1 = f1_score(y, y_pred, average='macro')
        acc = accuracy_score(y, y_pred)
        precision = precision_score(
            y, y_pred, average='macro')
        recall = recall_score(y, y_pred, average='macro')
        print(f"F1 score (macro) = {f1:.4f}")
        print(f"Accuracy = {acc:.4f}")
        print(f"Precision = {precision:.4f}")
        print(f"Recall = {recall:.4f}")
        print(f"New clusters = {num_new_large:6d}")
        scores = {
            'f1': f1,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'new_clusters': num_new_large,
        }
        return scores

    def score_clusters(self, clusters):
        """
        score the clusters found
        """
        cluster_list = sorted(clusters.values(), key=lambda x: -x.frac)

        # show unknown clusters first
        cluster_list = sorted(
            cluster_list,
            key=lambda x: int(x.name[0:3] != "UNK"))

        # delete old wordcloud files
        for f in glob.glob(f"{self.save_dir}/wordcloud*.png"):
            os.remove(f)

        for cluster in cluster_list:
            save_file = os.path.join(self.save_dir,
                                     f"wordcloud-{cluster.name}.png")
            show_wordcloud(
                cluster.freqs,
                cluster.name,
                save_file,
                save_only=True)

            # the top 3 entity classes in this cluster
            top_entities = []
            for (entity, count) in cluster.class_freqs[0:3]:
                top_entities += [
                    {'class': self.mapping[entity],
                     'count':count}]
            cluster.classes = top_entities

        # save clusters of NER unknowns only
        for cluster in cluster_list:
            save_file = os.path.join(self.save_dir,
                                     f"wordcloud-{cluster.name}-new.png")
            if len(cluster.freqs_unknown) > 0:
                show_wordcloud(
                    cluster.freqs_unknown,
                    cluster.name,
                    save_file,
                    save_only=True)

        return cluster_list

    def save_scores(self, cluster_list, scores):
        """
        save the scores to file
        """
        new_clusters = [c for c in cluster_list if len(c.freqs_unknown) > 4]
        big_clusters = [c for c in cluster_list if len(c.freqs) > 5]
        write_results_page(
            big_clusters,
            new_clusters,
            self.save_dir,
            self.run_name,
            scores,
        )

    @timer
    def evaluate_model(self,
                       load_dir: str,
                       sample_size: int,
                       head: str = "enc",
                       verbose: int = 1) -> dict:
        """
        Run the model.
        """
        self.verbose = verbose

        self.make_load_model(load_dir)

        # predict the requested sample size
        # z is the latent space
        z_sample = self.predict(sample_size, head)

        return self.do_evaluation(z_sample)

    def do_evaluation(self, z_sample):
        """
        Evaluate the model.
        - z_sample: the latent space
        - y_sample: the true labels
        - str_sample: the original strings
        """
        # cluster the latent space using requested algorithm
        y_pred_sample, y_label = self.apply_cluster_algo(z_sample)

        # assign labels to the clusters
        all_clusters, clusters, cluster_f1 = self.eval_cluster(
                                                    y_pred_sample, y_label)

        # list of new clusters by id number
        labels_no = {v.clus_no: k for k,v in clusters.items()}
        labels_ent = {v.entity_id: k for k,v in clusters.items()}
        new_labels = {**labels_no, **labels_ent}

        # overall scores
        scores_agg = self.show_core_metrics(
            y_pred_sample, all_clusters, new_labels)

        # cluster scores
        cluster_list = self.score_clusters(clusters)

        # output file
        self.save_scores(cluster_list, scores_agg)

        scores = {**scores_agg, 'cluster F1': cluster_f1} # type: ignore

        return scores

    def benchmark_model(self, sample_size: int = 0, verbose: int = 1) -> None:
        """
        Run the clustering on sample data without runnin the model.
        """
        self.verbose = verbose

        # predict the requested sample size
        self.get_sample(sample_size)

        self.do_evaluation(self.x_sample)

    def random_benchmark(self, sample_size: int = 0, verbose: int = 1) -> dict:
        """
        Benchmark a sample with cluster allocation from cluster algorithm only
        """
        self.verbose = verbose

        # predict the requested sample size
        self.get_sample(sample_size)

        # cluster the latent space using requested algorithm
        y_pred_sample, y_label = self.apply_cluster_algo(self.x_sample)

        # assign labels to the clusters
        all_clusters, clusters, cluster_f1 = self.eval_cluster(
            y_pred_sample, y_label, rearrange=False)

        # list of new clusters by id number
        new_labels = {v.clus_no: k for k,v in clusters.items()}

        # overall scores
        scores_agg = self.show_core_metrics(
            y_pred_sample, all_clusters, new_labels=new_labels)

        # cluster scores
        cluster_list = self.score_clusters(clusters)

        # output file
        self.save_scores(cluster_list, scores_agg)

        scores = {**scores_agg, 'cluster F1': cluster_f1}

        return scores


def train_and_evaluate_model(self, eval_size, verbose=1):
    """
    Make and evaluate a model.
    Arguments:
        run_name: name of the run.
        data_rows: number of rows to use.
        n_clusters: number of clusters to use.
        entity_count: number of entities to use.
    """
    self.verbose = verbose
    self.make_model()
    self.train_model()
    self.evaluate_model(eval_size)
