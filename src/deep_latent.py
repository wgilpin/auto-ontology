import os
import math
from typing import Optional
import warnings
import umap
import umap.plot as plt_u
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
import matplotlib.pyplot as plt
from pandas import DataFrame
from IPython.display import Image
from keras.utils import plot_model
from tqdm.notebook import trange
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import pairwise


from data import load_data
from timer import timer
from cluster_metrics import do_evaluation, eval_cluster, show_core_metrics,\
    score_clusters, save_scores, do_clustering

warnings.filterwarnings("ignore")

ENTITY_FILTER_LIST = ['GPE', 'PERSON', 'ORG', 'DATE', 'NORP',
                      'TIME', 'PERCENT', 'LOC', 'QUANTITY', 'MONEY', 'FAC', 'CARDINAL',
                      'EVENT', 'PRODUCT', 'WORK_OF_ART', 'ORDINAL', 'LANGUAGE']


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

        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.x_sample = None
        self.y_sample = None
        self.str_sample = None
        self.shorts_sample = None
        self.mapping: Optional[dict] = None
        self.strings: list[str] = []
        self.shorts: list[str] = []
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
            "tolerance": 1e-6,
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
        tol = self.config['tolerance']  # tolerance threshold to stop training
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

    def get_sample(self, sample_size: int) -> None:
        """
        Get a sample of the data
        """
        if self.config['train_size'] != sample_size or self.x_sample is None:
            self.output("Load Data ")
            self.x, self.y, self.mapping, self.strings, self.shorts = load_data(
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
            shorts_sample = self.shorts[sample_idx]
            x_sample = self.x[sample_idx]
        else:
            x_sample = self.x
            y_sample = self.y
            str_sample = self.strings
            shorts_sample = self.shorts

        self.x_sample = x_sample
        self.y_sample = y_sample
        self.str_sample = str_sample
        self.shorts_sample = shorts_sample

    def predict(self, head: str):
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

        self.get_sample(sample_size)

        # predict the requested sample size
        # z is the latent space
        z_sample = self.predict(head)

        # cluster the latent space using requested algorithm
        y_pred_sample, _ = self.apply_cluster_algo(z_sample)

        raw_sample = DataFrame({
            'text': self.str_sample,
            'y_true': self.y_sample,
            'y_pred': y_pred_sample,
            'shorts': self.shorts_sample
        })

        assert self.mapping is not None
        return do_evaluation(
            raw_sample, self.mapping, self.verbose, self.save_dir, self.run_name)


    def benchmark_model(self, sample_size: int = 0, verbose: int = 1) -> None:
        """
        Run the clustering on sample data without runnin the model.
        """
        self.verbose = verbose

        # predict the requested sample size
        self.get_sample(sample_size)

        # cluster the latent space using requested algorithm
        y_pred_sample, _ = self.apply_cluster_algo(self.x_sample)

        raw_sample = DataFrame({
            'text': self.str_sample,
            'y_true': self.y_sample,
            'y_pred': y_pred_sample,
            'shorts': self.shorts_sample,
        })

        assert self.mapping is not None
        do_evaluation(raw_sample, self.mapping, self.verbose, self.save_dir, self.run_name)

    def random_benchmark(self, sample_size: int = 0, verbose: int = 1) -> dict:
        """
        Benchmark a sample with cluster allocation from cluster algorithm only
        """
        self.verbose = verbose

        # predict the requested sample size
        self.get_sample(sample_size)

        # cluster the latent space using requested algorithm
        y_pred_sample, _ = self.apply_cluster_algo(self.x_sample)

        sample = DataFrame({
            'text': self.str_sample,
            'y_true': self.y_sample,
            'y_pred': y_pred_sample,
            'shorts': self.shorts_sample,
        })

        # assign labels to the clusters
        assert self.mapping is not None
        all_clusters, clusters, cluster_f1 = eval_cluster(
            sample, self.mapping, rearrange=False)

        # list of new clusters by id number
        new_labels = {v.clus_no: k for k,v in clusters.items()}

        # overall scores
        scores_agg = show_core_metrics(
                                y_pred_sample,
                                all_clusters,
                                new_labels,
                                self.mapping,
                                self.y_sample,
                                self.save_dir)

        # cluster scores
        cluster_list = score_clusters(clusters, self.save_dir, self.mapping)

        # output file
        save_scores(cluster_list, scores_agg, self.save_dir, self.run_name)

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
