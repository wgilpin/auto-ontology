# %%

from metrics import plot_confusion
from IPython.display import Image
from keras.utils import plot_model
import numpy as np
import tensorflow.keras.backend as k
from tensorflow.keras.layers import Dense, Input, Layer, InputSpec
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
import metrics
from data import load_data


# %%
size = 10000
max_iter = 140
# entities = ['ORG', 'GPE', 'PERSON', ]  # , 'NORP',  ]
x, y, mapping, strings = load_data(size, get_text=True)
n_clusters = len(np.unique(y))+10
print(x.shape)


# %%

def autoencoder_model(layer_specs: list, act: str='relu', init_fn: str='glorot_uniform'):
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
    x = input_img

    # hidden layers in encoder
    for i in range(layers-1):
        x = Dense(
            layer_specs[i + 1],
            activation=act,
            kernel_initializer=init_fn,
            name=f'encoder_{i}')(x)

    # latent layer
    encoder = Dense(
        layer_specs[-1],
        kernel_initializer=init_fn,
        name=f'encoder_{layers - 1}')(x)

    x = encoder
    # hidden layers in decoder
    for i in range(layers-1, 0, -1):
        x = Dense(
            layer_specs[i],
            activation=act,
            kernel_initializer=init_fn,
            name=f'decoder_{i}')(x)

    # output
    x = Dense(layer_specs[0], kernel_initializer=init_fn, name='decoder_0')(x)
    decoder = x
    return (Model(inputs=input_img, outputs=decoder, name='AE'),
            Model(inputs=input_img, outputs=encoder, name='encoder'))


# %%
kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred_kmeans = kmeans.fit_predict(x)


# %%
metrics.acc(y, y_pred_kmeans)


# %%
dims = [x.shape[-1], 500, 500, 2000, 50]
init = VarianceScaling(
                    mode='fan_in',
                    scale=1. / 3.,
                    distribution='uniform')
pretrain_optimizer = SGD(lr=1, momentum=0.9)
pretrain_epochs = 300
batch_size = 256
save_dir = './results'


# %%


class ClusteringLayer(Layer):
    """
    Clustering layer predicts the cluster assignments for each sample in the batch.
    Calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(num_clusters=10))
    ```
    # Arguments
        num_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` which
            represents the initial cluster centers from pretraining.
        alpha: degrees of freedom parameter in Student's t-distribution.
            Default to 1.0.
    """

    def __init__(self, num_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=k.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(
            shape=(self.num_clusters, input_dim),
            initializer='glorot_uniform',
            name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. 
                shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (k.sum(k.square(k.expand_dims(inputs,
                   axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        # Make sure each sample's 10 values add up to 1.
        q = k.transpose(k.transpose(q) / k.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.num_clusters

    def get_config(self):
        config = {'n_clusters': self.num_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




# %%
autoencoder, encoder = autoencoder_model(dims, init=init)



# %%
autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
autoencoder.fit(x, x, batch_size=batch_size,
                epochs=pretrain_epochs)  # , callbacks=cb)
autoencoder.save_weights(save_dir + '/jae_weights.h5')


# %%
autoencoder.load_weights(save_dir+'/jae_weights.h5')
clustering_layer = ClusteringLayer(
    n_clusters, name='clustering')(encoder.output)
model = Model(inputs=encoder.input,
              outputs=[clustering_layer, autoencoder.output])


# %%
plot_model(model, to_file='model.png', show_shapes=True)
Image(filename='model.png')


# %%
kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict(encoder.predict(x))
model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
y_pred_last = np.copy(y_pred)


# %%
model.compile(loss=['kld', 'mse'], loss_weights=[
              0.1, 1], optimizer=pretrain_optimizer)


# %%
loss = 0
index = 0
maxiter = 8000
update_interval = 140
index_array = np.arange(x.shape[0])
tol = 0.001  # tolerance threshold to stop training


# %%


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T



# %%
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q, _ = model.predict(x, verbose=0)
        # update the auxiliary target distribution p
        p = target_distribution(q)

        # evaluate the clustering performance
        y_pred = q.argmax(1)
        if y is not None:
            acc = np.round(metrics.acc(y, y_pred), 5)
            nmi = np.round(metrics.nmi(y, y_pred), 5)
            ari = np.round(metrics.ari(y, y_pred), 5)
            loss = np.round(loss, 5)
            print(f'Acc = {acc:.5f}, nmi = {nmi:.5f}, ari = {ari:.5f}'
                  f' ; loss={loss}')

        # check stop criterion
        delta_label = np.sum(y_pred != y_pred_last).astype(
            np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    idx = index_array[index *
                      batch_size: min((index+1) * batch_size, x.shape[0])]
    loss = model.train_on_batch(x=x[idx], y=[p[idx], x[idx]])
    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

model.save_weights(save_dir + '/b_DEC_model_final.h5')


# %%

model.load_weights(save_dir + '/b_DEC_model_final.h5')


# %%
# Eval.
q, _ = model.predict(x, verbose=0)
p = target_distribution(q)  # update the auxiliary target distribution p

# evaluate the clustering performance
y_pred = q.argmax(1)
if y is not None:
    acc = np.round(metrics.acc(y, y_pred), 5)
    nmi = np.round(metrics.nmi(y, y_pred), 5)
    ari = np.round(metrics.ari(y, y_pred), 5)
    loss = np.round(loss, 5)
    print(f'Acc = {acc:.5f}, nmi = {nmi:.5f}, ari = {ari:.5f}'
          f' ; loss={loss}')


# %% [markdown]
# 

# %%
len(np.unique(y_pred))


# %%
print(mapping)


# %%
plot_confusion(y, y_pred, mapping, 8)


# %%
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def show_wordcloud(i: int, cluster: dict)-> None:
    """
    Show wordcloud for a cluster.
    """
    freqs = cluster['freqs']
    frac = cluster['frac']
    n = cluster['n']
    name = cluster['name']
    print(f'{i}: "{name}", {n} items, ({frac*100:.2f}% confidence)')
    if len(freqs) > 0:
        wc = WordCloud().generate_from_frequencies(freqs)
        plt.figure(figsize=(16, 14))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.show()
    else:
        print(f"No words for cluster {cluster}")


# %%
import numpy as np
import pandas as pd

print ("CLUSTERS")
clusters = {}
predicted = pd.DataFrame({'text':strings, 'y_pred':y_pred, 'y_true':y})
for cluster_no in range(n_clusters):
    y_pred_for_key = predicted[predicted['y_pred']==cluster_no]
    true_label = 'UNKNOWN'
    modal_value = y_pred_for_key['y_true'].mode()
    if len(modal_value)>0:
        if modal_value[0] in mapping:
            true_label = mapping[modal_value[0]]
        # confidence - fraction of this cluster that is actually this cluster
        y_true_this_cluster = len(y_pred_for_key[y_pred_for_key['y_true']==modal_value[0]])
        frac = y_true_this_cluster/len(y_pred_for_key)
    else:
        frac = 0

    # wordcloud
    unique, counts = np.unique(y_pred_for_key['text'], return_counts=True)
    freq_list = np.asarray((unique, counts)).T
    freq_list =  sorted(freq_list, key=lambda x: -x[1])[0:50]
    freqs = {w: f for w,f in freq_list}
    entry = {'freqs':freqs, 'frac':frac, 'n':len(y_pred_for_key)}
    if true_label == 'UNKNOWN':
        clusters[f"UNK-{cluster_no}"] = entry
    elif true_label in clusters:
        if clusters[true_label]['frac'] < frac:
            # we found a better cluster for this label
            clusters[true_label] = entry
        else:
            # this cluster is worse than the one we already have, so it's unknown
            clusters[f"UNK-{cluster_no} Was {true_label}"] = entry
    else:
        clusters[true_label] = entry

cluster_list = [{
    **clusters[c],
    'name': c,
    'idx': idx} for idx, c in enumerate(clusters)]
cluster_list = sorted(cluster_list, key=lambda x: -x['frac'])

# show unknown clusters first
for i, cluster in enumerate(cluster_list):
    if cluster['name'][0:3] == "UNK":
        show_wordcloud(i, cluster)

# next show known clusters
for i, cluster in enumerate(cluster_list):
    if cluster['name'][0:3] != "UNK":
        show_wordcloud(i, cluster)

# %% [markdown]
# # Tuning

# %%

import os
import numpy as np
import tensorflow.keras.backend as k
import matplotlib.pyplot as plt
import metrics
from pandas import DataFrame
from metrics import plot_confusion
from IPython.display import Image
from tensorflow.keras import models
from keras.utils import plot_model
from tqdm import tqdm
from tensorflow.keras.layers import Dense, Input, Layer, InputSpec
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.cluster import KMeans
from data import load_data
from wordcloud import WordCloud
import seaborn as sns


# %%
def autoencoder_model(layer_specs: list, act: str='tanh', init_fn: str='glorot_uniform'):
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
    x = input_img

    # hidden layers in encoder
    for i in range(layers-1):
        x = Dense(
            layer_specs[i + 1],
            activation=act,
            kernel_initializer=init_fn,
            name=f'encoder_{i}')(x)

    # latent layer
    encoder = Dense(
        layer_specs[-1],
        kernel_initializer=init_fn,
        name=f'encoder_{layers - 1}')(x)

    x = encoder
    # hidden layers in decoder
    for i in range(layers-1, 0, -1):
        x = Dense(
            layer_specs[i],
            activation=act,
            kernel_initializer=init_fn,
            name=f'decoder_{i}')(x)

    # output
    x = Dense(layer_specs[0], kernel_initializer=init_fn, name='decoder_0')(x)
    decoder = x
    return (Model(inputs=input_img, outputs=decoder, name='AE'),
            Model(inputs=input_img, outputs=encoder, name='encoder'))

# %%
class ClusteringLayer(Layer):
    """
    Clustering layer predicts the cluster assignments for each sample in the batch.
    Calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(num_clusters=10))
    ```
    # Arguments
        num_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` which
            represents the initial cluster centers from pretraining.
        alpha: degrees of freedom parameter in Student's t-distribution.
            Default to 1.0.
    """

    def __init__(self, num_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=k.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(
            shape=(self.num_clusters, input_dim),
            initializer='glorot_uniform',
            name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. 
                shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (k.sum(k.square(k.expand_dims(inputs,
                   axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        # Make sure each sample's 10 values add up to 1.
        q = k.transpose(k.transpose(q) / k.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.num_clusters

    def get_config(self):
        config = {'n_clusters': self.num_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# %%
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


# %%
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering
from sklearn import mixture
from scipy.stats import multivariate_normal
from sklearn.manifold import TSNE


def do_clustering(clustering: str, n_clusters: int, z_state: DataFrame, params={}):
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
    
    if 'eps' in params:
        dbscan_eps = params['eps']
    if 'min_samples' in params:
        dbscan_min_samples = params['min_samples']

    if clustering == 'GMM':
        gmix = mixture.GaussianMixture(
            n_components=n_clusters, covariance_type='full')
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
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        y_pred = kmeans.fit_predict(z_state)
        centers = kmeans.cluster_centers_
    elif clustering == 'DBSCAN':
        dbscan = DBSCAN(
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
            metric='manhattan')
        y_pred = dbscan.fit_predict(z_state)
        centers = dbscan.components_
    elif clustering == 'OPTICS':
        optics = OPTICS(min_samples=dbscan_min_samples)
        y_pred = optics.fit_predict(z_state)
        centers = optics.components_
    elif clustering=="agg":
        agg = AgglomerativeClustering(n_clusters=n_clusters, affinity='manhattan', linkage='average')
        y_pred = agg.fit_predict(z_state)
        centers = None
    else:
        raise ValueError('Clustering algorithm not specified/unknown.')

    return y_pred, centers

# %%
from linear_assignment import linear_assignment

def cluster_acc(y_true, y_pred, y_pred_cluster):
    y_true = y_true.astype(np.int64)
    assert y_pred_cluster.size == y_true.size
    D = max(y_pred_cluster.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred_cluster.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    c_loss = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred_cluster.size
    print(f"Cluster Loss {c_loss} on {y_pred_cluster.size} clusters")
    return c_loss

# %%
from linear_assignment import linear_assignment

def cluster_loss(clustering:str, n_clusters: int):
    def loss(y_true, y_pred):
        y_pred_cluster, _ = do_clustering(clustering, n_clusters, y)
        return cluster_acc(y_true, y_pred, y_pred_cluster)
    return loss

# %%
# write_messages.py

from jinja2 import Environment, FileSystemLoader
from pathlib import Path

def write_results_page(clusters, save_dir, test_name):
    
    environment = Environment(loader=FileSystemLoader("templates/"))
    template = environment.get_template("index.jinja")

    results_filename = os.path.join(save_dir, "index.html")
    results_template = environment.get_template("index.jinja")
    context = {
        "clusters": clusters,
        "test_name": test_name,
    }
    with open(results_filename, mode="w", encoding="utf-8") as results:
        results.write(results_template.render(context))
        full_filename = Path(results_filename).absolute()
        print (f'... wrote results  <a href="{full_filename}">{full_filename}</a>')

# %%
def show_wordcloud(i: int, cluster: dict, filepath: str, width: int=16, save_only: bool=False)-> None:
    """
    Show wordcloud for a cluster.
    """
    freqs = cluster['freqs']
    frac = cluster['frac']
    n = cluster['n']
    name = cluster['name']
    print(f'{i}: "{name}", {n} items, ({frac*100:.2f}% confidence)')
    if len(freqs) > 0:
        wc = WordCloud(width=800, height=500).generate_from_frequencies(freqs)
        if not save_only:
            plt.figure(figsize=(width, width-1))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            plt.show()
        wc.to_file(filepath)
    else:
        print(f"No words for cluster {cluster}")

# %%
import warnings 
warnings.filterwarnings("ignore")

ENTITY_FILTER_LIST = ['GPE', 'PERSON', 'ORG', 'DATE', 'NORP',
    'TIME', 'PERCENT', 'LOC', 'QUANTITY', 'MONEY', 'FAC', 'CARDINAL',
    'EVENT', 'PRODUCT', 'WORK_OF_ART', 'ORDINAL', 'LANGUAGE']

class DeepCluster():

    def __init__(
                self,
                run_name: str,
                train_size: int,
                num_clusters: int,
                cluster: str="GMM",
                entities: list[str]=None,
                entity_count: int=0,
                dims: list[int] = None,
                loss_weights: list[float] = None,
                maxiter:int=8000,
                ):

        self.cluster = cluster
        self.num_clusters = num_clusters

        if entities is not None and entity_count > 0:
            raise ValueError('entities and entity_count cannot both be specified')
        if entities is None:
            if entity_count==0:
                self.entities = ENTITY_FILTER_LIST
            else:
                self.entities = ENTITY_FILTER_LIST[:entity_count]
        else:
            self.entities = entities
        
        self.x = None
        self.y = None
        self.mapping = None
        self.strings = None
        self.y_pred_last = None
        self.input_dim = 768
        self.batch_size = 256
        
        self.dims = [768, 500, 500, 2000, 100] if dims is None else dims
        self.loss_weights = loss_weights
        self.run_name = run_name
        self.train_size = train_size
        self.maxiter = maxiter
        self.model = None
        self.encoder = None
        self.autoencoder = None
        self.save_dir = None
        self.verbose = 1

    def output(self, s:str)->None:
        if self.verbose > 0:
            print(s)

    def make_data(self, oversample: bool=True) -> None:
        
        self.output("Load Data")
        self.x, self.y, self.mapping, self.strings = load_data(
                                    self.train_size,
                                    entity_filter=self.entities,
                                    get_text=True,
                                    oversample=oversample,
                                    verbose=self.verbose)
        self.input_dim = self.x.shape[1]
        self.output("Data Loaded")   

    
    def init_cluster_centers(self) -> None:
        """
        Initialize cluster centers by randomly sampling from the data.
        """
        self.output("cluster init")
        if self.x.shape[0] > 10000:
            x_sample = self.x[np.random.choice(self.x.shape[0], 100, replace=False)]
        else:
            x_sample = self.x
        y_pred, centers = do_clustering(
            'GMM' if self.cluster=='GMM' else 'Kmeans',
            self.num_clusters,
            self.encoder.predict(x_sample))
        del x_sample
        self.model.get_layer(name='clustering').set_weights([centers])
        self.y_pred_last = np.copy(y_pred)
        self.output("cluster init done")

    def make_model(self) -> None:
        
        init = VarianceScaling(
                            mode='fan_in',
                            scale=1. / 3.,
                            distribution='uniform')
        pretrain_optimizer = 'adam'# SGD(learning_rate=1, momentum=0.9)
        
        self.autoencoder, self.encoder = autoencoder_model(self.dims, init_fn=init)
        self.autoencoder.compile(
            optimizer=pretrain_optimizer,
            loss=['mse'])

        
        clustering_layer = ClusteringLayer(
                            self.num_clusters,
                            alpha=0.9,
                            name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input,
                    outputs=[clustering_layer, self.autoencoder.output])
        self.model.compile(
            loss=['kld', 'mse', cluster_loss(self.cluster, self.num_clusters)],
            loss_weights= [0.3, 1.0, 0.4] if 
                self.loss_weights is None else self.loss_weights,
            optimizer=SGD(learning_rate=0.5, momentum=0.9))
        self.output("model compiled")
        
        self.save_dir = f'./results/{self.run_name}'
        if not os.path.exists(self.save_dir):
            # create save dir
            os.makedirs(self.save_dir)
        img_file = os.path.join(self.save_dir, 'model.png')
        plot_model(self.model, to_file=img_file, show_shapes=True)
        Image(filename=img_file)

    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T


    
    def train_model(self):
        """
        Run the model.
        """
        self.make_data(oversample=True)

        self.output("Data Loaded")   

        max_iter = 140
        pretrain_epochs = 300
        
        self.make_model()
        

        self.output("Training autoencoder")
        early_stopping_cb = EarlyStopping(
            monitor='loss', patience=5, verbose=1, min_delta=0.0003)
        history = self.autoencoder.fit(
                                self.x,
                                self.x,
                                batch_size=self.batch_size,
                                epochs=pretrain_epochs, 
                                verbose=0,
                                callbacks=[early_stopping_cb])
        self.autoencoder.save_weights(os.path.join(self.save_dir, 'jae_weights.h5'))
        self.output("Trained autoencoder")
        if self.verbose > 0:
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.title('Autoencoder pretraining loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.show()

        # init cluster centres before train
        self.init_cluster_centers()

        # train full model
        self.train()
        print("Training Done")


    def train(self):
        loss = 0
        index = 0
        update_interval = 140
        index_array = np.arange(self.x.shape[0])
        tol = 0.001  # tolerance threshold to stop training

        for ite in range(int(self.maxiter)):
            if ite % update_interval == 0:
                q, _ = self.model.predict(self.x, verbose=0)
                # update the auxiliary target distribution p
                p = self.target_distribution(q)

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if self.y is not None:
                    acc = np.round(metrics.acc(self.y, y_pred), 5)
                    nmi = np.round(metrics.nmi(self.y, y_pred), 5)
                    ari = np.round(metrics.ari(self.y, y_pred), 5)
                    loss = np.round(loss, 5)
                    self.output(f'Iter: {ite} Acc = {acc:.5f}, nmi = {nmi:.5f}, '
                                f'ari = {ari:.5f} ; loss={loss}')

                # check stop criterion
                delta_label = np.sum(y_pred != self.y_pred_last).astype(
                    np.float32) / y_pred.shape[0]
                self.y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    self.output(f'delta_label {delta_label} < tol {tol}')
                    self.output('Reached tolerance threshold. Stopping training.')
                    break
            idx = index_array[
                    index * self.batch_size : 
                    min((index+1) * self.batch_size, self.x.shape[0])]
            loss = self.model.train_on_batch(
                                            x=self.x[idx],
                                            y=[p[idx],
                                            self.x[idx]],
                                            reset_metrics=True,)
            try:
                if (index + 1) * self.batch_size <= self.x.shape[0]:
                    index = index + 1
                else:
                    index = 0

            except:
                print('e')

        if self.verbose == 0:
            # final values
            print(f'Iter: {ite} Acc = {acc:.5f}, nmi = {nmi:.5f}, '
                    f'ari = {ari:.5f} ; loss={loss}')
        self.model.save_weights(os.path.join(self.save_dir, 'DEC_model_final.h5'))

    def cluster_pred_acc(self):
        NER_only= DataFrame({'y':self.y, 'y_clus':self.y_pred})
        unk_idx = [k for k, v in self.mapping.items() if v == 'UNKNOWN'][0]
        NER_only.drop(NER_only.index[NER_only['y']==unk_idx], inplace=True)
        NER_match = NER_only[NER_only['y']==NER_only['y_clus']]
        # fraction that match
        frac = NER_match.shape[0]/NER_only.shape[0]
        return frac

    def evaluate_model(self, eval_size: int) -> None:
        """
        Run the model.
        """
        if self.train_size != eval_size:
            self.output("Load Data")
            self.x, self.y, self.mapping, self.strings = load_data(
                                                            eval_size,
                                                            get_text=True)
            self.output("Data Loaded")   

        self.make_model()

        ae_weights_file = os.path.join(self.save_dir, 'jae_weights.h5')
        self.output(f"Loading AE weights from {ae_weights_file}")
        self.autoencoder.load_weights(ae_weights_file)
        
        model_weights_file = os.path.join(self.save_dir, 'DEC_model_final.h5')
        self.output(f"Loading model weights from {model_weights_file}")
        self.model.load_weights(model_weights_file)
        
        # predict cluster labels
        self.output("Predicting...")
        q, _ = self.model.predict(self.x, verbose=1)
        p = self.target_distribution(q)  # update the auxiliary target distribution p


        # evaluate the clustering performance
        self.output("Evaluating...")
        self.y_pred = q.argmax(1)
        if self.y is not None:
            acc = np.round(metrics.acc(self.y, self.y_pred), 5)
            nmi = np.round(metrics.nmi(self.y, self.y_pred), 5)
            ari = np.round(metrics.ari(self.y, self.y_pred), 5)
            cluster_acc = self.cluster_pred_acc()
            print(f'Acc = {acc:.5f}, nmi = {nmi:.5f}, ari = {ari:.5f}'
                  f' ; Cluster Acc={cluster_acc:.5f}')

        # confusion matrix
            nmi = np.round(metrics.nmi(self.y, self.y_pred), 5)
        cm_width = max(8, len(np.unique(self.y_pred)) * 2)
        cm_width = min(16, cm_width)
        plot_confusion(self.y, self.y_pred, self.mapping, self.save_dir, cm_width)

        # show wordclouds for each cluster
        self.output ("CLUSTERS")
        clusters = {}
        predicted = DataFrame({
            'text':self.strings,
            'y_pred':self.y_pred,
            'y_true':self.y})
        for cluster_no in tqdm(range(self.num_clusters)):
            y_pred_for_key = predicted[predicted['y_pred']==cluster_no]
            true_label = 'UNKNOWN'
            modal_value = y_pred_for_key['y_true'].mode()
            if len(modal_value)>0:
                if modal_value[0] in self.mapping:
                    true_label = self.mapping[modal_value[0]]
                # confidence - fraction of this cluster that is actually this cluster
                y_true_this_cluster = len(
                    y_pred_for_key[y_pred_for_key['y_true']==modal_value[0]])
                frac = y_true_this_cluster/len(y_pred_for_key)
            else:
                frac = 0

            # wordcloud
            unique, counts = np.unique(y_pred_for_key['text'], return_counts=True)
            freq_list = np.asarray((unique, counts)).T
            freq_list =  sorted(freq_list, key=lambda x: -x[1])[0:50]
            freqs = {w: f for w,f in freq_list}
            entry = {'freqs':freqs, 'frac':frac, 'n':len(y_pred_for_key)}
            if true_label == 'UNKNOWN':
                clusters[f"UNK-{cluster_no}"] = entry
            elif true_label in clusters:
                if clusters[true_label]['frac'] < frac:
                    # we found a better cluster for this label
                    clusters[true_label] = entry
                else:
                    # this cluster is worse than this one, so it's unknown
                    clusters[f"UNK-{cluster_no} Was {true_label}"] = entry
            else:
                clusters[true_label] = entry

        cluster_list = [{
            **clusters[c],
            'name': c,
            'idx': idx} for idx, c in enumerate(clusters)]
        cluster_list = sorted(cluster_list, key=lambda x: -x['frac'])

        display_list = []
        # show unknown clusters first
        for i, cluster in enumerate(cluster_list):
            if cluster['name'][0:3] == "UNK":
                save_file = os.path.join(self.save_dir,
                                        f"wordcloud-{cluster['name']}.png")
                show_wordcloud(i, cluster, save_file, save_only=True)
                display_list.append(cluster)

        # next show known clusters
        for i, cluster in enumerate(cluster_list):
            if cluster['name'][0:3] != "UNK":
                save_file = os.path.join(self.save_dir,
                                        f"wordcloud-{cluster['name']}.png")
                show_wordcloud(i, cluster, save_file, save_only=True)
                display_list.append(cluster)

        
        self.output(write_results_page(display_list, self.save_dir, self.run_name))


    def visualise_tsne(self):
        tsne = TSNE(
                n_components=2,
                verbose=1,
                random_state=123,
                n_iter=300,
                learning_rate='auto')
        x_enc = self.encoder.predict(self.x)
        z = tsne.fit_transform(x_enc)
        df_tsne = pd.DataFrame()
        df_tsne["y"] = self.y_pred
        df_tsne["comp-1"] = z[:,0]
        df_tsne["comp-2"] = z[:,1]
        plt.figure(figsize=(18,14))
        sns.scatterplot(x="comp-1", y="comp-2", hue=df_tsne.y.tolist(),
                palette=sns.color_palette(
                        "hls",
                        len(ENTITY_FILTER_LIST)),
                data=df_tsne).set(title="Labelled embeddings T-SNE projection") 

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

# %%
stop

# %% [markdown]
# # Evaluate

# %%
# %history -g -f jupyter_history.py


# %%
del dc

# %%
dc = DeepCluster('test-0-40latent', dims=[768, 500, 500, 2000, 40],
    entity_count=10, train_size=0, num_clusters=25, maxiter=2000)
dc.train_and_evaluate_model(10000, verbose=1)

# %%

dc = DeepCluster('test-0-100latent', dims=[768, 500, 500, 2000, 100],
    entity_count=10, train_size=0, num_clusters=25, maxiter=1000)
dc.train_and_evaluate_model(10000, verbose=1)

# %%
dc = DeepCluster('test-0-250latent', dims=[768, 500, 500, 2000, 250],
    entity_count=10, train_size=0, num_clusters=25, maxiter=1000)
dc.train_and_evaluate_model(10000, verbose=1)

# %%

dc.visualise_tsne()

# %%
dc = DeepCluster('test1', train_size=0, num_clusters=25).train_and_evaluate_model(10000)

# %%
# %history -g -f jupyter_history3.py

# %%
make_data(10000, oversample=False)
"Done"

# %%
dc = DeepCluster('test-0', entity_count=10, train_size=0, num_clusters=25).train_and_evaluate_model(10000)


# %%
train_and_evaluate_model('test-none-3k', train_size=3000, eval_size=10000, n_clusters=25, entity_count=10)


# %%
model = evaluate_model('test-none-3k', eval_size=10000, n_clusters=25)


# %%

serialise_model(model, 'test-none-3k')

# %%
models.load_model('./results/test-none-3k')

# %%
train_model('test-none-10k', train_size=10000, n_clusters=25, entity_count=10)


# %%
evaluate_model('test-none-10k', eval_size=10000, n_clusters=25)


# %%
evaluate_model('test-none-30', train_size=30, eval_size=1000, n_clusters=25)


# %%
train_and_evaluate_model('test3', train_size=1000, eval_size=10000, n_clusters=25, entity_count=10)


# %%
train_and_evaluate_model('test1', train_size=10000, eval_size=10000, n_clusters=25, entity_count=0)

# %%
train_and_evaluate_model('test1-2', train_size=10000, eval_size=10000, n_clusters=25, entity_count=0)

# %%
evaluate_model('test1-2', eval_size=10000, n_clusters=25, include_unclass=True)

# %%
train_model('test1', cluster="GMM", data_rows=1000, entity_count=0, n_clusters=20 )

# %%
train_model('test1', cluster="GMM", data_rows=1000, entity_count=0, n_clusters=20 )


# %%
evaluate_model('test1', data_rows=1000, n_clusters=20 )

# %%
evaluate_model('test1', data_rows=1000, entity_count=0, n_clusters=20 )

# %%
train_and_evaluate_model('test2', train_size=10000, eval_size=10000, n_clusters=15, entity_count=10)

# %%
train_model('reset-metrics', cluster='Kmeans', data_rows=1000, entity_count=0, n_clusters=20 )


# %%
train_model('reset-metrics', cluster='Kmeans', data_rows=10000, entity_count=10, n_clusters=15 )


# %%
train_model('reset-metrics-dbscan', cluster="DBSCAN", data_rows=1000, entity_count=0, n_clusters=20 )


# %%
train_model('reset-metrics-dbscan', cluster="DBSCAN", data_rows=1000, entity_count=10, n_clusters=15)


# %%


# %%
train_model('reset-metrics-dbscan', cluster="OPTICS", data_rows=1000, entity_count=0, n_clusters=20 )

# %%
train_model('reset-metrics-optics', cluster="OPTICS", data_rows=1000, entity_count=10, n_clusters=15 )


# %%
evaluate_model('reset-metrics-dbscan',
        entity_count=10,
        data_rows=1000,
        n_clusters=20,
        cluster="DBSCAN",
        )

# %%


# %% [markdown]
# # benchmark

# %%
# optimal eps https://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012/pdf

from sklearn.neighbors import NearestNeighbors

def optimal_eps(X, n_neighbors=10):
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)

# %%
def cluster_score(y, y_pred, n_clusters):
    """
    Compute the cluster score.
    Arguments:
        y: true labels.
        y_pred: predicted labels.
        n_clusters: number of clusters.
    Returns:
        cluster score.
    """
    # compute the cluster score
    score = 0
    for i in range(n_clusters):
        score += np.sum(y_pred[y==i]==i)
    return score/len(y)

# %%
def hypertune_density_clustering():
    """
    hypertune the density clustering algorithms.
    """
    eps_vals = [30000.0, 40000.0, 50000.0]
    x, y, mapping, strings = load_data(
                                    1000,
                                    oversample=True,
                                    get_text=True)
    print(f"Optimal epsilon: {optimal_eps(x)}")
    for eps in eps_vals:
        # predict cluster labels
        print(f"Predicting...for epsilon={eps}")
        y_pred, _ = do_clustering('DBSCAN', 25, x, params={'eps':eps})
        print(f"ACC: {cluster_score(y, y_pred, 25)}")
        # confusion matrix
        cm_width = max(8, len(np.unique(y_pred)) * 2)
        cm_width = min(16, cm_width)
        plot_confusion(y, y_pred, mapping, size=cm_width, save_dir=None, details=False)

# %%
hypertune_density_clustering()

# %%
def run_benchmark(cluster:str, eval_size:int, n_clusters:int):
    x, y, mapping, strings = load_data(
                                    eval_size,
                                    oversample=False,
                                    get_text=True)
    save_dir = f'./results/bm/{cluster}'
    if not os.path.exists(save_dir):
        # create save dir
        os.makedirs(save_dir)

    
    # predict cluster labels
    print("Predicting...")
    y_pred, _ = do_clustering(cluster, n_clusters, x)
    # print(f"ACC: {cluster_acc(y, y_pred)}")
    
    # confusion matrix
    cm_width = max(8, len(np.unique(y_pred)) * 2)
    cm_width = min(16, cm_width)
    plot_confusion(y, y_pred, mapping, save_dir, cm_width)

    # show wordclouds for each cluster
    print ("BENCHMARK CLUSTERS")
    clusters = {}
    predicted = DataFrame({'text':strings, 'y_pred':y_pred, 'y_true':y})
    for cluster_no in range(n_clusters):
        y_pred_for_key = predicted[predicted['y_pred']==cluster_no]
        true_label = 'UNKNOWN'
        modal_value = y_pred_for_key['y_true'].mode()
        if len(modal_value)>0:
            if modal_value[0] in mapping:
                true_label = mapping[modal_value[0]]
            # confidence - fraction of this cluster that is actually this cluster
            y_true_this_cluster = len(
                y_pred_for_key[y_pred_for_key['y_true']==modal_value[0]])
            frac = y_true_this_cluster/len(y_pred_for_key)
        else:
            frac = 0

        # wordcloud
        unique, counts = np.unique(y_pred_for_key['text'], return_counts=True)
        freq_list = np.asarray((unique, counts)).T
        freq_list =  sorted(freq_list, key=lambda x: -x[1])[0:50]
        freqs = {w: f for w,f in freq_list}
        entry = {'freqs':freqs, 'frac':frac, 'n':len(y_pred_for_key)}
        if true_label == 'UNKNOWN':
            clusters[f"UNK-{cluster_no}"] = entry
        elif true_label in clusters:
            if clusters[true_label]['frac'] < frac:
                # we found a better cluster for this label
                clusters[true_label] = entry
            else:
                # this cluster is worse than this one, so it's unknown
                clusters[f"UNK-{cluster_no} Was {true_label}"] = entry
        else:
            clusters[true_label] = entry

    cluster_list = [{
        **clusters[c],
        'name': c,
        'idx': idx} for idx, c in enumerate(clusters)]
    cluster_list = sorted(cluster_list, key=lambda x: -x['frac'])

    display_list = []
    # show unknown clusters first
    for i, cluster in enumerate(cluster_list):
        if cluster['name'][0:3] == "UNK":
            save_file = os.path.join(save_dir,
                                     f"wordcloud-{cluster['name']}.png")
            show_wordcloud(i, cluster, save_file, save_only=True)
            display_list.append(cluster)

    # next show known clusters
    for i, cluster in enumerate(cluster_list):
        if cluster['name'][0:3] != "UNK":
            save_file = os.path.join(save_dir,
                                     f"wordcloud-{cluster['name']}.png")
            show_wordcloud(i, cluster, save_file, save_only=True)
            display_list.append(cluster)

    
    print(write_results_page(display_list, save_dir, cluster))

# %%
run_benchmark('Kmeans', 10000, 25)

# %%
run_benchmark('GMM', 10000, 25)

# %%
run_benchmark('agg', 10000, 25)


