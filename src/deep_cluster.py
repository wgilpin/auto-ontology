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
import umap

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
        """
        Predict the cluster labels y_pred and calculate the accuracy against y.
        """
        NER_only= DataFrame({'y':self.y, 'y_clus':self.y_pred})
        unk_tuple = [k for k, v in self.mapping.items() if v == 'UNKNOWN']
        unk_idx = unk_tuple[0] if len(unk_tuple) > 0 else None
        NER_only.drop(NER_only.index[NER_only['y']==unk_idx], inplace=True)
        NER_match = NER_only[NER_only['y']==NER_only['y_clus']]
        # fraction that match
        frac = NER_match.shape[0]/NER_only.shape[0]
        return frac

    def make_load_model(self):
        """
        Make the model and load the weights.
        """
        self.make_model()

        ae_weights_file = os.path.join(self.save_dir, 'jae_weights.h5')
        self.output(f"Loading AE weights from {ae_weights_file}")
        self.autoencoder.load_weights(ae_weights_file)

        model_weights_file = os.path.join(self.save_dir, 'DEC_model_final.h5')
        self.output(f"Loading model weights from {model_weights_file}")
        self.model.load_weights(model_weights_file)

    def evaluate_model(self, eval_size: int, verbose:int=1) -> None:
        """
        Run the model.
        """
        self.verbose = verbose

        if self.train_size != eval_size:
            self.output("Load Data")
            self.x, self.y, self.mapping, self.strings = load_data(
                                                            eval_size,
                                                            get_text=True,
                                                            verbose=verbose)
            self.output("Data Loaded")   

        self.make_load_model()

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

    
    def visualise_umap(self, sample:int=1000, embs:str="z"):
        if embs == "z":
            # encoder output
            z_enc = self.encoder.predict(self.x)
        elif embs == "x":
            # raw BERT embeddings
            z_enc = self.x
        indices = np.random.choice(z_enc.shape[0], sample, replace=False)
        labels = self.y_pred[indices]
        labels = np.asarray(
            [(self.mapping[l] if l in self.mapping else l) for l in labels ])
        z_sample = z_enc[indices]
        mapper = umap.UMAP(metric='manhattan').fit(z_sample)
        import umap.plot as plt_u
        plt_u.points(mapper, labels=labels)
    
    
    

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
