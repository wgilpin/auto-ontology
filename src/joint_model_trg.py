# %% [markdown]
# # Model Build

# %%
# %history -g -f history.py

# %%

# import os
# import math
# import numpy as np
# import glob
# from typing import Any
import tensorflow as tf
# import tensorflow.keras as k
# import matplotlib.pyplot as plt
# import metrics
# from pandas import DataFrame, crosstab
# from metrics import plot_confusion
# from IPython.display import Image
# from tensorflow.keras import models
# from keras.utils import plot_model
# from tqdm.notebook import trange
# from tensorflow.keras.layers import Dense, Input, Layer, InputSpec, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.initializers import VarianceScaling
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.cluster import KMeans
# from sklearn.metrics import (
#             f1_score, accuracy_score, precision_score, recall_score, classification_report)
# from data import load_data
# from wordcloud import WordCloud
# from timer import timer
from grid_search import grid_search
from deep_latent import DeepLatentCluster
# import umap

# %%
tf.get_logger().setLevel('ERROR')

# %%
from collections import defaultdict

def summarise_scores(scores: list[dict]) -> None:
    """
    summarise the scores.
    """
    groups = defaultdict(list)
    results = {}
    for s in scores:
        groups[s['run_name']].append(s)
        results[s['run_name']] = {
                            'cluster_f1':0.0,
                            'f1': 0.0,
                            'precision': 0.0,
                            'recall': 0.0,
                            'n': 0}
    # add up each group
    for group, scores in groups.items():
        for score in scores:
            results[group]['cluster_f1'] += score['cluster_f1']
            results[group]['f1'] += score['f1']
            results[group]['precision'] += score['precision']
            results[group]['recall'] += score['recall']
            results[group]['n'] += 1
    # average each group
    for group, scores in groups.items():
        results[group]['cluster_f1'] /= len(scores)
        results[group]['f1'] /= len(scores)
        results[group]['precision'] /= len(scores)
        results[group]['recall'] /= len(scores)


    # print results
    print(
        f"{'Run Name':<40} {'Runs':<10} {'F1 avg':<12} {'F1 Cluster':<12} "
        f"{'F1 Global':<12} {'Precision':<12} {'Recall':<12}")
    for run, s in results.items():
        print(
            f"{run:<40} {s['n']:<10}  {(s['f1']+s['cluster_f1'])/2:<12.4f} "
            f"{s['cluster_f1']:<12.4f} {s['f1']:<12.4f}"
            f"{s['precision']:<12.4f} {s['recall']:<12.4f}")

# %% [markdown]
# ## Grid Search Utils

# %%
def do_run(cfg, idx, n_runs):
    run_name = (f"test-")
    

    # append all cfg to run_name
    for k, v in cfg.items():
        run_name += f"{k}={v}-"

    print('-'*50)
    print(f"{idx}/{n_runs}: {run_name}")
    print('-'*50)
    print(cfg)
    dc = DeepLatentCluster(
        run_name,
        {
            **cfg,
        
        })
    dc.make_model()
    dc.train_model(verbose=0)
    score = dc.evaluate_model(
        run_name,
        sample_size=3000,
        verbose=0)
    score['run_name'] = run_name
    return score
    
from collections import defaultdict

def summarise_scores(scores: list[dict]) -> None:
    """
    summarise the scores.
    """
    groups = defaultdict(list)
    results = {}
    for s in scores:
        groups[s['run_name']].append(s)
        results[s['run_name']] = {
                            'cluster_f1':0.0,
                            'f1': 0.0,
                            'precision': 0.0,
                            'recall': 0.0,
                            'n': 0}
    # add up each group
    for group, scores in groups.items():
        for score in scores:
            results[group]['cluster_f1'] += score['cluster F1']
            results[group]['f1'] += score['f1']
            results[group]['precision'] += score['precision']
            results[group]['recall'] += score['recall']
            results[group]['n'] += 1
    # average each group
    for group, scores in groups.items():
        results[group]['cluster_f1'] /= len(scores)
        results[group]['f1'] /= len(scores)
        results[group]['precision'] /= len(scores)
        results[group]['recall'] /= len(scores)


    # print results
    print(
        f"{'Run Name':<40} {'Runs':<10} {'F1 avg':<12} {'F1 Cluster':<12} "
        f"{'F1 Global':<12} {'Precision':<12} {'Recall':<12}")
    for run, s in results.items():
        print(
            f"{run:<40} {s['n']:<10}  {(s['f1']+s['cluster_f1'])/2:<12.4f} "
            f"{s['cluster_f1']:<12.4f} {s['f1']:<12.4f}"
            f"{s['precision']:<12.4f} {s['recall']:<12.4f}")

# %%
stop

# %% [markdown]
# # Eval

# %% [markdown]
# ## Base model

# %%
tf.get_logger().setLevel('ERROR')

dc = None
dc = DeepLatentCluster(
        'test-latent-all',
        {
            'train_size':0,
            'reconstr_weight':1.0,
            'latent_weight':1e-5,
            "cluster": None
            "noise_factor": 0.0,
        })
dc.make_model()
dc.train_model()

# %% [markdown]
# # Latent Head

# %% [markdown]
# ## OPTICS

# %%
#%%time

dc = None
dc = DeepLatentCluster(
    'test-latent-all-OPTICS',
    {
        'train_size':0,
        'reconstr_weight':1.0,
        'latent_weight':1e-5,
        "cluster": "OPTICS"
    })

dc.evaluate_model('test-latent-all', sample_size=2000)

# %%
dc.evaluate_model('test-latent-all', sample_size=2000)

# %% [markdown]
# ## Agglomerative Clustering

# %%
%%time

tf.get_logger().setLevel('ERROR')

dc = None
dc = DeepLatentCluster(
    'test-latent-all-agg',
    {
        'train_size':10000,
        'reconstr_weight':1.0,
        'latent_weight':1e-5,
        "cluster": "agg"
    })
# dc.make_model()
# dc.train_model()
dc.evaluate_model('test-latent-all', sample_size=4000)

# %% [markdown]
# ## K-means

# %%
%%time

dc = None
dc = DeepLatentCluster(
    'test-latent-all-Kmeans',
    {
        'train_size':0,
        'reconstr_weight':1.0,
        'latent_weight':1e-5,
        "cluster": "Kmeans"
    })
# dc.make_model()
# dc.train_model()
dc.evaluate_model('test-latent-all', sample_size=4000)

# %% [markdown]
# ## GMM

# %%
%%time

dc = None
dc = DeepLatentCluster(
    'test-latent-all-GMM',
    {
        'train_size':0,
        'reconstr_weight':1.0,
        'latent_weight':1e-5,
        "cluster": "GMM"
    })
# dc.make_model()
# dc.train_model()
dc.evaluate_model('test-latent-all', sample_size=4000)

# %% [markdown]
# # Encoder Head

# %%
%%time
#min cluster size

dc = None
dc = DeepLatentCluster(
    'test-latent-all-OPTICS-Enc-2',
    {
        'train_size':0,
        'reconstr_weight':1.0,
        'latent_weight':1e-5,
        "cluster": "OPTICS"
    })

dc.evaluate_model('test-latent-all', head="enc", sample_size=3000)

# %% [markdown]
# ## Optics-Encoder

# %%
%%time

dc = None
dc = DeepLatentCluster(
    'test-latent-all-OPTICS-Enc',
    {
        'train_size':0,
        'reconstr_weight':1.0,
        'latent_weight':1e-5,
        "cluster": "OPTICS"
    })

dc.evaluate_model('test-latent-all', head="enc", sample_size=3000, verbose=0)

# %% [markdown]
# ## Agglomerative-Encoder

# %%
%%time

dc = None
dc = DeepLatentCluster(
    'test-latent-all-agg-enc',
    {
        'train_size':0,
        'reconstr_weight':1.0,
        'latent_weight':1e-5,
        "cluster": "agg"
    })
# dc.make_model()
# dc.train_model()
dc.evaluate_model('test-latent-all', head='enc',  sample_size=4000, verbose=0)

# %% [markdown]
# ## KMeans-Encoder

# %%
%%time

dc = None
dc = DeepLatentCluster(
    'test-latent-all-Kmeans-Enc',
    {
        'train_size':0,
        'reconstr_weight':1.0,
        'latent_weight':1e-5,
        "cluster": "Kmeans"
    })

dc.evaluate_model('test-latent-all', head="enc", sample_size=4000, verbose=0)

# %% [markdown]
# ## GMM-Encoder

# %%
%%time

dc = None
dc = DeepLatentCluster(
    'test-latent-all-GMM-Enc',
    {
        'train_size':0,
        'reconstr_weight':1.0,
        'latent_weight':1e-5,
        "cluster": "GMM"
    })

dc.evaluate_model('test-latent-all', head="enc", sample_size=4000, verbose=0)

# %% [markdown]
# # Decoder Head
# 

# %% [markdown]
# ## Optics-AE

# %%
%%time

dc = None
dc = DeepLatentCluster(
    'test-latent-all-OPTICS-AE',
    {
        'train_size':0,
        'reconstr_weight':1.0,
        'latent_weight':1e-5,
        "cluster": "OPTICS"
    })

dc.evaluate_model('test-latent-all', head="ae", sample_size=3000, verbose=0)

# %% [markdown]
# ## GMM-AE

# %%
%%time

dc = None
dc = DeepLatentCluster(
    'test-latent-all-GMM-AE2',
    {
        'train_size':0,
        'reconstr_weight':1.0,
        'latent_weight':1e-5,
        "cluster": "GMM"
    })

dc.evaluate_model('test-latent-all', head="ae", sample_size=4000, verbose=0)

# %%
%%time

dc = None
dc = DeepLatentCluster(
    'test-latent-all-GMM-AE',
    {
        'train_size':0,
        'reconstr_weight':1.0,
        'latent_weight':1e-5,
        "cluster": "GMM"
    })

dc.evaluate_model('test-latent-all', head="ae", sample_size=4000, verbose=0)

# %% [markdown]
# ## Kmeans-AE

# %%
%%time

dc = None
dc = DeepLatentCluster(
    'test-latent-all-Kmeans-AE',
    {
        'train_size':0,
        'reconstr_weight':1.0,
        'latent_weight':1e-5,
        "cluster": "Kmeans"
    })

dc.evaluate_model('test-latent-all', head="ae", sample_size=4000, verbose=0)

# %%
stop

# %%
%history -f history.py

# %% [markdown]
# # Benchmarks

# %%
%%time
# benchmark with cluster rearrangement

dc = None
dc = DeepLatentCluster(
    'benchmark-10k-with-rearrange',
    {
        'train_size':0,
        "cluster": "Kmeans"
    })
# dc.make_model()
# dc.train_model()
dc.benchmark_model(sample_size=4000, verbose=0)

# %%
dc.visualise_umap(dc.x_sample, dc.y_sample, to_dir=False)

# %%
%%time
# benchmark without cluster rearrangement

dc = None
dc = DeepLatentCluster(
    'benchmark-10k-no-rearrange',
    {
        'train_size':0,
        'reconstr_weight':1.0,
        'latent_weight':1e-5,
        "cluster": "Kmeans"
    })
# dc.make_model()
# dc.train_model()
dc.random_benchmark(sample_size=4000, verbose=0)

# %%
%%time

dc = None
dc = DeepLatentCluster(
    'test-latent-10k-Benchmark',
    {
        'train_size':0,
        'reconstr_weight':1.0,
        'latent_weight':1e-5,
        "cluster": "OPTICS"
    })
# dc.make_model()
# dc.train_model()
dc.benchmark_model(sample_size=2000, verbose=0)

# %%
stop

# %%
tf.get_logger().setLevel('ERROR')

dc = DeepLatentCluster(
    'test-latent-all-10ent',
    {
        'entity_count': 10,
        'train_size':0,
        'reconstr_weight':1.0, 'latent_weight':1e-5
    })
dc.make_model()
# dc.train_model()

# %%
dc.evaluate_model(10000, sample_size=1000)

# %% [markdown]
# # Hypertuning

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

# %% [markdown]
# # Train with noise

# %%
tf.get_logger().setLevel('ERROR')

dc = None
dc = DeepLatentCluster(
        'test-latent-noise',
        {
            'train_size':0,
            'reconstr_weight':1.0,
            'latent_weight':1e-5,
            "cluster": None,
            "noise_factor": 0.5,
        })
dc.make_model()
dc.train_model(verbose=0)

# %% [markdown]
# # Evaluate all 

# %% [markdown]
# ## different entity counts in training

# %% [markdown]
# ## No noise

# %%
for entity_count in [0, 5, 10, 15]:
    run_name = f'test-latent-no-noise-{entity_count}-ents'
    print('-'*50)
    print(f"{run_name}")
    print('-'*50)

    dc = None
    dc = DeepLatentCluster(
        run_name,
        {
            'train_size':0,
            'reconstr_weight':1.0,
            'latent_weight':1e-5,
            "cluster": 'Kmeans',
            "entity_count": entity_count,
        })
    dc.make_model()
    # dc.train_model(verbose=0)
    dc.evaluate_model(
            run_name,
            head='z',
            sample_size=4000,
            verbose=0)

# %% [markdown]
# ## with noise

# %%
for entity_count in [0, 5, 10, 15]:
    run_name = f'test-latent-noise-{entity_count}-ents'
    print('-'*50)
    print(f"{run_name}")
    print('-'*50)

    dc = None
    dc = DeepLatentCluster(
        run_name,
        {
            'train_size':0,
            'reconstr_weight':1.0,
            'latent_weight':1e-5,
            "cluster": 'Kmeans',
            "entity_count": entity_count,
            "noise_factor": 0.5,
        })
    dc.make_model()
    # dc.train_model(verbose=0)
    dc.evaluate_model(
            run_name,
            head='z',
            sample_size=4000,
            verbose=0)

# %% [markdown]
# # Grid Search Clustering

# %%


# %% [markdown]
# ## with noise

# %%
heads = ["z", "ae", "enc"]
clusterers = {"Kmeans": 4000, "GMM": 4000, "OPTICS":3000, "agg":4000}
repeats = 3

scores = []

for head in heads:
    for clusterer in clusterers:
        run_name = f'test-latent-noise-15-ents-{head}-{clusterer}'
        print('-'*50)
        print(f"{run_name}")
        print('-'*50)
        
        for r in range(repeats):
            print(f"Run {r+1}")
            dc = None
            dc = DeepLatentCluster(
                run_name,
                {
                    'train_size':0,
                    'reconstr_weight':1.0,
                    'latent_weight':1e-5,
                    "cluster": clusterer,
                })
            
            score = dc.evaluate_model(
                                f'test-latent-noise-15-ents',
                                head=head,
                                sample_size=clusterers[clusterer],
                                verbose=0)

            score['run_name'] = run_name
            scores.append(score)
        


# %%
summarise_scores(scores)

# %% [markdown]
# ## Evaluate all without noise

# %%
heads = ["z", "ae", "enc"]
clusterers = {"Kmeans": 4000, "GMM": 4000, "OPTICS":3000, "agg":4000}

runs = 3
scores = []
for head in heads:
    for clusterer in clusterers:
        for r in range(runs):
            print(f"Run {r+1}")
            run_name = f'test-latent-noise-15-ents-{head}-{clusterer}'
            
            print('-'*50)
            print(f"{run_name}")
            print('-'*50)
            
            dc = None
            dc = DeepLatentCluster(
                run_name,
                {
                    'train_size':0,
                    'reconstr_weight':1.0,
                    'latent_weight':1e-5,
                    "cluster": clusterer,
                })
            
            score = dc.evaluate_model(
                        f'test-latent-noise-15-ents',
                        head=head,
                        sample_size=clusterers[clusterer],
                        verbose=0)
            score['run_name'] = run_name
            scores.append(score)

# %%
summarise_scores(scores)


# %% [markdown]
# # Radius BMs

# %%
# assert False, "One time run"
for radius in [8, 10]:
    load_data(0, oversample=False, radius=radius, verbose=0)

# %% [markdown]
# ## AE-head different radii

# %%
scores = []
num_runs = 3
for radius in [0,2,4,6, 8, 10]:
    for r in range(num_runs):
        run_name = f'test-latent-noise-15-ae-r{radius}-Kmeans'
        print('-'*50)
        print(f"{run_name}")
        print('-'*50)
        
        dc = None
        dc = DeepLatentCluster(
            run_name,
            {
                'train_size':0,
                'reconstr_weight':1.0,
                'latent_weight':1e-5,
                "cluster": "Kmeans",
                "radius": radius,
            })
        dc.make_model()
        dc.train_model(verbose=0)
        score = dc.evaluate_model(
                            run_name,
                            head="ae",
                            sample_size=4000,
                            verbose=0)
        score['run_name'] = run_name
        scores.append(score)
        


# %%
summarise_scores(scores)

# %% [markdown]
# ## Z-head different radii

# %%
scores = []
num_runs = 3
for radius in [0,2,4,6, 8, 10]:
    for r in range(num_runs):
        run_name = f'test-latent-noise-15-ents-r{radius}-Kmeans'
        print('-'*50)
        print(f"{run_name}")
        print('-'*50)
        
        dc = None
        dc = DeepLatentCluster(
            run_name,
            {
                'train_size':0,
                'reconstr_weight':1.0,
                'latent_weight':1e-5,
                "cluster": "Kmeans",
                "radius": radius,
            })
        # dc.make_model()
        # dc.train_model(verbose=0)
        score = dc.evaluate_model(
                            run_name,
                            head="z",
                            sample_size=4000,
                            verbose=0)
        score['run_name'] = run_name
        scores.append(score)
        
summarise_scores(scores)

# %% [markdown]
# ## Encoder head, different radii

# %%
scores = []
num_runs = 3
for radius in [0, 2, 4, 6, 8, 10]:
    for r in range(num_runs):
        run_name = f'test-latent-noise-15-ents-r{radius}-Kmeans-enc'
        print('-'*50)
        print(f"{run_name}")
        print('-'*50)
        
        dc = None
        dc = DeepLatentCluster(
            run_name,
            {
                'train_size':0,
                'reconstr_weight':1.0,
                'latent_weight':1e-5,
                "cluster": "Kmeans",
                "radius": radius,
            })
        dc.make_model()
        dc.train_model(verbose=0)
        score = dc.evaluate_model(
                            run_name,
                            head="enc",
                            sample_size=4000,
                            verbose=0)
        score['run_name'] = run_name
        scores.append(score)


# %%
summarise_scores(scores)

# %% [markdown]
# ## Data Balance

# %%
dc = None
dc = DeepLatentCluster(
    'test-latent-all',
    {
        'train_size':0,
        'entity_count': 15,
    })
dc.make_data(oversample=True)

# %%
from collections import defaultdict

cats = defaultdict(lambda: "Unknown",
{0: 'PERSON', 1: 'NORP', 2: 'ORG', 3: 'GPE', 4: 'LOC', 5: 'PRODUCT', 6: 'EVENT', 7: 'WORK_OF_ART',
        8: 'DATE', 9: 'TIME', 10: 'PERCENT', 11: 'MONEY', 12: 'QUANTITY', 13: 'CARDINAL', 14: 'FAC'})
pre = [[2, 2080],
       [3,  142],
       [4, 1950],
       [5, 2664],
       [6,  114],
       [7,   36],
       [8,   55],
       [9,   17],
       [12, 1659],
       [13,  143],
       [14,  151],
       [15,  111],
       [16,  139],
       [18,   62],
       [19,   95]]
sort_pre = sorted(pre, key=lambda x: -x[1])
# plot catplot
plt.xticks(rotation=90, ha='right')
plt.title("Class Distribution")
plt.bar([cats[x[0]] for x in sort_pre], [x[1] for x in sort_pre])
plt.show()


# %% [markdown]
# ## Loss Weights

# %%
scores = []
num_runs = 1
for i, l_weight in enumerate([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]):
    for j, r_weight in enumerate([0,0.25,0.5,0.75, 1.0]):
        for r in range(num_runs):
            run_name = f'test-latent-15-ents-lw{l_weight}-rw{r_weight}-Kmeans'
            print('-'*50)
            print(f"{(i)*8 + j}: {run_name}")
            print('-'*50)
            
            print({
                    'train_size':0,
                    'reconstr_weight':r_weight,
                    'latent_weight':l_weight,
                    "cluster": "Kmeans",
                    "radius": 6,
                })
            dc = None
            dc = DeepLatentCluster(
                run_name,
                {
                    'train_size':0,
                    'reconstr_weight':r_weight,
                    'latent_weight':l_weight,
                    "cluster": "Kmeans",
                    "radius": 6,
                })
            dc.make_model()
            dc.train_model(verbose=0)
            score = dc.evaluate_model(
                                run_name,
                                head="ae",
                                sample_size=2000,
                                verbose=0)
            score['run_name'] = run_name
            scores.append(score)
            
    summarise_scores(scores)

# %%
#%history -g -f jm_trg.py

# %%
from grid_search import grid_search

scores = []
num_runs = 1
config = {
    'train_size': [0],
    "radius": [6],
    'latent_weight': [0.4, 0.5, 1.0],
    'reconstr_weight': [0.8, 0.9, 1.0],
    'head': ['z'],
}


# %%

def do_run(cfg, idx, total):
    run_name = (f"test-latent-15-ents-lw{cfg['latent_weight']}-"
               f"rw{cfg['reconstr_weight']}-"
               f"{cfg['head']}-Kmeans")
    print('-'*50)
    print(f"{idx}/{total}: {run_name}")
    print('-'*50)
    print(cfg)
    dc = DeepLatentCluster(run_name, {**cfg})
    
    score = dc.evaluate_model(
        run_name,
        sample_size=2000,
        verbose=0)
    score['run_name'] = run_name
    scores.append(score)

grid_search(config, do_run, 5)

summarise_scores(scores)


# %%
from grid_search import grid_search

scores = []
num_runs = 1
config = {
    'train_size': [0],
    "radius": [6],
    'latent_weight': [0.05, 0.1, 0.2, 0.5, 1.0],
    'reconstr_weight': [0.1, 0.25, 0.5, 0.75, 1.0],
    'head': ['enc'],
}


def do_run(cfg, idx, n_runs):
    for n in range(n_runs):
        run_name = (f"test-latent-15-ents-lw{cfg['latent_weight']}-"
                f"rw{cfg['reconstr_weight']}-"
                f"{cfg['head']}-Kmeans")
        print('-'*50)
        print(f"{idx}: {run_name}")
        print('-'*50)
        print(cfg)
        dc = DeepLatentCluster(
            run_name,
            {
                **cfg,
            })
        # dc.make_model(1)
        # dc.train_model(verbose=0)
        score = dc.evaluate_model(
            run_name,
            sample_size=2000,
            verbose=0)
        score['run_name'] = run_name
        scores.append(score)

grid_search(config, do_run)



summarise_scores(scores)


# %% [markdown]
# # All together

# %%
from grid_search import grid_search

scores = []
num_runs = 1
config = {
    'train_size': [0],
    "radius": [6],
    'latent_weight': [0.9, 1.0, 1.1],
    'reconstr_weight': [0.65,0.75,0.85],
    'head': ['enc'],
    'cluster': ['OPTICS'],
    'noise_factor': [0.5]
}

grid_search(config, do_run)

# %%
summarise_scores(scores)

# %% [markdown]
# # Noise over best

# %%
from grid_search import grid_search

scores = []
num_runs = 1
config = {
    'train_size': [0],
    "radius": [6],
    'latent_weight': [0.9],
    'reconstr_weight': [0.75],
    'head': ['ae'],
    'cluster': ['OPTICS'],
    'noise_factor': [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
}

grid_search(config, do_run)

# %%
summarise_scores(scores)

# %% [markdown]
# ## Min cluster size for OPTICS

# %%
trained = False
ae_file = None
scores=[]
for min_cluster_size in [5, 10, 15, 20]:
    run_name = f'test-optics-mcs-{min_cluster_size}'
    print('-'*50)
    print(f"{run_name}")
    print('-'*50)

    dc = None
    dc = DeepLatentCluster(
        run_name,
        {
            'train_size':0,
            'reconstr_weight':0.75,
            'latent_weight':0.9,
            "cluster": 'OPTICS',
            "entity_count": 15,
            "noise_factor": 0.6,
            "head": "ae",
            "radius": 6,
            "alpha1": 80,
        })
    dc.make_model()
    
    if not trained:
        dc.train_model(verbose=0)
        trained = True
        ae_file = run_name

    score = dc.evaluate_model(
            run_name,
            head='z',
            sample_size=3000,
            verbose=0,
            config={
                'min_cluster_size': min_cluster_size,
                'ae_weights_file': ae_file,
                })
    score['run_name'] = run_name
    scores.append(score)

summarise_scores(scores)

# %%
run_name = f'test-kmeans-high-alpha'
print('-'*50)
print(f"{run_name}")
print('-'*50)

dc = None
dc = DeepLatentCluster(
    run_name,
    {
        'train_size':0,
        'reconstr_weight':0.75,
        'latent_weight':0.9,
        "cluster": 'Kmeans',
        "entity_count": 15,
        "noise_factor": 0.6,
        "head": "ae",
        "radius": 6,
        "alpha1": 80,
    })
dc.make_model()

# dc.train_model(verbose=0)

score = dc.evaluate_model(
        run_name,
        head='z',
        sample_size=4000,
        verbose=0,
        )
score['run_name'] = run_name
summarise_scores([score])


# %%
%tb plain

# %%
run_name = f'test-kmeans-high-alpha-enc'
print('-'*50)
print(f"{run_name}")
print('-'*50)

dc = None
dc = DeepLatentCluster(
    run_name,
    {
        'train_size':0,
        'reconstr_weight':0.75,
        'latent_weight':1.0,
        "cluster": 'Kmeans',
        "entity_count": 15,
        "noise_factor": 0.6,
        "head": "enc",
        "radius": 6,
        "alpha1": 80,
    })
dc.make_model()

dc.train_model(verbose=0)

score = dc.evaluate_model(
        run_name,
        head='enc',
        sample_size=2000,
        verbose=0,
        )
score['run_name'] = run_name
summarise_scores([score])


# %%
from grid_search import grid_search

num_runs = 1
config = {
    'train_size': [0],
    "radius": [6],
    'latent_weight': [1.0],
    'reconstr_weight': [0.75],
    'head': ['enc'],
    'cluster': ['Kmeans'],
    'noise_factor': [0.6],
    "entity_count": [15],
    'alpha1': [20, 40, 60, 80, 100]
}

scores = grid_search(config, do_run)

summarise_scores(scores)

# %%



