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
def do_run(cfg, idx, n_runs, sample_size) -> dict:
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
        sample_size=sample_size,
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

# %% [markdown]
# # Using KL Divergence
# 
# ## P from encoder, Q from latent space

# %%
from grid_search import grid_search

scores = []
num_runs = 1
config = {
    'train_size': [3000],
    "radius": [0, 6],
    'latent_weight': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    'reconstr_weight': [0.1, 0.25, 0.5, 0.75, 0.9],
    'head': ['z'],
    'cluster': ['Kmeans'],
    'entity_count': [15],
    'noise_factor': [0.1], 
}

scores = grid_search(config, do_run, sample_size=3000)

# %%
scores = [{'f1': 0.43730650547602323,
  'acc': 0.37333333333333335,
  'precision': 0.7934106920845421,
  'recall': 0.37419964193979244,
  'new_clusters': 1,
  'cluster F1': 0.14170528861085782,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=1e-05-reconstr_weight=0.1-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.432261611784046,
  'acc': 0.38133333333333336,
  'precision': 0.6878775887086137,
  'recall': 0.3861824971972584,
  'new_clusters': 4,
  'cluster F1': 0.2828108511719897,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=1e-05-reconstr_weight=0.25-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.45589782844549975,
  'acc': 0.4246666666666667,
  'precision': 0.794040657198552,
  'recall': 0.40814215633403317,
  'new_clusters': 5,
  'cluster F1': 0.2303709869152554,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=1e-05-reconstr_weight=0.5-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.5022728016131466,
  'acc': 0.41933333333333334,
  'precision': 0.8990372610493023,
  'recall': 0.41608696921851807,
  'new_clusters': 4,
  'cluster F1': 0.12564242345978052,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=1e-05-reconstr_weight=0.75-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4499187449452225,
  'acc': 0.4076666666666667,
  'precision': 0.6887318605154862,
  'recall': 0.4085278356537854,
  'new_clusters': 2,
  'cluster F1': 0.28484695556064943,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=1e-05-reconstr_weight=0.9-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.5013496058416255,
  'acc': 0.4256666666666667,
  'precision': 0.846526203866924,
  'recall': 0.4318203869372779,
  'new_clusters': 3,
  'cluster F1': 0.1649924206073258,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=0.0001-reconstr_weight=0.1-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.46403014394231307,
  'acc': 0.4086666666666667,
  'precision': 0.7939523900002715,
  'recall': 0.4029991237438047,
  'new_clusters': 3,
  'cluster F1': 0.19621000384081305,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=0.0001-reconstr_weight=0.25-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.45624342073979846,
  'acc': 0.38966666666666666,
  'precision': 0.7939451326968094,
  'recall': 0.3854932825029324,
  'new_clusters': 2,
  'cluster F1': 0.17193625133897922,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=0.0001-reconstr_weight=0.5-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.45837238595175067,
  'acc': 0.395,
  'precision': 0.7935901205527787,
  'recall': 0.4016526528230258,
  'new_clusters': 3,
  'cluster F1': 0.15890740673110526,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=0.0001-reconstr_weight=0.75-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.45394341529998633,
  'acc': 0.4063333333333333,
  'precision': 0.7411557243624525,
  'recall': 0.40096654134078885,
  'new_clusters': 3,
  'cluster F1': 0.21440059958468197,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=0.0001-reconstr_weight=0.9-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4901790785881954,
  'acc': 0.41533333333333333,
  'precision': 0.8459699168682403,
  'recall': 0.4181373210227381,
  'new_clusters': 3,
  'cluster F1': 0.13270330344614537,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=0.001-reconstr_weight=0.1-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4341183213477368,
  'acc': 0.3893333333333333,
  'precision': 0.6883649546188245,
  'recall': 0.38343453787872256,
  'new_clusters': 6,
  'cluster F1': 0.24173562443140562,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=0.001-reconstr_weight=0.25-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4259548442628574,
  'acc': 0.368,
  'precision': 0.793545908567038,
  'recall': 0.36548953889604924,
  'new_clusters': 4,
  'cluster F1': 0.1747156385315326,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=0.001-reconstr_weight=0.5-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4546150015553495,
  'acc': 0.39866666666666667,
  'precision': 0.741105182722471,
  'recall': 0.39655227852480557,
  'new_clusters': 4,
  'cluster F1': 0.20668644832412045,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=0.001-reconstr_weight=0.75-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4292558187827633,
  'acc': 0.37333333333333335,
  'precision': 0.7410886629777377,
  'recall': 0.37446072451010476,
  'new_clusters': 3,
  'cluster F1': 0.2189749743811756,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=0.001-reconstr_weight=0.9-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.46325369446267556,
  'acc': 0.388,
  'precision': 0.7932520887658986,
  'recall': 0.39387728580297304,
  'new_clusters': 5,
  'cluster F1': 0.18205138508143326,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=0.01-reconstr_weight=0.1-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.44299014261736547,
  'acc': 0.3953333333333333,
  'precision': 0.7410344735506955,
  'recall': 0.3887907719074544,
  'new_clusters': 5,
  'cluster F1': 0.22920680417683292,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=0.01-reconstr_weight=0.25-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.44210504744328527,
  'acc': 0.397,
  'precision': 0.7411679884643114,
  'recall': 0.39192288377513834,
  'new_clusters': 6,
  'cluster F1': 0.2513235915631248,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=0.01-reconstr_weight=0.5-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.49910122637905197,
  'acc': 0.422,
  'precision': 0.8468045112781956,
  'recall': 0.42315931667021056,
  'new_clusters': 4,
  'cluster F1': 0.1922939162907344,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=0.01-reconstr_weight=0.75-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.42856560378695036,
  'acc': 0.36933333333333335,
  'precision': 0.7405885848642224,
  'recall': 0.3759541459113332,
  'new_clusters': 3,
  'cluster F1': 0.3066696357268443,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=0.01-reconstr_weight=0.9-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.44179782629856557,
  'acc': 0.4,
  'precision': 0.741359493197188,
  'recall': 0.3953363057008904,
  'new_clusters': 5,
  'cluster F1': 0.29880498832344743,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=0.1-reconstr_weight=0.1-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.44688185404842673,
  'acc': 0.38066666666666665,
  'precision': 0.84630223403978,
  'recall': 0.3800648024243837,
  'new_clusters': 6,
  'cluster F1': 0.20856830373880264,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=0.1-reconstr_weight=0.25-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.45631438742388614,
  'acc': 0.383,
  'precision': 0.8458061275740834,
  'recall': 0.3889638284729305,
  'new_clusters': 4,
  'cluster F1': 0.1998930077827291,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=0.1-reconstr_weight=0.5-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4806391705916705,
  'acc': 0.4073333333333333,
  'precision': 0.8465002712967987,
  'recall': 0.40743763399969246,
  'new_clusters': 3,
  'cluster F1': 0.1499222866359648,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=0.1-reconstr_weight=0.75-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.450366699313075,
  'acc': 0.408,
  'precision': 0.7934970030380162,
  'recall': 0.39704054100329345,
  'new_clusters': 5,
  'cluster F1': 0.23503375664961185,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=0.1-reconstr_weight=0.9-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4376346807131586,
  'acc': 0.39866666666666667,
  'precision': 0.6885964912280701,
  'recall': 0.38917030853011525,
  'new_clusters': 4,
  'cluster F1': 0.287191129251313,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=1.0-reconstr_weight=0.1-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4650715634531797,
  'acc': 0.407,
  'precision': 0.7938664280484828,
  'recall': 0.39914348274672146,
  'new_clusters': 2,
  'cluster F1': 0.22481657506309377,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=1.0-reconstr_weight=0.25-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4279873481038767,
  'acc': 0.371,
  'precision': 0.8468507333908541,
  'recall': 0.3668879144313764,
  'new_clusters': 5,
  'cluster F1': 0.1682746054817698,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=1.0-reconstr_weight=0.5-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.45606443862645396,
  'acc': 0.38366666666666666,
  'precision': 0.8460058508815855,
  'recall': 0.3835605736742857,
  'new_clusters': 4,
  'cluster F1': 0.22434684098191593,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=1.0-reconstr_weight=0.75-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4430810411952046,
  'acc': 0.38366666666666666,
  'precision': 0.898856063388781,
  'recall': 0.3865578107253569,
  'new_clusters': 4,
  'cluster F1': 0.2462201738291996,
  'run_name': 'test-train_size=3000-radius=0-latent_weight=1.0-reconstr_weight=0.9-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.47535604419218724,
  'acc': 0.427,
  'precision': 0.7416798732171157,
  'recall': 0.41932933360715563,
  'new_clusters': 4,
  'cluster F1': 0.23649811447003077,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=1e-05-reconstr_weight=0.1-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.42708495651580985,
  'acc': 0.385,
  'precision': 0.6883621935517776,
  'recall': 0.38755096151055163,
  'new_clusters': 2,
  'cluster F1': 0.2466782543920266,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=1e-05-reconstr_weight=0.25-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.477497449907876,
  'acc': 0.42333333333333334,
  'precision': 0.7944337282998071,
  'recall': 0.4159023285557173,
  'new_clusters': 4,
  'cluster F1': 0.23505271047610193,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=1e-05-reconstr_weight=0.5-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.47780060079026726,
  'acc': 0.4156666666666667,
  'precision': 0.8466330058723452,
  'recall': 0.410298335859453,
  'new_clusters': 4,
  'cluster F1': 0.19536963742192504,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=1e-05-reconstr_weight=0.75-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.45626941757943046,
  'acc': 0.4093333333333333,
  'precision': 0.7410504454282123,
  'recall': 0.413645332100076,
  'new_clusters': 5,
  'cluster F1': 0.3028705623498342,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=1e-05-reconstr_weight=0.9-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4554522568916886,
  'acc': 0.4003333333333333,
  'precision': 0.8464778153334406,
  'recall': 0.3961331797997097,
  'new_clusters': 4,
  'cluster F1': 0.23113577892825155,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=0.0001-reconstr_weight=0.1-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.46689061558204553,
  'acc': 0.421,
  'precision': 0.7947122861586313,
  'recall': 0.4088599966033919,
  'new_clusters': 5,
  'cluster F1': 0.23237078690631502,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=0.0001-reconstr_weight=0.25-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4472735678338789,
  'acc': 0.38433333333333336,
  'precision': 0.7409412197493234,
  'recall': 0.3870278783127736,
  'new_clusters': 4,
  'cluster F1': 0.2543199324457601,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=0.0001-reconstr_weight=0.5-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.42737278570300374,
  'acc': 0.37333333333333335,
  'precision': 0.7408269937380324,
  'recall': 0.3772925232497709,
  'new_clusters': 3,
  'cluster F1': 0.274963832673038,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=0.0001-reconstr_weight=0.75-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4403594317476022,
  'acc': 0.408,
  'precision': 0.7408148168442971,
  'recall': 0.4002794322844886,
  'new_clusters': 3,
  'cluster F1': 0.3045966510810539,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=0.0001-reconstr_weight=0.9-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4617212578058269,
  'acc': 0.3923333333333333,
  'precision': 0.8466188037797603,
  'recall': 0.39255628874927767,
  'new_clusters': 4,
  'cluster F1': 0.19318442379751136,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=0.001-reconstr_weight=0.1-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.47494219871377846,
  'acc': 0.41233333333333333,
  'precision': 0.7938528067438487,
  'recall': 0.4118815673755618,
  'new_clusters': 4,
  'cluster F1': 0.1806448354703442,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=0.001-reconstr_weight=0.25-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4497835983142792,
  'acc': 0.39766666666666667,
  'precision': 0.7411232708104781,
  'recall': 0.3909553509557762,
  'new_clusters': 4,
  'cluster F1': 0.24087343678233972,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=0.001-reconstr_weight=0.5-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.41466208676888927,
  'acc': 0.37533333333333335,
  'precision': 0.7936138280805258,
  'recall': 0.3729535548807793,
  'new_clusters': 3,
  'cluster F1': 0.28302260681197794,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=0.001-reconstr_weight=0.75-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.46267402934003066,
  'acc': 0.4226666666666667,
  'precision': 0.7415210830864641,
  'recall': 0.4175049773085142,
  'new_clusters': 4,
  'cluster F1': 0.30164244569505355,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=0.001-reconstr_weight=0.9-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.43177696064703414,
  'acc': 0.38066666666666665,
  'precision': 0.7404809619238477,
  'recall': 0.38332794583012925,
  'new_clusters': 6,
  'cluster F1': 0.3205035679994407,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=0.01-reconstr_weight=0.1-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.45706409017428645,
  'acc': 0.415,
  'precision': 0.7419101306303865,
  'recall': 0.4020395224561144,
  'new_clusters': 6,
  'cluster F1': 0.2886318896990734,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=0.01-reconstr_weight=0.25-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4518583735458595,
  'acc': 0.39566666666666667,
  'precision': 0.7938884604132869,
  'recall': 0.3895499110641296,
  'new_clusters': 2,
  'cluster F1': 0.1794197221086479,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=0.01-reconstr_weight=0.5-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4115370544739927,
  'acc': 0.3913333333333333,
  'precision': 0.6886448820144645,
  'recall': 0.3747330262194725,
  'new_clusters': 2,
  'cluster F1': 0.32311496687971586,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=0.01-reconstr_weight=0.75-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.431584791009031,
  'acc': 0.37233333333333335,
  'precision': 0.7935718335996701,
  'recall': 0.3736533756805538,
  'new_clusters': 4,
  'cluster F1': 0.24671093083467674,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=0.01-reconstr_weight=0.9-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4558779547134392,
  'acc': 0.39366666666666666,
  'precision': 0.7938021137606883,
  'recall': 0.3990492006064068,
  'new_clusters': 6,
  'cluster F1': 0.24310036190172057,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=0.1-reconstr_weight=0.1-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.44039629576736583,
  'acc': 0.39166666666666666,
  'precision': 0.740691240544225,
  'recall': 0.391718411497697,
  'new_clusters': 5,
  'cluster F1': 0.2489421976225151,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=0.1-reconstr_weight=0.25-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.460179092698165,
  'acc': 0.396,
  'precision': 0.7932735063465115,
  'recall': 0.3980739778874385,
  'new_clusters': 2,
  'cluster F1': 0.21408575417443027,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=0.1-reconstr_weight=0.5-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4833618382407897,
  'acc': 0.404,
  'precision': 0.8992081021386629,
  'recall': 0.4060357601128996,
  'new_clusters': 3,
  'cluster F1': 0.16907894284150443,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=0.1-reconstr_weight=0.75-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4819674863132536,
  'acc': 0.418,
  'precision': 0.7941184554074482,
  'recall': 0.4187019199122403,
  'new_clusters': 5,
  'cluster F1': 0.21516964516957046,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=0.1-reconstr_weight=0.9-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.45793903617265713,
  'acc': 0.3913333333333333,
  'precision': 0.8465396188565698,
  'recall': 0.3891451244579757,
  'new_clusters': 3,
  'cluster F1': 0.20517502254690834,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=1.0-reconstr_weight=0.1-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4784614415430158,
  'acc': 0.42,
  'precision': 0.7941832603486739,
  'recall': 0.4102038079468194,
  'new_clusters': 6,
  'cluster F1': 0.17218102191116075,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=1.0-reconstr_weight=0.25-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4641860965157755,
  'acc': 0.3913333333333333,
  'precision': 0.7932216094661882,
  'recall': 0.40219539188928105,
  'new_clusters': 2,
  'cluster F1': 0.21437862952418563,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=1.0-reconstr_weight=0.5-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.47974237079234155,
  'acc': 0.4146666666666667,
  'precision': 0.7939191043301685,
  'recall': 0.4210966514120152,
  'new_clusters': 6,
  'cluster F1': 0.2556329805212054,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=1.0-reconstr_weight=0.75-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'},
 {'f1': 0.4323530617640801,
  'acc': 0.37666666666666665,
  'precision': 0.8460858027421495,
  'recall': 0.3740184044256973,
  'new_clusters': 2,
  'cluster F1': 0.1731584248664277,
  'run_name': 'test-train_size=3000-radius=6-latent_weight=1.0-reconstr_weight=0.9-head=z-cluster=Kmeans-entity_count=15-noise_factor=0.1-'}]
summarise_scores(scores)

# %%
from grid_search import grid_search

scores = []
num_runs = 1
config = {
    'train_size': [0],
    "radius": [0, 6],
    'latent_weight': [0.0005,0.001, 0.002],
    'reconstr_weight': [0.1, 0.2],
    'head': ['z'],
    'cluster': ['Kmeans'],
    'entity_count': [15],
    'noise_factor': [0.1], 
}

scores = grid_search(config, do_run, sample_size=4000)

# %%
summarise_scores(scores)

# %%
from grid_search import grid_search

scores = []
num_runs = 1
config = {
    'train_size': [0],
    "radius": [6],
    'latent_weight': [0.4, 0.5, 0.6],
    'reconstr_weight': [0.9, 0.2],
    'head': ['z'],
    'epochs': [1],
    'cluster': ['Kmeans'],
    'latent_loss': ['cross_entropy'],
    'entity_count': [15],
    'noise_factor': [0.1], 
}

scores = grid_search(config, do_run, sample_size=4000)

# %%
summarise_scores(scores)

# %%
from grid_search import grid_search

scores = []
num_runs = 1
config = {
    'train_size': [3000],
    "radius": [6],
    'latent_weight': [0.001],
    'reconstr_weight': [1.0],
    'head': ['z'],
    'epochs': [10],
    'cluster': ['Kmeans'],
    'latent_loss': ['cross_entropy'],
    'entity_count': [15],
    'noise_factor': [0.0, 0.1], 
}

scores = grid_search(config, do_run, sample_size=3000)

summarise_scores(scores)


