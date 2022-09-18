# %% [markdown]
# # Tuning

# %%
from deep_cluster import DeepCluster

# %%
dc = DeepCluster("summary", verbose=1)
dc.make_model()
dc.model.summary()

# %%
stop

# %% [markdown]
# # Evaluate

# %%
from deep_cluster import DeepCluster
from grid_search import grid_search, make_name
from cluster_metrics import summarise_scores

# %%
def do_run(cfg, idx, total, sample_size, verbose):
    cfg['sample_size'] = sample_size
    run_name = ("test-dc-" + make_name(cfg))
    print('-'*50)
    print(f"{idx+1}/{total}: {run_name}")
    print('-'*50)
    print(cfg)
    dc = DeepCluster(run_name, verbose=verbose, config=cfg)
    dc.make_model()
    dc.train_model()
    score = dc.evaluate_model(
        eval_size=2000,
        verbose=verbose)
    score['run_name'] = run_name
    return score


# %%
num_runs = 1
config = {
    'train_size': 0,
    "radius": 6,
    'maxiter': 2000,
    'kl_weight': [0.1, 0.5, 1.0],
    'mse_weight': [0.1, 0.5, 1.0],
    'acc_weight': [0.1, 0.5, 1.0],
}
scores = grid_search(config, do_run, 1, verbose=0)
summarise_scores(scores)

# %%
from deep_cluster import DeepCluster
from grid_search import grid_search, make_name
from cluster_metrics import summarise_scores

# %%
def do_eval_run(cfg, idx, total, sample_size, verbose):
    cfg['sample_size'] = sample_size
    save_name = ("test-dc-noUNK-" + make_name(cfg))
    run_name = ("test-dc-" + make_name(cfg))
    print('-'*50)
    print(f"{idx+1}/{total}: {run_name}")
    print('-'*50)
    print(cfg)
    dc = DeepCluster(run_name, verbose=verbose, config=cfg)
    dc.make_model()
    score = dc.evaluate_model(
        eval_size=2000,
        output=run_name,
        include_unk=True,
        verbose=verbose)
    score['run_name'] = save_name
    return score


# %%
# with unknowns
num_runs = 1
config = {
    'train_size': 0,
    "radius": 6,
    'maxiter': 2000,
    'kl_weight': [0.1, 0.5, 1.0],
    'mse_weight': [0.1, 0.5, 1.0],
    'acc_weight': [0.1, 0.5, 1.0],
}
scores = grid_search(config, do_eval_run, 1, verbose=0)
summarise_scores(scores)

# %%
# %history -g -f jupyter_history.py


# %%
del dc

# %%
dc = DeepCluster('test-0-40latent',
                 dims=[768, 500, 500, 2000, 40],
                 entity_count=10,
                 train_size=1000,
                 num_clusters=25,
                 maxiter=2000,
                 loss_weights=[1.0, 0.1, 0.1],
                 verbose=2)


# %%
dc.make_model()
print(dc.autoencoder.summary())


# %%
dc.train_and_evaluate_model(1000, verbose=1)

# %%
dc = DeepCluster(
        'test-0-40latent-2',
        config={
            'entity_count':10,
            'train_size':0,
            'num_clusters':25,
            'maxiter':2000,
        }
        )
dc.train_and_evaluate_model(10000, verbose=1)


# %%
dc = DeepCluster(
        'test-0-40latent',
        dims=[768, 500, 500, 2000, 40],
        entity_count=10,
        train_size=0,
        num_clusters=25,
        maxiter=2000)
dc.train_and_evaluate_model(10000, verbose=1)

# %%
dc = DeepCluster('test-0-40latent', dims=[768, 500, 500, 2000, 40],
    entity_count=10, train_size=0, num_clusters=25, maxiter=2000)
dc.evaluate_model(10000, verbose=0)
print("UMAP")
dc.visualise_umap(5000, embs="x")
dc.visualise_umap(5000, embs="z")

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
run_benchmark('Kmeans', 10000, 25)

# %%
run_benchmark('GMM', 10000, 25)

# %%
run_benchmark('agg', 10000, 25)

# %%
dc = DeepCluster('test1', train_size=0, num_clusters=25).train_and_evaluate_model(10000)


