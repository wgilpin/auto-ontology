# %%
from deep_cluster import DeepCluster
from deep_latent import DeepLatentCluster

# %%
stop
# to prevent the notebook from running the rest of the cell

# %% [markdown]
# ## Baseline

# %%
dc = DeepCluster(
        'test-0-40latent',
        dims=[768, 500, 500, 2000, 40],
        entity_count=10,
        train_size=0,
        num_clusters=25,
        maxiter=2000)
dc.make_load_model()
dc.evaluate_model(10000, verbose=0)

# %% [markdown]
# ## PDFs on default pre-trained model

# %%
dc = DeepCluster(
        'test-0-40latent',
        dims=[768, 500, 500, 2000, 40],
        entity_count=10,
        train_size=0,
        num_clusters=25,
        maxiter=2000)
dc.make_load_model()
dc.evaluate_model(10000, verbose=1, folder="pdfs/lines.txt", output="test-pdf-untrained")

# %% [markdown]
# ## PDFs retrained

# %%
dc = DeepCluster(
        'test-pdfs-dec',
        dims=[768, 500, 500, 2000, 40],
        entity_count=10,
        train_size=0,
        num_clusters=25,
        maxiter=2000)
dc.train_and_evaluate_model(10000, verbose=1, folder="pdfs/lines.txt")

# %% [markdown]
# ## PDFs via latent model

# %%
%%time
# benchmark with cluster rearrangement
from deep_latent import DeepLatentCluster
from cluster_metrics import summarise_scores

dc = None
dc = DeepLatentCluster(
    'test-latent-15-ents-lw0.5-rw1.0-z-Kmeans',
    {
        'train_size':0,
        "cluster": "Kmeans"
    })
score = dc.evaluate_model(
        'test-latent-15-ents-lw0.5-rw1.0-z-Kmeans', 
        sample_size=10000,
        verbose=0,
        folder="pdfs/lines.txt",
        output="test-pdf-latent")

summarise_scores([{**score, "run_name": "latent-bm"}])


