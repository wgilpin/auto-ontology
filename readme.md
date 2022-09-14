# Finding semantic clusters

For detailed descriptions and full references see 
https://github.com/wgilpin/project/blob/master/ProjectReport.pdf

To process a set of pdfs:

Makes a folder `pdfs` under the top-level project folder.

Place pdfs in that folder.

run the `process_pdfs.py` script from the project folder:

```
> python process_pdfs.py
```

`process_pdfs.py` can take an optional parameter with a folder name, which needs to exist under the project folder.

You will now have a file, `lines.txt` in the chosen folder. Copy / move it to the folder `./src/pdfs`, relative to the project folder.

Next, as exemplified in the file `pdfs_notebook.ipynb`, run the following code:

```python
from deep_cluster import DeepCluster

dc = DeepCluster(
        'test-pdfs-dec',
        dims=[768, 500, 500, 2000, 40],
        entity_count=10,
        train_size=0,
        num_clusters=25,
        maxiter=2000)
dc.train_and_evaluate_model(10000, verbose=1, folder="pdfs/lines.txt")
```

or to use the latent representation network:

```python
from deep_latent import DeepLatentCluster

dc = DeepLatentCluster(
    run_name='test-latent-all-Kmeans',
    config={
        'train_size':0,
        'reconstr_weight':1.0,
        'latent_weight':1e-5,
        "cluster": "Kmeans"
    })
dc.make_model()
dc.train_model()
dc.evaluate_model('test-latent-all', sample_size=4000)
```

`DeepLatentCluster` has many configuration options that can be supplied to the `config` param in dictionary form. For a comprehensive list, see the class init method, by common ones include:

* `train_size`: how many samples to include
* `reconstr_weight`: weight applied to loss from the autoencoder reconstruction
* `latent_weight`: weight applied to loss from the latent representation network
* `cluster`: algorithm to use:
  * `Kmeans` for k-means
  * `GMM` for Gaussian Mixture Methods
  * `OPTICS`
  * `agg` for agglomerative clustering
* `opt`: the optimizer function
* `noise_factor`: [0.0-1.0] How much gaussian noise to add to training data
* `entity_count`: how many spaCy entity classes
* `num_clusters`: target number of identified cluster, including the spaCy entities
* `max_iter`: how many training iterations per epoch
* `epochs`: how many training epochs
* `tolerance`: [float] the fraction of change of cluster allocations needed to continue training

Results will be stored in `./results/[run_name]` for the run name supplied to the class inititialiser.
