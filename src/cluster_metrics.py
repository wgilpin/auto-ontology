import os
import glob
from collections import defaultdict
from typing import Optional
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame, crosstab
from wordcloud import WordCloud
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score)
from jinja2 import Environment, FileSystemLoader
import sklearn.metrics
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering
from sklearn import mixture
from scipy.stats import multivariate_normal

from linear_assignment import linear_assignment
from cluster import Cluster, ClusterList

def output(verbose, s: str) -> None:
    """
    Print output if verbose is set.
    """
    if verbose > 0:
        print(s)

def cluster_acc(y_true, y_pred, y_pred_cluster):
    """
    Calculate clustering accuracy
    """
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
    
def cluster_loss(clustering:str, n_clusters: int):
    """
    Kearas loss function wrapper
    """
    def loss(y_true, y_pred):
        y_pred_cluster, _ = do_clustering(clustering, n_clusters, y_pred)
        return cluster_acc(y_true, y_pred, y_pred_cluster)
    return loss

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. 

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def plot_confusion(
        y,
        y_pred,
        mapping,
        save_dir: str,
        size: int=8,
        details: bool=True,
        new_labels: Optional[dict] = None):
    """
    Plot a confusion matrix.
    """

    sns.set(font_scale=3)
    confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred)
    entity_label_tups = [(k,v) for k,v in mapping.items()]
    entity_labels = [v for v,v in sorted(entity_label_tups, key=lambda tup: tup[0])]
    new_ent_labels = [v for _, v in sorted(new_labels.items())]

    # re-order confusion matrix into a diagonal

    for y_hat in range(len(confusion_matrix)):
        max = np.argmax(confusion_matrix[y_hat])
        confusion_matrix.T[[y_hat, max]] = confusion_matrix.T[[max, y_hat]]
    plt.figure(figsize=(size, size))
    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5})

    sns.heatmap(
        confusion_matrix,
        annot=details,
        fmt="d",
        cmap=sns.color_palette("crest", as_cmap=True),
        cbar=False,
        annot_kws={"size": 20},
        yticklabels=entity_labels, # type: ignore
        xticklabels=new_ent_labels, # type: ignore
        )
    plt.title("Confusion matrix\n", fontsize=20)
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Cluster label', fontsize=20)
    if save_dir:
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.show()

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

def get_freqs(word_list):
    """
    Get the frequencies of words in a list
    """
    unique, counts = np.unique(word_list, return_counts=True)
    freq_list = np.asarray((unique, counts)).T
    freq_list = sorted(freq_list, key=lambda x: -x[1])[0:50]
    freqs = {w: f for w, f in freq_list}
    return freqs

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


def rearrange_clusters(
                    sample,
                    mapping,
                    do_rearrange:bool=True,
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

    for clus_no in np.unique(sample['y_pred']): 
        if clus_no < 0:
            continue
        cluster = sample[sample['y_pred'] == clus_no]
        prob_ent = int(np.argmax(y_tru_per_clus[clus_no]))
        if mapping[prob_ent] == 'UNKNOWN':
            # get next best
            prob_ent = list(np.argsort(y_tru_per_clus[clus_no]))[-2]
        prob_lbl = mapping[prob_ent]
        frac = y_tru_frac_by_clus[clus_no][prob_ent]

        # wordcloud
        freqs = get_freqs(cluster['shorts'].values)
        unknown_cluster = cluster[cluster['y_true'] == 0]
        freqs_unknown = get_freqs(unknown_cluster['shorts'].values)
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

    for k, v in clusters.items():
        v.name = k

    return sample, clusters

def show_wordcloud(
        freqs: dict,
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

def calc_metrics(true_pos, false_pos, false_neg):
    """
    Calculate f1, precision and recall
    """
    if true_pos + false_pos == 0:
        precision = 0
    else:
        precision = true_pos / (true_pos + false_pos)
    if true_pos + false_neg == 0:
        recall = 0
    else:
        recall = true_pos / (true_pos + false_neg)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall

def eval_cluster(
                sample: DataFrame,
                mapping: dict,
                rearrange: bool=True):
    """
    show wordclouds for each cluster
    """

    sample, clusters = rearrange_clusters(
                                    sample,
                                    mapping,
                                    rearrange)

    # confusion
    f1_list = []
    size_list = []
    cluster_scores = {}
    for cluster_name, ce in clusters.items():
        c = sample[sample.y_pred == ce.clus_no]
        c_ent = ce.entity_id

        # the right entity class in the right cluster
        true_pos = c[
            # in this cluster
            (c.y_pred_new == c_ent) &\
            # and this is the right class
            (c.y_true == c_ent)].shape[0]

        # this cluster, we think it's right entity but not the right entity
        false_pos = c[
            # in this cluster
            (c.y_pred_new == c_ent) &\
            # but not the right entity class
            (c.y_true != c_ent)].shape[0]

        # it's the right entity in wrong cluster
        false_neg = sample[
            # not in this cluster
            (sample.y_pred_new != c_ent) &\
            # but should be
            (sample.y_true == c_ent)].shape[0]

        f1, prec, rec = calc_metrics(true_pos, false_pos, false_neg)

        cluster_scores[cluster_name] = {
            'F1': f1,
            'precision': prec,
            'recall': rec,
            'true_pos': true_pos,
            'false_pos': false_pos,
            'false_neg': false_neg,
        }

        if cluster_name[0:3] == 'UNK':
            f1_list.append(f1)
            size_list.append(ce.n/sample.shape[0])

        print(f"#{cluster_name}:{ce.clus_no} size:{len(c)} "
              f"prec:{prec:.4f} rec:{rec:.4f} f1:{f1:.4f}")

    # full cluster
    f1 = np.dot(f1_list, size_list)
    print(f"\nF1 by Known Clusters: {f1:.4f}")

    return sample, clusters, f1

def show_core_metrics(y_pred_sample, all_clusters, new_labels, mapping, y_sample, save_dir):
    """
    show the core metrics for the clustering
    """
    assert mapping is not None
    assert y_sample is not None

    # confusion matrix
    num_predicted = len(np.unique(y_pred_sample))
    cm_width = max(8, num_predicted * 2)
    cm_width = min(16, cm_width)
    plot_confusion(y_sample,
                    y_pred_sample,
                    mapping,
                    save_dir,
                    size=cm_width,
                    new_labels=new_labels)

    # how big are the predicted clusters
    cluster_counts = np.unique(y_pred_sample, return_counts=True)[1]
    # how big are the new ones?

    num_new = cluster_counts[len(mapping):]
    # how many of the new ones are more than 2.5% of the sample?
    num_new_large = len(num_new[num_new > (len(y_sample) * 0.025)])

    # metrics
    y = all_clusters['y_true']
    y_pred = all_clusters['y_pred_new']
    f1 = f1_score(y, y_pred, average='macro')
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(
        y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')
    print(f"F1 score (macro) = {f1:.4f}")
    print(f"Accuracy = {accuracy:.4f}")
    print(f"Precision = {precision:.4f}")
    print(f"Recall = {recall:.4f}")
    print(f"New clusters = {num_new_large:6d}")
    scores = {
        'f1': f1,
        'acc': accuracy,
        'precision': precision,
        'recall': recall,
        'new_clusters': num_new_large,
    }
    return scores

def score_clusters(clusters: ClusterList, save_dir: str, mapping: dict) -> ClusterList:
    """
    score the clusters found
    """
    cluster_list = sorted(clusters.values(), key=lambda x: -x.frac)

    # show unknown clusters first
    cluster_list = sorted(
        cluster_list,
        key=lambda x: int(x.name[0:3] != "UNK"))

    # delete old wordcloud files
    for f in glob.glob(f"{save_dir}/wordcloud*.png"):
        os.remove(f)

    for cluster in cluster_list:
        save_file = os.path.join(save_dir,
                                    f"wordcloud-{cluster.name}.png")
        show_wordcloud(
            cluster.freqs,
            cluster.name,
            save_file,
            save_only=True)

        # the top 3 entity classes in this cluster
        top_entities: list[dict] = []
        for (entity, count) in cluster.class_freqs[0:3]:
            top_entities += [
                {'class': mapping[entity],
                    'count':count}]
        cluster.classes = top_entities

    # save clusters of NER unknowns only
    for cluster in cluster_list:
        save_file = os.path.join(save_dir,
                                    f"wordcloud-{cluster.name}-new.png")
        if len(cluster.freqs_unknown) > 0:
            show_wordcloud(
                cluster.freqs_unknown,
                cluster.name,
                save_file,
                save_only=True)

    return cluster_list  # type: ignore


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


def save_scores(cluster_list, scores, save_dir, run_name):
    """
    save the scores to file
    """
    new_clusters = [c for c in cluster_list if len(c.freqs_unknown) > 4]
    big_clusters = [c for c in cluster_list if len(c.freqs) > 5]
    write_results_page(
        big_clusters,
        new_clusters,
        save_dir,
        run_name,
        scores,
    )

def do_clustering(
        clustering: str,
        n_clusters: int,
        z_state: DataFrame,
        params: Optional[dict] = None) -> tuple[np.ndarray, np.ndarray]:
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
    if params is  None:
        params = {}
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
            verbose=params['verbose'] if 'verbose' in params else 0)
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

    return y_pred, centers  # type: ignore

def do_evaluation(
            sample: DataFrame,
            mapping: dict,
            verbose: int,
            save_dir:str,
            run_name:str) -> dict:
    """
    Evaluate the model.
    - z_sample: the latent space
    - y_sample: the true labels
    - str_sample: the original strings
    """
    assert 'text' in sample.columns
    assert 'y_true' in sample.columns
    assert 'y_pred' in sample.columns
    assert 'shorts' in sample.columns

    if verbose>0:
        print("CLUSTERS")

    # assign labels to the clusters
    assert mapping is not None
    all_clusters, clusters, cluster_f1 = eval_cluster(
                                            sample, mapping)

    # list of new clusters by id number
    labels_no = {v.clus_no: k for k,v in clusters.items()}
    labels_ent = {v.entity_id: k for k,v in clusters.items()}
    new_labels = {**labels_no, **labels_ent}

    # overall scores
    scores_agg = show_core_metrics(
        sample.y_pred,
        all_clusters,
        new_labels, mapping,
        sample.y_true,
        save_dir)

    # cluster scores
    cluster_list = score_clusters(
                            clusters=clusters,
                            save_dir=save_dir,
                            mapping=mapping)

    # output file
    save_scores(cluster_list, scores_agg, save_dir, run_name)

    scores = {**scores_agg, 'cluster F1': cluster_f1} # type: ignore

    return scores
