import pytest

from tensorflow.keras.layers import Layer

from deep_cluster import DeepCluster


@pytest.fixture(scope='module')
def dc():
    dc = DeepCluster('test', dims=[768, 500, 500, 2000, 40],
            entity_count=10, train_size=1000, num_clusters=25, maxiter=200)
    dc.verbose = 0
    yield dc


def test_init_dc(dc):
    # init ok
    assert dc is not None
    yield

def test_load_data_to_model(dc):
    # load data ok
    dc.make_data()
    n_samples = dc.x.shape[0]
    assert n_samples > 0
    assert dc.x.shape == (n_samples, 768)

def test_load_make_model(dc):
    # make model ok
    dc.autoencoder = None
    dc.encoder = None
    dc.model = None
    dc.make_model()
    assert dc.autoencoder is not None
    assert dc.encoder is not None
    assert dc.model is not None

def test_cluster_init(dc):
    # cluster init ok
    dc.y_pred_last = None
    dc.init_cluster_centers()
    n_samples = dc.x.shape[0]
    assert dc.y_pred_last.shape == (n_samples, )

def test_train_model(dc):
    dc.train_model()

def test_predict(dc):
    # cluster prediction accuracy ok
    dc.y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dc.y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert dc.cluster_pred_acc() == 1.0
    dc.y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert dc.cluster_pred_acc() == 0.9
