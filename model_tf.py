# From
# Dahal, P. (2018).
# Learning Embedding Space for Clustering From Deep Representations.
# 2018 IEEE International Conference on Big Data (Big Data), 3747–3755.

import tensorflow as tf
import numpy as np

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import mixture
from scipy.optimize import linear_sum_assignment

import warnings


class ClusterNetwork(object):

    def __init__(self,
                 latent_dim=10,
                 latent_weight=0.01,
                 noise_factor=0.5,
                 keep_prob=1.0,
                 alpha1=20,
                 alpha2=1,
                 learning_rate=0.001,
                 initializer='xavier',
                 optimizer='adam',
                 rec_loss='binary_crossentropy',
                 n_clusters=10,
                 l1_reg=0.0,
                 clustering='Kmeans',
                 decay_rate=1,
                 decay_interval=10,
                 rec_weight=1.0,
                 ):

        # Make hyperparameters instance variables.
        self.latent_dim = latent_dim
        self.latent_weight = latent_weight
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.keep_prob = keep_prob
        self.noise_factor = noise_factor
        self.learning_rate = learning_rate
        self.initializer = initializer
        self.optimizer = optimizer
        self.rec_loss = rec_loss
        self.n_clusters = n_clusters
        self.l1_reg = l1_reg
        self.clustering = clustering
        self.decay_rate = decay_rate
        self.decay_interval = decay_interval
        self.rec_weight = rec_weight
        initializers = {
            'xavier': tf.keras.initializers.GlorotUniform(),
            'uniform': tf.random_uniform_initializer(-1, 1)
        }
        self.initializer = initializers[self.initializer]

        # Random seed for Numpy and Tensorflow.
        self.random_seed = 42
        tf.random.set_seed(self.random_seed)

        tf.compat.v1.disable_eager_execution()

        # silence sklearn warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    def encoder(self, X, is_training):

        with tf.compat.v1.variable_scope('encoder', initializer=self.initializer):

            h1 = tf.keras.layers.Dense(500)(X)
            h2 = tf.keras.layers.Dense(500)(h1)
            h3 = tf.keras.layers.Dense(2000)(h2)

            z = tf.keras.layers.Dense(self.latent_dim)(h3)

            return z

    def decoder(self, z, training):

        with tf.compat.v1.variable_scope('decoder', initializer=self.initializer):

            h1 = tf.keras.layers.Dense(2000)(z)
            h2 = tf.keras.layers.Dense(500)(h1)
            h3 = tf.keras.layers.Dense(500)(h2)

            X_out_logits = tf.keras.layers.Dense(self.out_dim)(h3)
            X_out = tf.nn.sigmoid(X_out_logits)

            return X_out_logits, X_out

    def latent_network(self, z, training):

        with tf.compat.v1.variable_scope('latent_network', initializer=tf.initializers.identity):

            h1 = tf.keras.layers.Dense(2000)(z)
            h2 = tf.keras.layers.Dense(500)(h1)
            h3 = tf.keras.layers.Dense(500)(h2)
            h4 = tf.keras.layers.Dense(500)(h3)
            h = tf.keras.layers.Dense(self.latent_dim)(h4)

            return h

    def pairwise_sqd_distance(self, X, batch_size):

        tiled = tf.tile(tf.expand_dims(X, axis=1),
                        tf.stack([1, batch_size, 1]))
        tiled_trans = tf.transpose(tiled, perm=[1, 0, 2])
        diffs = tiled - tiled_trans
        sqd_dist_mat = tf.reduce_sum(tf.square(diffs), axis=2)
        tf.print(sqd_dist_mat, [sqd_dist_mat])

        return sqd_dist_mat

    def make_q(self, z, batch_size, alpha):

        sqd_dist_mat = self.pairwise_sqd_distance(z, batch_size)
        q = tf.pow((1 + sqd_dist_mat/alpha), -(alpha+1)/2)
        q = tf.linalg.set_diag(q, tf.zeros(shape=[batch_size]))
        q = q / tf.reduce_sum(q, axis=0, keepdims=True)
        #q = 0.5*(q + tf.transpose(q))
        q = tf.clip_by_value(q, 1e-10, 1.0)

        return q

    def train(self, train_X, train_y, train_batch_size, pretrain_epochs=10, train_epochs=100):

        # Reset Tensorflow graph and set random seed.
        tf.compat.v1.reset_default_graph()
        tf.random.set_seed(self.random_seed)

        # Placeholders.
        self.out_dim = train_X.shape[1]
        X = tf.compat.v1.placeholder(shape=[None, self.out_dim], dtype=tf.float32)
        is_training = tf.compat.v1.placeholder(dtype=tf.bool)

        # Global step variable.
        self.global_step = tf.Variable(
            0, name='global_step', trainable=False, dtype=tf.int32)
        increment_global_step_op = tf.compat.v1.assign(
            self.global_step, self.global_step+1)

        # Decaying learning rate.
        self.learning_rate = tf.Variable(
            self.learning_rate, name='learning_rate', trainable=False, dtype=tf.float32)
        # Check and set optimizer.
        optimizers = {
            'sgd': tf.compat.v1.train.GradientDescentOptimizer(self.learning_rate),
            'adam': tf.compat.v1.train.AdamOptimizer(self.learning_rate),
            'sgd_mom': tf.compat.v1.train.MomentumOptimizer(self.learning_rate, momentum=0.9, use_nesterov=True),
            'rmsprop': tf.compat.v1.train.RMSPropOptimizer(self.learning_rate),
            'adagrad': tf.compat.v1.train.AdagradOptimizer(self.learning_rate)
        }

        if self.optimizer not in optimizers.keys():
            raise ValueError("optimizer should be in {}".format(
                optimizers.keys()))
        else:
            self.optimizer = optimizers[self.optimizer]

        # Autoencoder model.
        ae_loss, ae_optimize, z_op = self.ae_model(
            X, is_training, step=self.global_step)

        # Latent model.
        latent_loss, latent_optimize, latent_op = self.latent_model(
            z_op, ae_loss, is_training, step=self.global_step)

        # GPU config.
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        # List for storing ACCs.
        accs = []

        # Initialize session.
        with tf.compat.v1.Session(config=config) as sess:

            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())

            # Pretrain the autoencoder model with only reconstruction loss.
            for step in range(pretrain_epochs):

                _, self.globalstep = sess.run(
                    [increment_global_step_op, self.global_step])

                num_train_batches, start = int(
                    train_X.shape[0]/train_batch_size), 0

                for _ in range(num_train_batches):

                    end = start + train_batch_size
                    limit = end if end < train_X.shape[0] else train_X.shape[0]
                    idx = np.arange(start, limit)
                    _, rec_loss = sess.run([ae_optimize, ae_loss], {
                                           X: train_X[idx], is_training: True})
                    start = end

                z_test = sess.run(z_op, {X: train_X, is_training: False})
                print('Epoch: {0}\nReconstruction Loss: {1}'.format(
                    self.globalstep, rec_loss))
                accuracy = self.metrics(train_y, z_test)
                accs.append(accuracy)

            z, zl = sess.run([z_op, latent_op], {
                             X: train_X, is_training: False})
            np.save('z_state_0', z)
            np.save('zl_state_0', zl)

            # Train the AE and Representation network model together.
            print('Starting joint training...')

            for step in range(train_epochs):
                _, self.globalstep = sess.run(
                    [increment_global_step_op, self.global_step])

                num_train_batches, start = int(
                    train_X.shape[0]/train_batch_size), 0
                for _ in range(num_train_batches):

                    end = start + train_batch_size
                    limit = end if end < train_X.shape[0] else train_X.shape[0]
                    idx = np.arange(start, limit)
                    _, z_loss, rec_loss = sess.run([latent_optimize, latent_loss, ae_loss], {
                                                   X: train_X[idx], is_training: True})
                    start = end

                # Save latent space Z and embedded space E every two epochs.
                if step != 0 and step % 2 == 0:
                    zl, zz = sess.run([latent_op, z_op], {
                                      X: train_X, is_training: False})
                    np.save('zl_state_{}'.format(step), zl)
                    np.save('z_state_{}'.format(step), zz)

                zl = sess.run(latent_op, {X: train_X, is_training: False})
                print('Epoch: {0}\nLatent Loss: {1}\nReconstruction Loss: {2}'.format(
                    self.globalstep, z_loss, rec_loss))
                accuracy = self.metrics(train_y, zl)
                accs.append(accuracy)
                if accuracy >= max(accs):
                    print('found best acc...')
                    zl = sess.run(latent_op, {X: train_X, is_training: False})
                    np.save('zl_state_best', zl)
            return accs

    def metrics(self, y, z_state):
        """
        y is labels, not one hot encoded vector.
        """
        if self.clustering == 'GMM':
            gmix = mixture.GaussianMixture(
                n_components=self.n_clusters, covariance_type='full')
            gmix.fit(z_state)
            y_pred = gmix.predict(z_state)
        elif self.clustering == 'Kmeans':
            # silence sklearn warnings
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
            y_pred = kmeans.fit_predict(z_state)
        else:
            raise ValueError('Clustering algorithm not specified/unknown.')

        def cluster_acc(y_true, y_pred):
            y_true = y_true.astype(np.int64)
            assert y_pred.size == y_true.size
            D = max(y_pred.max(), y_true.max()) + 1
            w = np.zeros((D, D), dtype=np.int64)
            for i in range(y_pred.size):
                w[y_pred[i], y_true[i]] += 1

            ind = linear_sum_assignment(w.max() - w)
            ind = np.asarray(ind)
            ind = np.transpose(ind)
            return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

        acc = np.round(cluster_acc(y, y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)

        print('-'*40)
        print('Clustering Accuracy: {}'.format(acc))
        print('NMI: {}'.format(nmi))
        print('Adjusted Rand Index: {}'.format(ari))
        print('-'*40)

        return acc

    def ae_model(self, X, training, step):

        batch_size = tf.shape(X)[0]

        # Add noise to the input to feed to Denoising encoder model.
        X_noisy = X + self.noise_factor * \
            tf.random.normal(shape=tf.shape(X), mean=0.0,
                             stddev=1.0, dtype=tf.float32)
        X_noisy = tf.clip_by_value(X, 0.0, 1.0)

        # Pass through encoder and decoder.
        z = self.encoder(X_noisy, training)
        X_out_logits, X_out = self.decoder(z, training)

        # Calculate Reconstruction loss.
        if self.rec_loss == 'mse':
            reconstr_loss = tf.reduce_mean(
                tf.math.squared_difference(X, X_out), axis=1)
        else:
            reconstr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=X, logits=X_out_logits), axis=1)
        reconstr_loss = tf.reduce_mean(reconstr_loss)

        # Total Reconstruction Loss.
        reconstr_loss = reconstr_loss

        # Perform backprop wrt reconstruction loss and update encoder and decoder variables.
        grads = self.optimizer.compute_gradients(reconstr_loss)
        vars_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='encoder') + \
            tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
        grad_list = [(g, v) for g, v in grads if v in vars_list]
        optimize_op = self.optimizer.apply_gradients(grad_list)

        return reconstr_loss, optimize_op, z

    def latent_model(self, z_enc, rec_loss, training, step):

        batch_size = tf.shape(z_enc)[0]

        # Pass representation vectos through latent network.
        z = self.latent_network(z_enc, training)

        # Calculate probabilty distributions for input and output of representation network.
        p = self.make_q(z_enc, batch_size, alpha=self.alpha1)
        q = self.make_q(z, batch_size, alpha=self.alpha2)

        # Cross entropy loss.
        # latent_loss = tf.reduce_sum(tf.multiply(p, tf.log(p) - tf.log(q))) # KL Divergence.
        latent_loss = tf.reduce_sum(-(tf.multiply(p, tf.compat.v1.log(q))))

        # Joint loss.
        joint_loss = tf.constant(self.rec_weight)*rec_loss + \
            tf.constant(self.latent_weight)*latent_loss

        if step != 0 and step % self.decay_interval == 0:
            self.learning_rate = tf.multiply(
                tf.constant(self.decay_rate), self.learning_rate)

        # Calculate gradients of variables w.r.t latent, reconstruction and joint loss.
        grads_joint = self.optimizer.compute_gradients(joint_loss)
        grads_latent = self.optimizer.compute_gradients(latent_loss)
        grads_rec = self.optimizer.compute_gradients(rec_loss)

        # Update latent network weights with latent loss.
        grad_list = [(g, v) for g, v in grads_latent if v in tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='latent_network')]
        optimize_op_latent = self.optimizer.apply_gradients(grad_list)

        # Update encoder weights with joint loss.
        grad_list = [(g, v) for g, v in grads_joint if v in tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')]
        optimize_op_enc = self.optimizer.apply_gradients(grad_list)

        # Update decoder weights with reconstruction loss.
        grad_list = [(g, v) for g, v in grads_rec if v in tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')]
        optimize_op_dec = self.optimizer.apply_gradients(grad_list)

        optimize_op = tf.group(
            optimize_op_latent, optimize_op_enc, optimize_op_dec)

        return latent_loss, optimize_op, z
