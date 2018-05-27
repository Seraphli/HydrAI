import tensorflow as tf
from tfac.queue_input import QueueInput

class ModeKeys(object):
    TRAIN = 'train'
    PREDICT = 'predict'


class NN(object):
    def __init__(self, feature_size, action_size, sample_fn):
        self.feature_size = feature_size
        self.action_size = action_size
        self.sample_fn = sample_fn
        self.build_input()
        idx, batch_features, batch_labels = self.qi.build_op(32)
        idx, pred_features, pred_labels = self.qi.build_op(1)
        self.train_net = self.build_net(batch_features, batch_labels,
                                        ModeKeys.TRAIN, False, "mcts_")
        self.pred_net = self.build_net(pred_features, pred_labels,
                                       ModeKeys.PREDICT, tf.AUTO_REUSE, "mcts_")
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(max_to_keep=10)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.sess.graph.finalize()
        self.is_qi_run = False

    def run(self):
        self.qi.run(self.sess, self.sample_fn)

    def train(self):
        if not self.is_qi_run:
            self.run()
        self.sess.run(self.train_net["train_op"])

    def predict(self):
        return self.sess.run(self.pred_net["predictions"])

    def build_input(self):
        self.features = {
            "s": tf.placeholder(tf.float32, [None, *self.feature_size])
        }

        self.labels = {
            "pi": tf.placeholder(tf.float32, [None, self.action_size]),
            "z": tf.placeholder(tf.float32, [None, 1]),
        }
        self.qi = QueueInput(self.features, self.labels, [400, 20])

    def build_net(self, features, labels, mode, reuse, prefix):
        training = mode == ModeKeys.TRAIN
        with tf.variable_scope(prefix, reuse=reuse):
            w_init = tf.contrib.layers.xavier_initializer()
            b_init = tf.constant_initializer(0.1)
            x = features["s"]
            x = tf.layers.conv2d(
                x, 32, 8, 4, activation=tf.nn.relu, name="conv_1",
                kernel_initializer=w_init, bias_initializer=b_init)
            x = tf.layers.batch_normalization(
                x, training=training,
                momentum=0.997, epsilon=1e-5, name="bn_1")
            x = tf.layers.conv2d(
                x, 64, 4, 2, activation=tf.nn.relu, name="conv_2",
                kernel_initializer=w_init, bias_initializer=b_init)
            x = tf.layers.batch_normalization(
                x, training=training,
                momentum=0.997, epsilon=1e-5, name="bn_2")
            x = tf.layers.conv2d(
                x, 64, 3, 1, activation=tf.nn.relu, name="conv_3",
                kernel_initializer=w_init, bias_initializer=b_init)
            x = tf.layers.batch_normalization(
                x, training=training,
                momentum=0.997, epsilon=1e-5, name="bn_3")
            pi = tf.layers.flatten(x)
            pi = tf.layers.dense(pi, 512, activation=tf.nn.relu,
                                 name="pi_dense_1")
            pi = tf.layers.batch_normalization(
                pi, training=training,
                momentum=0.997, epsilon=1e-5, name="bn_4")
            pi = tf.layers.dense(pi, self.action_size,
                                 name="pi_dense_2")
            v = tf.layers.flatten(x)
            v = tf.layers.dense(v, 512, activation=tf.nn.relu,
                                name="v_dense_1")
            v = tf.layers.batch_normalization(
                v, training=training,
                momentum=0.997, epsilon=1e-5, name="bn_5")
            v = tf.layers.dense(v, 1, name="v_dense_2")
            v = tf.nn.sigmoid(v)

            predictions = {
                "pi": tf.nn.softmax(pi, name="pi"),
                "v": tf.identity(v, name="v")
            }

            if mode == ModeKeys.PREDICT:
                net = {"predictions": predictions}
                return net

            with tf.name_scope("loss"):
                epsilon = 1e-10
                mse_loss = tf.losses.mean_squared_error(labels["z"], v)
                cross_entropy = (labels["pi"] + epsilon) * tf.log(pi + epsilon)
                ce_loss = tf.reduce_mean(-tf.reduce_sum(cross_entropy, 1))
                l2_loss = 1e-4 * tf.add_n([tf.nn.l2_loss(v)
                                           for v in tf.trainable_variables()])
                loss = mse_loss + ce_loss + l2_loss

            optimizer = tf.train.AdamOptimizer()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss)

            net = {
                "predictions": predictions,
                "loss": loss,
                "train_op": train_op
            }
            return net

    def load(self, load_path):
        checkpoint = tf.train.get_checkpoint_state(load_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            return True
        return False

    def save(self, save_path):
        self.saver.save(self.sess, save_path)

    def close(self):
        self.qi.close()
        self.sess.close()
