import tensorflow as tf
from tfac.queue_input import QueueInput


class ModeKeys(object):
    TRAIN = 'train'
    PREDICT = 'predict'


class NN(object):
    def __init__(self, feature_size, action_size, sample_fn, prefix):
        self.feature_size = feature_size
        self.action_size = action_size
        self.sample_fn = sample_fn
        graph = tf.Graph()
        with graph.as_default():
            self.build_input()
            idx, batch_features, batch_labels = self.qi_train.build_op(32)
            self.train_net = self.build_net(
                batch_features, batch_labels,
                ModeKeys.TRAIN, False, prefix)
            self.pred_net = self.build_net(
                self.features_predict, {},
                ModeKeys.PREDICT, tf.AUTO_REUSE, prefix)
            self.sess = tf.InteractiveSession()
            self.saver = tf.train.Saver(max_to_keep=10)
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            self.run()
            self.sess.graph.finalize()
            self.summary_writer = tf.summary.FileWriter('tf_log/' + prefix,
                                                        self.sess.graph)

    def run(self):
        self.qi_train.run(self.sess, self.sample_fn)

    def train(self):
        _, loss, merged, step = self.sess.run(
            [self.train_net["train_op"], self.train_net["loss"],
             self.train_net["merged"], self.train_net["global_step"]])
        self.summary_writer.add_summary(merged, step)
        return loss

    def predict(self, s):
        return self.sess.run(self.pred_net["predictions"],
                             feed_dict={self.features_predict["s"]: s})

    def build_input(self):
        self.features_train = {
            "s": tf.placeholder(tf.float32, (None, *self.feature_size))
        }
        self.labels_train = {
            "a": tf.placeholder(tf.int32, (None,))
        }

        self.features_predict = {
            "s": tf.placeholder(tf.float32, (None, *self.feature_size))
        }
        self.qi_train = QueueInput(self.features_train, self.labels_train,
                                   [320, 320])

    def build_net(self, features, labels, mode, reuse, prefix):
        training = mode == ModeKeys.TRAIN
        with tf.variable_scope(prefix, reuse=reuse):
            w_init = tf.contrib.layers.xavier_initializer()
            b_init = tf.constant_initializer(0.1)
            x = features["s"]
            x = tf.layers.conv2d(
                x, 32, 4, 2, activation=tf.nn.relu, name="conv_1",
                kernel_initializer=w_init, bias_initializer=b_init)
            tf.summary.histogram('conv_1/activation', x)
            x = tf.layers.batch_normalization(
                x, training=training,
                momentum=0.997, epsilon=1e-5, name="bn_1")
            x = tf.layers.conv2d(
                x, 32, 4, 2, activation=tf.nn.relu, name="conv_2",
                kernel_initializer=w_init, bias_initializer=b_init)
            tf.summary.histogram('conv_2/activation', x)
            x = tf.layers.batch_normalization(
                x, training=training,
                momentum=0.997, epsilon=1e-5, name="bn_2")
            x = tf.layers.conv2d(
                x, 64, 4, 2, activation=tf.nn.relu, name="conv_3",
                kernel_initializer=w_init, bias_initializer=b_init)
            tf.summary.histogram('conv_3/activation', x)
            x = tf.layers.batch_normalization(
                x, training=training,
                momentum=0.997, epsilon=1e-5, name="bn_3")
            x = tf.layers.conv2d(
                x, 64, 3, 1, activation=tf.nn.relu, name="conv_4",
                kernel_initializer=w_init, bias_initializer=b_init)
            tf.summary.histogram('conv_4/activation', x)
            x = tf.layers.batch_normalization(
                x, training=training,
                momentum=0.997, epsilon=1e-5, name="bn_4")
            pi = tf.layers.flatten(x)
            pi = tf.layers.dense(pi, 512, activation=tf.nn.relu,
                                 name="pi_dense_1")
            tf.summary.histogram('pi_dense_1/activation', x)
            pi = tf.layers.batch_normalization(
                pi, training=training,
                momentum=0.997, epsilon=1e-5, name="bn_5")
            pi = tf.layers.dense(pi, self.action_size,
                                 name="pi_dense_2")
            tf.summary.histogram('pi_dense_2/activation', x)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.name[len(prefix):], var)

            predictions = {
                "pi": tf.nn.softmax(pi, name="pi")
            }

            if mode == ModeKeys.PREDICT:
                net = {"predictions": predictions}
                return net

            with tf.name_scope("loss"):
                ce_loss = tf.losses.sparse_softmax_cross_entropy(
                    labels["a"], pi)
                l2_loss = 1e-4 * tf.add_n([tf.nn.l2_loss(var)
                                           for var in tf.trainable_variables()])
                loss = ce_loss + l2_loss

            tf.summary.scalar('ce_loss', ce_loss)
            tf.summary.scalar('l2_loss', l2_loss)
            tf.summary.scalar('loss', loss)

            global_step = tf.train.get_or_create_global_step()
            optimizer = tf.train.AdamOptimizer()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, prefix)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=global_step)

            merged = tf.summary.merge_all()

            net = {
                "predictions": predictions,
                "loss": loss,
                "train_op": train_op,
                "merged": merged,
                "global_step": global_step
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
        self.qi_train.close()
        self.sess.close()
