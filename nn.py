from tf_util import *
from util import *
from collections import namedtuple
import os
import json
import datetime

_MOMENTUM = 0.9

NNArgs = namedtuple("NNArgs", "action, env")


class NN(object):
    def __init__(self, args):
        self.args = args
        self.output_size = self.args.action

    def make_session(self):
        """Make and return a tensorflow session

        Returns:
            Session: tensorflow session
        """
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        cfg = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=cfg)
        return self.sess

    def _build_input(self):
        s = tf.placeholder(tf.uint8, [None, 84, 84, 5], name='s')
        a = tf.placeholder(tf.uint8, [None, ], name='a')
        inputs = {"s": s, "a": a}
        return inputs

    def _build_net(self, mode):
        inputs = self._build_input()
        is_training = mode == "training"
        data_format = "channels_last"
        net = inputs["s"]
        net = conv2d_bn_relu(net, 32, 8, 4, is_training, data_format)
        net = conv2d_bn_relu(net, 64, 4, 2, is_training, data_format)
        net = conv2d_bn_relu(net, 64, 3, 1, is_training, data_format)
        net = tf.layers.flatten(net)
        net = tf.layers.dense(net, 512)
        net = dense_batch_norm_relu(net, is_training)
        net = tf.layers.dense(net, self.output_size)

        predict = tf.nn.softmax(net)

        if mode == "predict":
            return predict

        a = tf.one_hot(inputs["a"], self.output_size)
        cross_entropy = tf.losses.softmax_cross_entropy(
            logits=net, onehot_labels=a)
        l2_loss = 10 ** -4 * tf.add_n([tf.nn.l2_loss(v)
                                       for v in tf.trainable_variables()])
        loss = cross_entropy + l2_loss

        global_step = tf.train.get_or_create_global_step()
        boundaries = [4 * 10 ** 5, 6 * 10 ** 5]
        values = [10 ** -2, 10 ** -3, 10 ** -4]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32), boundaries, values)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=_MOMENTUM)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
        return predict, loss, train_op

    def predict(self, data):
        self.model = self._build_net("predict")
        if not self.load_model():
            init = tf.global_variables_initializer()
            self.sess.run(init)
        self.sess.graph.finalize()

    def train(self, data):
        self.model = self._build_net("training")
        if not self.load_model():
            init = tf.global_variables_initializer()
            self.sess.run(init)
        self.sess.graph.finalize()

    def load_model(self):
        def clean_up():
            self.score = None
            U.main_logger.info("No model loaded")

        self.saver = tf.train.Saver(max_to_keep=50)
        model_path = get_path('model/' + self.args.env)
        subdir = next(os.walk(model_path))[1]
        if subdir:
            clean_up()
            return False

        cmd = input("Found {} saved model(s), "
                    "do you want to load? [y/N]".format(len(subdir)))
        if not ('y' in cmd or 'Y' in cmd):
            clean_up()
            return False

        def ls_model():
            print("Choose one:")
            for i in range(len(subdir)):
                state_fn = model_path + '/' + subdir[i] + '/state.json'
                with open(state_fn, 'r') as f:
                    state = json.load(f)
                print("[{}]: Score: {}, Path: {}".
                      format(i, state['score'], subdir[i]))

        index = 0
        if len(subdir) > 1:
            ls_model()
            index = int(input("Index: "))
        load_path = model_path + '/' + subdir[index]
        state_fn = load_path + '/state.json'
        with open(state_fn, 'r') as f:
            state = json.load(f)
        checkpoint = tf.train.get_checkpoint_state(load_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            U.main_logger.info("Successfully loaded model: Score: {}, Path: {}".
                               format(state['score'], checkpoint.model_checkpoint_path))
            self.score = state['score']
            return True

    def save_model(self):
        save_path = get_path('model/' + self.args.env
                             + '/' + datetime.datetime.now().
                             strftime('%Y%m%d_%H%M%S'))
        U.main_logger.info("Save model at {} with score {:.2f}".
                           format(save_path, self.score))
        self.saver.save(self.sess, save_path + '/model.ckpt')
        with open(save_path + '/state.json', 'w') as f:
            json.dump({'score': self.score, 'args': vars(self.args)}, f)
