import tensorflow as tf

cfg = tf.ConfigProto()
cfg.gpu_options.per_process_gpu_memory_fraction = 0.35
sess = tf.Session(config=cfg)
print(sess.run(tf.contrib.memory_stats.MaxBytesInUse()))
print(sess.run(tf.contrib.memory_stats.BytesInUse()))
print(sess.run(tf.contrib.memory_stats.BytesLimit()))