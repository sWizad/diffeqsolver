import tensorflow as tf
import numpy as np
import traceback
import os
import sys

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0, trainable=True):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale), trainable=trainable)
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias), trainable=trainable)
        return tf.matmul(x, w)+b

def conv(x, scope, *, nf, rf, stride, pad='VALID', init_scale=1.0, trainable=True):
    with tf.variable_scope(scope):
        nin = x.get_shape()[3].value
        w = tf.get_variable("w", [rf, rf, nin, nf], initializer=ortho_init(init_scale), trainable=trainable)
        b = tf.get_variable("b", [nf], initializer=tf.constant_initializer(0.0), trainable=trainable)
        return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=pad)+b

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def gaussian_log_likelihood(x, mean, sig, eps=1e-8):
    # compute log P(x) for diagonal Guassian
    # -1/2 log( (2pi)^k sig_1 * sig_2 * ... * sig_k ) -  sum_i 1/2sig_i^2 (x_i - m_i)^2
    return -0.5 * tf.reduce_sum( tf.log(2.*np.pi*sig*sig + eps)
                               + tf.square(x-mean)/(sig*sig + eps), axis=1)

def colored_hook(home_dir):
  """Colorizes python's error message.
  Args:
    home_dir: directory where code resides (to highlight your own files).
  Returns:
    The traceback hook.
  """

  def hook(type_, value, tb):
    def colorize(text, color, own=0):
      """Returns colorized text."""
      endcolor = "\x1b[0m"
      codes = {
          "green": "\x1b[0;32m",
          "green_own": "\x1b[1;32;40m",
          "red": "\x1b[0;31m",
          "red_own": "\x1b[1;31m",
          "yellow": "\x1b[0;33m",
          "yellow_own": "\x1b[1;33m",
          "black": "\x1b[0;90m",
          "black_own": "\x1b[1;90m",
          "cyan": "\033[1;36m",
      }
      return codes[color + ("_own" if own else "")] + text + endcolor

    for filename, line_num, func, text in traceback.extract_tb(tb):
      basename = os.path.basename(filename)
      own = (home_dir in filename) or ("/" not in filename)

      print(colorize("\"" + basename + '"', "green", own) + " in " + func)
      print("%s:  %s" % (
          colorize("%5d" % line_num, "red", own),
          colorize(text, "yellow", own)))
      print("  %s" % colorize(filename, "black", own))

    print(colorize("%s: %s" % (type_.__name__, value), "cyan"))
  return hook
