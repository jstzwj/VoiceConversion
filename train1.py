import params
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import train1_loader

def train(logdir = params.logdir_path + '/default/train1', queue = True):
    print("train1 start...")
    '''
    To classify images using a recurrent neural network, we consider every image
    row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
    handle 28 sequences of 28 steps for every sample.
    '''
    # Network Parameters
    # num_input = 28 # MNIST data input (img shape: 28*28)
    # timesteps = 28 # timesteps
    num_hidden = 128 # hidden layer num of features
    num_classes = 61 # MNIST total classes (0-9 digits)

    # tf Graph input
    # X = tf.placeholder("float", [None, timesteps, num_input])
    # Y = tf.placeholder("float", [None, num_classes])

    x_mfcc = tf.placeholder(tf.float32, shape=(params.Train1.batch_size, None, params.Default.n_mfcc))
    y_ppgs = tf.placeholder(tf.int32, shape=(params.Train1.batch_size, None,))
    y_spec = tf.placeholder(tf.float32, shape=(params.Train1.batch_size, None, 1 + params.Default.n_fft // 2))
    y_mel = tf.placeholder(tf.float32, shape=(params.Train1.batch_size, None, params.Default.n_mels))

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }


    def RNN(x, weights, biases):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, axis=1)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    logits = RNN(x_mfcc, weights, biases)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=y_ppgs))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params.Train1.learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_ppgs, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        for epoch in range(1, params.Train1.training_epochs_num + 1):
            for step in range(1, params.Train1.training_steps_num+1):
                mfcc, ppg = train1_loader.get_batch(params.Train1.batch_size)
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={x_mfcc: mfcc, y_ppgs: ppg})
                if step % params.Train1.display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                        Y: batch_y})
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                        "{:.4f}".format(loss) + ", Training Accuracy= " + \
                        "{:.3f}".format(acc))

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type = str, default = 'default', help = 'experiment case name')
    arguments = parser.parse_args()
    return arguments

if __name__ == '__main__':
    args = get_arguments()
    case = args.c
    logdir = '{}/{}/train1'.format(params.logdir_path, case)
    train(logdir=logdir, queue=False)
    print("Done")