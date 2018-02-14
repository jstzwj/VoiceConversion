import params
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import train1_loader

def train(logdir = params.logdir_path + '/default/train1', queue = True):
    print("train1 start...")

    x_mfcc = tf.placeholder(tf.float32, shape=(params.Train1.batch_size, None, params.Default.n_mfcc))
    y_ppgs = tf.placeholder(tf.int32, shape=(params.Train1.batch_size, None,))
    y_spec = tf.placeholder(tf.float32, shape=(params.Train1.batch_size, None, 1 + params.Default.n_fft // 2))
    y_mel = tf.placeholder(tf.float32, shape=(params.Train1.batch_size, None, params.Default.n_mels))

    # tf Graph input
    # x = tf.placeholder("float", [None, params.Train1.seq_max_len, 1])
    # y = tf.placeholder("float", [None, params.Default.n_ppgs])
    # A placeholder for indicating each sequence length
    seqlen = tf.placeholder(tf.int32, [None])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([params.Train1.hidden_units, params.Default.n_ppgs]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([params.Default.n_ppgs]))
    }


    def dynamicRNN(x, seqlen, weights, biases):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, params.Train1.seq_max_len, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(params.Train1.hidden_units)

        # Get lstm cell output, providing 'sequence_length' will perform dynamic
        # calculation.
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                    sequence_length=seqlen)

        # When performing dynamic calculation, we must retrieve the last
        # dynamically computed output, i.e., if a sequence length is 10, we need
        # to retrieve the 10th output.
        # However TensorFlow doesn't support advanced indexing yet, so we build
        # a custom op that for each sample in batch size, get its length and
        # get the corresponding relevant output.

        # 'outputs' is a list of output at every timestep, we pack them in a Tensor
        # and change back dimension to [batch_size, n_step, n_input]
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])

        # Hack to build the indexing and retrieve the right output.
        batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        index = tf.range(0, batch_size) * params.Train1.seq_max_len + (seqlen - 1)
        # Indexing
        outputs = tf.gather(tf.reshape(outputs, [-1, params.Train1.hidden_units]), index)

        # Linear activation, using outputs computed above
        return tf.matmul(outputs, weights['out']) + biases['out']

    pred = dynamicRNN(x_mfcc, seqlen, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_ppgs))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params.Train1.learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
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
                sess.run(optimizer, feed_dict={x_mfcc: mfcc, y_ppgs: ppg, seqlen: batch_seqlen})
                if step % params.Train1.display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={x_mfcc: mfcc, y_ppgs: ppg, seqlen: batch_seqlen})
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                        "{:.4f}".format(loss) + ", Training Accuracy= " + \
                        "{:.3f}".format(acc))

        print("Optimization Finished!")

        # Calculate accuracy
        test_data = testset.data
        test_label = testset.labels
        test_seqlen = testset.seqlen
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                        seqlen: test_seqlen}))

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