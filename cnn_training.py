import chess
import tensorflow as tf
import numpy as np
import random
import math

##############Creating Dataset##############
def generateData(size):
    size = size - 2
    _X = np.zeros(10)
    _Y = random.randrange(len(_X))
    _X[_Y] = 1
    _Y = math.ceil((_Y-1)/2)
    line_y = np.zeros(5)
    line_y[_Y] = 1



    _X2 = np.zeros(10)
    _Y2 = random.randrange(len(_X2))
    _X2[_Y2] = 1
    _Y2 = math.ceil((_Y2-1) / 2)
    line_y2 = np.zeros(5)
    line_y2[_Y2] = 1

    data_X = np.vstack((_X2, _X))
    data_Y = np.vstack((line_y2, line_y))

    for _ in range(size):

        _X = np.zeros(10)
        _Y = random.randrange(len(_X))
        _X[_Y] = 1
        _Y = math.ceil((_Y - 1) / 2)
        line_y = np.zeros(5)
        line_y[_Y] = 1
        #print(_X)
        #print(line_y)
        data_X = np.vstack((data_X, _X))
        data_Y = np.vstack((data_Y, line_y))

    training_X = np.matrix(data_X)
    training_Y = np.matrix(data_Y)
    return training_X,training_Y
###########################################

def main():
    #writer = tf.summary.FileWriter(".logs/training",sess.graph)

    training_X, training_Y = generateData(100)
    testing_X, testing_Y = generateData(100)

    training_epochs = 1000

    num_inputs = 10
    num_outputs = 5


    layer_1_nodes = 20
    layer_2_nodes = 15
    layer_3_nodes = 10


    with tf.variable_scope('input'):
        input_layer = tf.reshape(fea)
        X = tf.placeholder(tf.float32, shape=[None,num_inputs], name = "X")
    #print(training_X)
    #print(training_Y)


    #Layer 1
    with tf.variable_scope('layer_1'):
        w1 = tf.get_variable("weights1", shape=[num_inputs,layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(name = "biases1", shape =[layer_1_nodes], initializer=tf.zeros_initializer())
        y1 = tf.nn.relu(tf.matmul(X,w1) + b1)

    #Layer 2
    with tf.variable_scope('layer_2'):
        w2 = tf.get_variable("weights2", shape =[layer_1_nodes,layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(name = "biases2", shape = [layer_2_nodes],initializer= tf.zeros_initializer())
        y2 = tf.nn.relu(tf.matmul(y1,w2) + b2)

    # Layer 3
    with tf.variable_scope('layer_3'):
        w3 = tf.get_variable("weights3", shape=[layer_2_nodes, layer_3_nodes],initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
        y3 = tf.nn.relu(tf.matmul(y2, w3) + b3)

    #Output Layer
    with tf.variable_scope('output'):
        w4 = tf.get_variable("weights4",shape = [layer_3_nodes,num_outputs],initializer=tf.contrib.layers.xavier_initializer())
        b4 = tf.get_variable(name = "biases4",shape=[num_outputs], initializer=tf.zeros_initializer())
        prediction = tf.matmul(y3,w4) + b4

    with tf.variable_scope('cost'):
        Y = tf.placeholder(tf.float32, [None, 5], name = "Y")
        #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(final_y), reduction_indices=[1]))
        #train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)
        cost = tf.reduce_mean(tf.squared_difference(prediction,Y))

    #sess = tf.InteractiveSession()
    #tf.global_variables_initializer().run()
    with tf.variable_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate= 0.001).minimize(cost)

    with tf.variable_scope('logging'):
        tf.summary.scalar('current_cost',cost)
        summary = tf.summary.merge_all()


    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        training_writer = tf.summary.FileWriter(".logs/training",session.graph)
        testing_writer = tf.summary.FileWriter(".logs/testing",session.graph)


        for epoch in range(training_epochs):
            session.run(optimizer, feed_dict={X:training_X, Y:training_Y})

            if epoch % 10 == 0:
                training_cost, training_summary = session.run([cost,summary], feed_dict= {X:training_X, Y:training_Y})
                testing_cost, testing_summary = session.run([cost,summary], feed_dict={X:testing_X,Y:testing_Y})
                training_writer.add_summary(training_summary,epoch)
                testing_writer.add_summary(testing_summary,epoch)
                print("Epoch: {} - Training Cost: {} Testing Cost {}:".format(epoch, training_cost,testing_cost))

        final_training_cost = session.run(cost, feed_dict={X: training_X, Y: training_Y})
        final_testing_cost = session.run(cost, feed_dict={X:testing_X,Y:testing_Y})



        print("Final Training cost: {}".format(final_training_cost))
        print("Final Testing cost: {}".format(final_testing_cost))






if __name__ == "__main__":
    main()