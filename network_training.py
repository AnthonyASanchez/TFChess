import chess
import tensorflow as tf
import numpy as np
import random
import math

##############Creating Dataset##############
def generateData(size, length):
    size = size - 2
    _X = np.zeros(length)
    _Y = random.randrange(len(_X))
    _X[_Y] = 1
    _Y = math.ceil((_Y-1)/2)
    line_y = np.zeros(int(length/2))
    line_y[_Y] = 1



    _X2 = np.zeros(length)
    _Y2 = random.randrange(len(_X2))
    _X2[_Y2] = 1
    _Y2 = math.ceil((_Y2-1) / 2)
    line_y2 = np.zeros(int(length/2))
    line_y2[_Y2] = 1

    data_X = np.vstack((_X2, _X))
    data_Y = np.vstack((line_y2, line_y))

    for _ in range(size):
        if(size%500==0):
            print(size)
        _X = np.zeros(length)
        _Y = random.randrange(len(_X))
        _X[_Y] = 1
        _Y = math.ceil((_Y - 1) / 2)
        line_y = np.zeros(int(length/2))
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

    training_X, training_Y = generateData(1000,768)
    testing_X, testing_Y = generateData(1000,768)

    training_epochs = 100

    num_inputs = 768
    num_outputs = 384


    layer_input = 768
    layer_1_nodes = 2048
    layer_2_nodes = 2028
    layer_3_nodes = 2048

    with tf.variable_scope('input'):
        X = tf.placeholder(tf.float32, shape=[None,num_inputs], name = "X")
    #print(training_X)
    #print(training_Y)


    #Input Layer
    with tf.variable_scope('layer_input'):
        w1 = tf.get_variable("weights1", shape=[num_inputs,layer_input], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(name = "biases1", shape =[layer_input], initializer=tf.zeros_initializer())
        y1 = tf.nn.relu(tf.matmul(X,w1) + b1)

    #Layer 1
    with tf.variable_scope('layer_1'):
        w2 = tf.get_variable("weights2", shape =[layer_input,layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(name = "biases2", shape = [layer_1_nodes],initializer= tf.zeros_initializer())
        y2 = tf.nn.relu(tf.matmul(y1,w2) + b2)

    # Layer 2
    with tf.variable_scope('layer_2'):
        w3 = tf.get_variable("weights3", shape=[layer_1_nodes, layer_2_nodes],initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable(name="biases3", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
        y3 = tf.nn.relu(tf.matmul(y2, w3) + b3)

    # Layer 3
    with tf.variable_scope('layer_3'):
        w4 = tf.get_variable("weights3", shape=[layer_2_nodes, layer_3_nodes],
                             initializer=tf.contrib.layers.xavier_initializer())
        b4 = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
        y4 = tf.nn.relu(tf.matmul(y3, w4) + b4)

    #Output Layer
    with tf.variable_scope('output'):
        w5 = tf.get_variable("weights4",shape = [layer_3_nodes,num_outputs],initializer=tf.contrib.layers.xavier_initializer())
        b5 = tf.get_variable(name = "biases4",shape=[num_outputs], initializer=tf.zeros_initializer())
        prediction = tf.matmul(y4,w5) + b5

    with tf.variable_scope('cost'):
        Y = tf.placeholder(tf.float32, [None, num_outputs], name = "Y")
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

        training_writer = tf.summary.FileWriter("../.logs/training",session.graph)
        testing_writer = tf.summary.FileWriter("../.logs/testing",session.graph)


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








                #tf.scalar_summary("cost", cross_entropy)
    #tf.scalar_summary("accuracy", accuracy)
    #for _ in range(100):
    #  sess.run(train_step, feed_dict={x: training_X, y_: training_Y})
    #correct_prediction = tf.equal(tf.argmax(final_y,1), tf.argmax(y_,1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #print(sess.run(accuracy, feed_dict={x: testing_X, y_:testing_Y}))


    #sess.close()


if __name__ == "__main__":
    main()