'''
Created on Aug 12, 2017

@author: wangke

这个代码是一次性输入的，因为每个代码片段的traces数不一样，所以导致了只能一个batch内训练所有数据
'''

import numpy as np
import tensorflow as tf

from sklearn import metrics

rnn = tf.compat.v1.nn.rnn_cell

num_epochs = 10
learning_rate = 0.0001
n_hidden = 200

vocabulary_size = 10000 # To be changed: input vocabulary
CLASSES = 2 # To be changed: prediction classes
batch_size = 10000 # To be changed
# batch_size = 100 # To be changed
program_number = 1000 # To be changed: number of programs
num_of_neurons=100

class StateTraining:
    '''
    classdocs
    '''

    def __init__(self, all_symbol_state_traces, variable_state_trace_lengths, variable_variable_trace_lengths, one_hot_encoding_vectors,\
                 test_all_symbol_state_traces, test_variable_state_trace_lengths, test_variable_variable_trace_lengths, test_one_hot_encoding_vectors):
        '''
        Constructor
        '''
        self.all_symbol_state_traces = all_symbol_state_traces
        self.variable_state_trace_lengths = variable_state_trace_lengths

        self.variable_variable_trace_lengths = variable_variable_trace_lengths
        #y
        self.one_hot_encoding_vectors = one_hot_encoding_vectors
        
        self.test_all_symbol_state_traces = test_all_symbol_state_traces
        self.test_variable_state_trace_lengths = test_variable_state_trace_lengths

        self.test_variable_variable_trace_lengths = test_variable_variable_trace_lengths
        self.test_one_hot_encoding_vectors = test_one_hot_encoding_vectors
        
    
    def train_evaluate(self):
        
        # # one state (a tuple of variable values) from one program will be one data input
        # #state 的输入
        # x = tf.placeholder(tf.int32, [batch_size, None])
        # # the length of each data input before padding
        # #每条state trace的未经padding的长度
        # vv = tf.placeholder(tf.int32, [batch_size])
        # # the length of each program (number of states) before padding.
        # #每个代码片段存在多少state traces?
        # vs = tf.placeholder(tf.int32, [program_number])
        # #每个代码片段的类别
        # y = tf.placeholder(tf.int32, [program_number, CLASSES])

        # one state (a tuple of variable values) from one program will be one data input
        # state 的输入
        x = tf.placeholder(tf.int32, [None,num_of_neurons ])
        # the length of each data input before padding
        # 每条state trace的未经padding的长度
        vv = tf.placeholder(tf.int32, [None])
        # the length of each program (number of states) before padding.
        # 每个代码片段存在多少state traces?
        vs = tf.placeholder(tf.int32, [None])
        # 每个代码片段的类别
        y = tf.placeholder(tf.int32, [None, CLASSES])
        
        W = tf.Variable(tf.random_normal([n_hidden, CLASSES]))        
        b = tf.Variable(tf.random_normal([CLASSES]))
        
        # Embedding layer
        embeddings = tf.get_variable('embedding_matrix', [vocabulary_size, n_hidden])
        #将x重新embedding，（因为要输入到rnn中）
        embedding_rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

        #此上下文管理器验证（可选）values是否来自同一图形，其实就是层的名字
        with tf.variable_scope("embedding"):
            #隐藏神经元的个数
            rnn_embedding_cell = rnn.GRUCell(n_hidden)
            #所有数据
            #输出：每个cell的输出（[batch_size,cell.output_size(n_hideen)]）,最后一个cell的输出（state）
            #输入：GRU记忆单元cell，输入的数据[batch_size,max_time,embed_size]，sequence_length（list，长度取决于输入了几个数据，代表每个state的长度）
            all_programs_states_outputs_embedding, _ = tf.nn.dynamic_rnn(rnn_embedding_cell, embedding_rnn_inputs, sequence_length=vv, dtype=tf.float32)
            #重新构建tensor
            all_programs_states_embedding = tf.gather_nd(all_programs_states_outputs_embedding, tf.stack([tf.range(batch_size), vv-1], axis = 1))

        #list(100,Tensor(100,200))
        #每个batch的program的embedding
        batch_of_program_state_embeddings = tf.split(all_programs_states_embedding, program_number, 0)
        batch_of_program_state_embeddings = tf.stack(batch_of_program_state_embeddings, 0)

        with tf.variable_scope("prediction"):
            rnn_prediction_cell = rnn.GRUCell(n_hidden)
            rnn_prediction_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_prediction_cell] * 2)
            all_programs, _ = tf.nn.dynamic_rnn(rnn_prediction_cell, batch_of_program_state_embeddings, sequence_length=vs, dtype=tf.float32)
            #重新构建tensor
            all_programs_final_state = tf.gather_nd(all_programs, tf.stack([tf.range(program_number), vs-1], axis = 1))

        #矩阵相乘
        prediction = tf.matmul(all_programs_final_state, W) + b
        #比较prediction在列上的最大索引值和y上的是否相同
        correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        #降维计算平均值，cast是tensor的数据类型转换
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # 计算f1
        model_pred = tf.argmax(prediction, 1)
        model_True = tf.argmax(y, 1)

        #计算分类问题交叉熵损失函数
        #这个函数的返回值不是一个数，而是一个向量。如果要求最终的交叉熵损失，我们需要再做一步tf.reduce_sum操作
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        #自适应矩估计），是一个寻找全局最优点的优化算法，引入了二次梯度校正
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999).minimize(loss)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            for i in range(num_epochs):

                # Demo purpose only for one batch add a for loop to read multiple batches

                _,_loss = sess.run(

                    [train_step, loss],

                    feed_dict={
                        x : self.all_symbol_state_traces,
                        y : np.stack(self.one_hot_encoding_vectors, axis = 0),
                        vs : self.variable_state_trace_lengths,
                        vv : self.variable_variable_trace_lengths,
                    })

                print("training iteration is %s and total_loss: %s "%(i,_loss))

            _accuracy, _y_pred, _y_true = sess.run(

                [accuracy, model_pred, model_True],

                feed_dict={
                    x: self.test_all_symbol_state_traces,
                    y: np.stack(self.test_one_hot_encoding_vectors, axis=0),
                    vs: self.test_variable_state_trace_lengths,
                    vv: self.test_variable_variable_trace_lengths,
                })

            print("The accuracy is %s" % _accuracy)
            print("Precision", metrics.precision_score(_y_true, _y_pred))
            print("Recall", metrics.recall_score(_y_true, _y_pred))
            print("f1_score", metrics.f1_score(_y_true, _y_pred))
            print("confusion_matrix")
            print(metrics.confusion_matrix(_y_true, _y_pred))



if __name__ == '__main__':
    #训练集
    #traces量为10000
    train_x = np.random.randint(10, 50, (10000, 100))
    #labels为1000，也就是代码片段数，因为是one-hot所以是二维
    train_y = np.random.randint(0, 2, (1000,2))
    #每个state的长度
    train_vv = np.random.randint(1, 10, (10000))
    #每个代码片段数有多少个states
    train_vs = np.random.randint(1, 10, (1000))

    #测试集
    test_x = np.random.randint(10, 50, (1000, 100))
    test_y = np.random.randint(0, 2, (100,2))
    # 每个trace的长度
    test_vv = np.random.randint(1, 10, (1000))
    # 每个代码片段数有多少个states，在这里面可能不会超过batchsize，先不用思考
    test_vs = np.random.randint(1, 10, (100))
    liger1 = StateTraining(train_x, train_vs, train_vv, train_y, test_x, test_vs, test_vv, test_y)
    liger1.train_evaluate()

    #这个数量是随机的，不知道有没有问题