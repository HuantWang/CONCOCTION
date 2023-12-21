'''
2022.12.14

@author: xrz
1. 此代码是基于StateTraining.py（wang ke'18），进行了分batch的训练
由于之前x的维度和y的维度无法对应（因为每个代码片段traces的个数不一样），此代码中规定了num_of_trace_for_pre_code，则维度可以确定了
2. 添加了f1等评估指标

'''

import numpy as np
import tensorflow as tf
import nni

from sklearn import metrics

# from tensorflow.contrib import rnn

rnn = tf.compat.v1.nn.rnn_cell

# num_epochs = 1
# learning_rate = 0.0001
# n_hidden = 200
#
# vocabulary_size = 10000 # To be changed: input vocabulary
# CLASSES = 2 # To be changed: prediction classes
# #此处的batchsize指的是traces而不是y_label
# batch_size = 100 # To be changed
# # batch_size = 100 # To be changed
# #直接利用batch代替了
# # program_number = 256 # To be changed: number of programs
#
# # sequence_length
# num_of_neurons=100
# #每个代码片段有几条traces
# num_of_trace_for_pre_code = 2


class StateTraining:
    '''
    classdocs
    '''

    def __init__(self, all_symbol_state_traces,  one_hot_encoding_vectors,\
                 test_all_symbol_state_traces,  test_one_hot_encoding_vectors,
                 num_of_neurons, n_hidden, CLASSES,learning_rate,num_epochs,vocabulary_size,
                num_of_trace_for_pre_code,batch_size):
        '''
        Constructor
        '''
        self.num_of_neurons = num_of_neurons
        self.n_hidden = int(n_hidden)
        self.CLASSES = CLASSES
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.vocabulary_size = vocabulary_size
        self.num_of_trace_for_pre_code = num_of_trace_for_pre_code
        self.batch_size = batch_size

        #y的batch
        batch_number2 = int(batch_size / num_of_trace_for_pre_code)
        self.all_symbol_state_traces = all_symbol_state_traces
        self.variable_state_trace_lengths = np.array([num_of_trace_for_pre_code for x in range(batch_number2)])

        self.variable_variable_trace_lengths = np.array([num_of_neurons for x in range(batch_size)])
        #y
        self.one_hot_encoding_vectors = one_hot_encoding_vectors
        
        self.test_all_symbol_state_traces = test_all_symbol_state_traces
        if len(test_all_symbol_state_traces) >= batch_size:
            self.test_variable_state_trace_lengths = np.array([num_of_trace_for_pre_code for x in range(batch_number2)])
        else:
            self.test_variable_state_trace_lengths = np.array([num_of_trace_for_pre_code for x in range((int)(len(test_all_symbol_state_traces)/num_of_trace_for_pre_code))])

        if len(test_all_symbol_state_traces) >= batch_size:
            self.test_variable_variable_trace_lengths = np.array([num_of_neurons for x in range(batch_size)])
        else:
            self.test_variable_variable_trace_lengths = np.array([num_of_neurons for x in range(len(test_all_symbol_state_traces))])
        self.test_one_hot_encoding_vectors = test_one_hot_encoding_vectors

    def generate_next_batch(self,traces,y,batch):
        # print(batch)
        index1 = np.arange(0, len(traces))
        if self.batch_size*batch+self.batch_size <= len(traces):
            index1 = index1[self.batch_size*batch:self.batch_size*batch+self.batch_size]
        else:
            index1 = index1[self.batch_size * batch:]
        # print("1:"+ str(self.batch_size*batch) +":"+str(self.batch_size*batch+self.batch_size))
        traces_list = [traces[i] for i in index1]

        index2 = np.arange(0, len(y))
        batch_number2 = int(self.batch_size/self.num_of_trace_for_pre_code)
        if batch_number2*batch+batch_number2 <= len(y):
            index2 = index2[batch_number2*batch:batch_number2*batch+batch_number2]
        else:
            index2 = index2[batch_number2 * batch:]
        y_list = [y[i] for i in index2]
        # print("2:" + str(batch_number2*batch)+":"+str(batch_number2*batch+batch_number2))

        return np.asarray(traces_list), np.asarray(y_list)

    def train_evaluate(self):

        # one state (a tuple of variable values) from one program will be one data input
        #state 的输入,不定义batchsize的大小
        #训练
        batches = int(len(self.all_symbol_state_traces)/self.batch_size)
        #测试
        batches_test = int(len(self.test_all_symbol_state_traces) / self.batch_size)
        #y
        batch_number2 = int(self.batch_size / self.num_of_trace_for_pre_code)

        #本来这里应该是这样，但是为了训练集和测试集可输入不同的数据，设置为None
        # x = tf.placeholder(tf.int32, [batch_size, num_of_neurons])
        # # the length of each data input before padding
        # # 每条state trace的未经padding的长度
        # vv = tf.placeholder(tf.int32, [batch_size])
        # # the length of each program (number of states) before padding.
        # # 每个代码片段存在多少state traces?
        # vs = tf.placeholder(tf.int32, [batch_number2])
        # # 每个代码片段的类别
        # y = tf.placeholder(tf.int32, [batch_number2, CLASSES])

        x = tf.placeholder(tf.int32, [None,self.num_of_neurons])
        # the length of each data input before padding
        #每条state trace的未经padding的长度
        vv = tf.placeholder(tf.int32, [None])
        # the length of each program (number of states) before padding.
        #每个代码片段存在多少state traces?
        vs = tf.placeholder(tf.int32, [None])
        #每个代码片段的类别
        y = tf.placeholder(tf.int32, [None, self.CLASSES])

        W = tf.Variable(tf.random_normal([self.n_hidden, self.CLASSES]))
        b = tf.Variable(tf.random_normal([self.CLASSES]))

        # Embedding layer
        embeddings = tf.get_variable('embedding_matrix', [self.vocabulary_size, self.n_hidden])
        #将x重新embedding，（因为要输入到rnn中）
        embedding_rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

        #此上下文管理器验证（可选）values是否来自同一图形，其实就是层的名字
        with tf.variable_scope("embedding"):
            #隐藏神经元的个数
            rnn_embedding_cell = rnn.GRUCell(self.n_hidden)
            #所有数据
            #输出：每个cell的输出（[batch_size,cell.output_size(n_hideen)]）,最后一个cell的输出（state）
            #输入：GRU记忆单元cell，输入的数据[batch_size,max_time,embed_size]，sequence_length（list，长度取决于输入了几个数据，代表每个state的长度）
            all_programs_states_outputs_embedding, _ = tf.nn.dynamic_rnn(rnn_embedding_cell, embedding_rnn_inputs, sequence_length=vv, dtype=tf.float32)
            #重新构建tensor
            # (100,200)
            #由于这里用的是batch_size，所以当数据量小于batch_size,就会出现问题
            all_programs_states_embedding = tf.gather_nd(all_programs_states_outputs_embedding, tf.stack([tf.range(self.batch_size), vv-1], axis = 1))

        #list(100,Tensor(100,200))
        #每个batch的program的embedding
        #同理，数据集小于batch_size会有问题
        batch_of_program_state_embeddings = tf.split(all_programs_states_embedding, batch_number2, 0)


        #tensor(100,100,200)=》对应数据10000,program 100
        batch_of_program_state_embeddings = tf.stack(batch_of_program_state_embeddings, 0)

        with tf.variable_scope("prediction"):
            rnn_prediction_cell = rnn.GRUCell(self.n_hidden)
            rnn_prediction_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_prediction_cell] * 2)
            #100,5,200
            all_programs, _ = tf.nn.dynamic_rnn(rnn_prediction_cell, batch_of_program_state_embeddings, sequence_length=vs, dtype=tf.float32)
            #重新构建tensor
            all_programs_final_state = tf.gather_nd(all_programs, tf.stack([tf.range(batch_number2), vs-1], axis = 1))


        #矩阵相乘,这个其实就是Fully-connected Laye
        #其实就是 y  = model(x)

        print(f"this is W:{W}")
        print(f"this is all_programs_final_state:{all_programs_final_state}")
        print(f"this is b:{b}")
        prediction = tf.matmul(all_programs_final_state, W) + b
        print(f"this is prediction:{prediction}")


        sess = tf.compat.v1.Session()

        # # Execute the graph and store the value that `e` represents in `result`.
        # result = sess.run(prediction)
        # print(f"this is result:{result}")
        #
        # with tf.compat.v1.Session as sess:
        #     sess.run(tf.global_variables_initializer())
        #     A_v, B_v = sess.run([tf.argmax(prediction,1), prediction])
        #
        # print(f"this is tf.argmax(prediction,1):{ A_v}{ B_v}")

        #计算auc
        #比较prediction在列上的最大索引值和y上的是否相同
        #true  false
        correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        #降维计算平均值，cast是tensor的数据类型转换
        #1 0
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        #计算f1
        model_pred = tf.argmax(prediction,1)
        model_True = tf.argmax(y,1)


        #论文最后的softmax层
        #计算分类问题交叉熵损失函数
        #这个函数的返回值不是一个数，而是一个向量。如果要求最终的交叉熵损失，我们需要再做一步tf.reduce_sum操作
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        #自适应矩估计），是一个寻找全局最优点的优化算法，引入了二次梯度校正
        train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999).minimize(loss)

        #保存模型
        saver=tf.train.Saver()
        saverDir="./savedmodel/"
        bestF1intrain=0


        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            #train
            print("self.num_epochs:")
            print(self.num_epochs)
            for i in range(int(self.num_epochs)):
                for batch in range(int(batches)):
                    # Demo purpose only for one batch add a for loop to read multiple batches
                    X_batch_np,Y_batch_np = self.generate_next_batch(self.all_symbol_state_traces,self.one_hot_encoding_vectors,batch)
                    # X_batch = np.reshape(X_batch_np,(X_batch_np[0],num_of_neurons))
                    # # X_batch_np = tf.stack(X_batch_np, 0)
                    # Y_batch = np.reshape(Y_batch_np,[int(batch_size/num_of_trace_for_pre_code), CLASSES])
                    _, _loss = sess.run(

                        [train_step, loss],

                        feed_dict={
                            x: X_batch_np,
                            y: Y_batch_np,
                            vs: self.variable_state_trace_lengths,
                            vv: self.variable_variable_trace_lengths,
                        })

                    print("epoch:%s -> training iteration is %s and total_loss: %s " % (i, batch,_loss))

                    for batch in range(int(batches_test)):
                        X_batch_np, Y_batch_np = self.generate_next_batch(self.test_all_symbol_state_traces,
                                                                          self.test_one_hot_encoding_vectors, batch)

                        #
                        # Y_batch = Y_batch_np[:, None]
                        _accuracy, _y_pred, _y_true = sess.run(

                            [accuracy, model_pred, model_True],

                            feed_dict={
                                x: X_batch_np,
                                y: Y_batch_np,
                                vs: self.test_variable_state_trace_lengths,
                                vv: self.test_variable_variable_trace_lengths,
                            })

                        print("epoch:%s -> Testing The accuracy is %s" % (i,_accuracy))
                        # print(f"this is _y_pred:{_y_pred},{len(_y_pred)}")
                        print("epoch:%s -> Testing Precision" % i, metrics.precision_score(_y_true, _y_pred))
                        print("epoch:%s -> Testing Recall" % i, metrics.recall_score(_y_true, _y_pred))
                        print("epoch:%s -> Testing f1_score" % i, metrics.f1_score(_y_true, _y_pred))
                        # print("confusion_matrix")
                        # print(metrics.confusion_matrix(_y_true, _y_pred))
                        
                        if(metrics.f1_score(_y_true, _y_pred)>bestF1intrain):
                            bestF1intrain=metrics.f1_score(_y_true, _y_pred)
                            saver.save(sess,saverDir+str(bestF1intrain)+".cpkt")
                            print(f"store model in :{saverDir+str(bestF1intrain)}+.cpkg")
                    nni.report_final_result(bestF1intrain)



            #test，test的batch和train的不一定一样
            print(f"this is batches_test:{batches_test}")
            if batches_test!=0:
                for batch in range(int(batches_test)):
                    X_batch_np, Y_batch_np = self.generate_next_batch(self.test_all_symbol_state_traces,
                                                                      self.test_one_hot_encoding_vectors,batch)

                    #
                    # Y_batch = Y_batch_np[:, None]
                    _accuracy,_y_pred,_y_true = sess.run(

                        [accuracy,model_pred,model_True],

                        feed_dict={
                                x : X_batch_np,
                                y : Y_batch_np,
                                vs : self.test_variable_state_trace_lengths,
                                vv : self.test_variable_variable_trace_lengths,
                            })

                    print("The accuracy is %s" % _accuracy)
                    # print(f"this is _y_pred:{_y_pred},{len(_y_pred)}")
                    print("Precision", metrics.precision_score(_y_true, _y_pred))
                    print("Recall", metrics.recall_score(_y_true, _y_pred))
                    print("f1_score", metrics.f1_score(_y_true, _y_pred))
                    print("confusion_matrix")
                    print(metrics.confusion_matrix(_y_true, _y_pred))
                    # fpr, tpr, tresholds = metrics.roc_curve(_y_true, _y_pred)

            else:
                X_batch_np = self.test_all_symbol_state_traces
                Y_batch_np = self.test_one_hot_encoding_vectors

                #
                # Y_batch = Y_batch_np[:, None]
                _accuracy, _y_pred, _y_true = sess.run(

                    [accuracy, model_pred, model_True],

                    feed_dict={
                        x: X_batch_np,
                        y: Y_batch_np,
                        vs: self.test_variable_state_trace_lengths,
                        vv: self.test_variable_variable_trace_lengths,
                    })

                print("The accuracy is %s" % _accuracy)
                print("Precision", metrics.precision_score(_y_true, _y_pred))
                print("Recall", metrics.recall_score(_y_true, _y_pred))
                print("f1_score", metrics.f1_score(_y_true, _y_pred))
                print("confusion_matrix")
                print(metrics.confusion_matrix(_y_true, _y_pred))
                # fpr, tpr, tresholds = metrics.roc_curve(_y_true, _y_pred)

        # 重启一个Session
        with tf.Session() as sess2:
            sess2.run(tf.global_variables_initializer())
            saver.restore(sess2, saverDir+str(bestF1intrain)+".cpkt")
            print(f"load the bestF1model in :{saverDir+str(bestF1intrain)}.cpkt")
            for batch in range(int(batches_test)):
                X_batch_np, Y_batch_np = self.generate_next_batch(self.test_all_symbol_state_traces,
                                                                  self.test_one_hot_encoding_vectors, batch)

                #
                # Y_batch = Y_batch_np[:, None]
                _accuracy, _y_pred, _y_true = sess2.run(

                    [accuracy, model_pred, model_True],

                    feed_dict={
                        x: X_batch_np,
                        y: Y_batch_np,
                        vs: self.test_variable_state_trace_lengths,
                        vv: self.test_variable_variable_trace_lengths,
                    })

                print("The accuracy is %s" % _accuracy)
                # print(f"this is _y_pred:{_y_pred},{len(_y_pred)}")
                print("Precision", metrics.precision_score(_y_true, _y_pred))
                print("Recall", metrics.recall_score(_y_true, _y_pred))
                print("f1_score", metrics.f1_score(_y_true, _y_pred))
                print("confusion_matrix")
                print(metrics.confusion_matrix(_y_true, _y_pred))
                # fpr, tpr, tresholds = metrics.roc_curve(_y_true, _y_pred)
                

class StateTraining_pre:
    '''
    classdocs
    '''

    def __init__(self, all_symbol_state_traces,  one_hot_encoding_vectors,\
                 test_all_symbol_state_traces,  test_one_hot_encoding_vectors,
                 num_of_neurons, n_hidden, CLASSES,learning_rate,num_epochs,vocabulary_size,
                num_of_trace_for_pre_code,batch_size,model_to_load):
        '''
        Constructor
        '''
        self.model_to_load=model_to_load
        self.num_of_neurons = num_of_neurons
        self.n_hidden = n_hidden
        self.CLASSES = CLASSES
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.vocabulary_size = vocabulary_size
        self.num_of_trace_for_pre_code = num_of_trace_for_pre_code
        self.batch_size = batch_size

        #y的batch
        batch_number2 = int(batch_size / num_of_trace_for_pre_code)
        self.all_symbol_state_traces = all_symbol_state_traces
        self.variable_state_trace_lengths = np.array([num_of_trace_for_pre_code for x in range(batch_number2)])

        self.variable_variable_trace_lengths = np.array([num_of_neurons for x in range(batch_size)])
        #y
        self.one_hot_encoding_vectors = one_hot_encoding_vectors
        
        self.test_all_symbol_state_traces = test_all_symbol_state_traces
        if len(test_all_symbol_state_traces) >= batch_size:
            self.test_variable_state_trace_lengths = np.array([num_of_trace_for_pre_code for x in range(batch_number2)])
        else:
            self.test_variable_state_trace_lengths = np.array([num_of_trace_for_pre_code for x in range((int)(len(test_all_symbol_state_traces)/num_of_trace_for_pre_code))])

        if len(test_all_symbol_state_traces) >= batch_size:
            self.test_variable_variable_trace_lengths = np.array([num_of_neurons for x in range(batch_size)])
        else:
            self.test_variable_variable_trace_lengths = np.array([num_of_neurons for x in range(len(test_all_symbol_state_traces))])
        self.test_one_hot_encoding_vectors = test_one_hot_encoding_vectors

    def generate_next_batch(self,traces,y,batch):
        # print(batch)
        index1 = np.arange(0, len(traces))
        if self.batch_size*batch+self.batch_size <= len(traces):
            index1 = index1[self.batch_size*batch:self.batch_size*batch+self.batch_size]
        else:
            index1 = index1[self.batch_size * batch:]
        # print("1:"+ str(self.batch_size*batch) +":"+str(self.batch_size*batch+self.batch_size))
        traces_list = [traces[i] for i in index1]

        index2 = np.arange(0, len(y))
        batch_number2 = int(self.batch_size/self.num_of_trace_for_pre_code)
        if batch_number2*batch+batch_number2 <= len(y):
            index2 = index2[batch_number2*batch:batch_number2*batch+batch_number2]
        else:
            index2 = index2[batch_number2 * batch:]
        y_list = [y[i] for i in index2]
        # print("2:" + str(batch_number2*batch)+":"+str(batch_number2*batch+batch_number2))

        return np.asarray(traces_list), np.asarray(y_list)

    def train_evaluate(self):

        # one state (a tuple of variable values) from one program will be one data input
        #state 的输入,不定义batchsize的大小
        #训练
        batches = int(len(self.all_symbol_state_traces)/self.batch_size)
        #测试
        batches_test = int(len(self.test_all_symbol_state_traces) / self.batch_size)
        #y
        batch_number2 = int(self.batch_size / self.num_of_trace_for_pre_code)

        #本来这里应该是这样，但是为了训练集和测试集可输入不同的数据，设置为None
        # x = tf.placeholder(tf.int32, [batch_size, num_of_neurons])
        # # the length of each data input before padding
        # # 每条state trace的未经padding的长度
        # vv = tf.placeholder(tf.int32, [batch_size])
        # # the length of each program (number of states) before padding.
        # # 每个代码片段存在多少state traces?
        # vs = tf.placeholder(tf.int32, [batch_number2])
        # # 每个代码片段的类别
        # y = tf.placeholder(tf.int32, [batch_number2, CLASSES])

        x = tf.placeholder(tf.int32, [None,self.num_of_neurons])
        # the length of each data input before padding
        #每条state trace的未经padding的长度
        vv = tf.placeholder(tf.int32, [None])
        # the length of each program (number of states) before padding.
        #每个代码片段存在多少state traces?
        vs = tf.placeholder(tf.int32, [None])
        #每个代码片段的类别
        y = tf.placeholder(tf.int32, [None, self.CLASSES])

        W = tf.Variable(tf.random_normal([self.n_hidden, self.CLASSES]))
        b = tf.Variable(tf.random_normal([self.CLASSES]))

        # Embedding layer
        embeddings = tf.get_variable('embedding_matrix', [self.vocabulary_size, self.n_hidden])
        #将x重新embedding，（因为要输入到rnn中）
        embedding_rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

        #此上下文管理器验证（可选）values是否来自同一图形，其实就是层的名字
        with tf.variable_scope("embedding"):
            #隐藏神经元的个数
            rnn_embedding_cell = rnn.GRUCell(self.n_hidden)
            #所有数据
            #输出：每个cell的输出（[batch_size,cell.output_size(n_hideen)]）,最后一个cell的输出（state）
            #输入：GRU记忆单元cell，输入的数据[batch_size,max_time,embed_size]，sequence_length（list，长度取决于输入了几个数据，代表每个state的长度）
            all_programs_states_outputs_embedding, _ = tf.nn.dynamic_rnn(rnn_embedding_cell, embedding_rnn_inputs, sequence_length=vv, dtype=tf.float32)
            #重新构建tensor
            # (100,200)
            #由于这里用的是batch_size，所以当数据量小于batch_size,就会出现问题
            all_programs_states_embedding = tf.gather_nd(all_programs_states_outputs_embedding, tf.stack([tf.range(self.batch_size), vv-1], axis = 1))

        #list(100,Tensor(100,200))
        #每个batch的program的embedding
        #同理，数据集小于batch_size会有问题
        batch_of_program_state_embeddings = tf.split(all_programs_states_embedding, batch_number2, 0)


        #tensor(100,100,200)=》对应数据10000,program 100
        batch_of_program_state_embeddings = tf.stack(batch_of_program_state_embeddings, 0)

        with tf.variable_scope("prediction"):
            rnn_prediction_cell = rnn.GRUCell(self.n_hidden)
            rnn_prediction_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_prediction_cell] * 2)
            #100,5,200
            all_programs, _ = tf.nn.dynamic_rnn(rnn_prediction_cell, batch_of_program_state_embeddings, sequence_length=vs, dtype=tf.float32)
            #重新构建tensor
            all_programs_final_state = tf.gather_nd(all_programs, tf.stack([tf.range(batch_number2), vs-1], axis = 1))


        #矩阵相乘,这个其实就是Fully-connected Laye
        #其实就是 y  = model(x)

        print(f"this is W:{W}")
        print(f"this is all_programs_final_state:{all_programs_final_state}")
        print(f"this is b:{b}")
        prediction = tf.matmul(all_programs_final_state, W) + b
        print(f"this is prediction:{prediction}")


        sess = tf.compat.v1.Session()

        # # Execute the graph and store the value that `e` represents in `result`.
        # result = sess.run(prediction)
        # print(f"this is result:{result}")
        #
        # with tf.compat.v1.Session as sess:
        #     sess.run(tf.global_variables_initializer())
        #     A_v, B_v = sess.run([tf.argmax(prediction,1), prediction])
        #
        # print(f"this is tf.argmax(prediction,1):{ A_v}{ B_v}")

        #计算auc
        #比较prediction在列上的最大索引值和y上的是否相同
        #true  false
        correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        #降维计算平均值，cast是tensor的数据类型转换
        #1 0
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        #计算f1
        model_pred = tf.argmax(prediction,1)
        model_True = tf.argmax(y,1)


        #论文最后的softmax层
        #计算分类问题交叉熵损失函数
        #这个函数的返回值不是一个数，而是一个向量。如果要求最终的交叉熵损失，我们需要再做一步tf.reduce_sum操作
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        #自适应矩估计），是一个寻找全局最优点的优化算法，引入了二次梯度校正
        train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999).minimize(loss)

        #保存模型
        saver=tf.train.Saver()
        saverDir="./savedmodel/"
        bestF1intrain=0


        # 重启一个Session
        with tf.Session() as sess2:
            sess2.run(tf.global_variables_initializer())
            # saver.restore(sess2, saverDir+str(bestF1intrain)+".cpkt")
            saver.restore(sess2, self.model_to_load)
            print(f"load the bestF1model in :{self.model_to_load}")
            for batch in range(int(batches_test)):
                X_batch_np, Y_batch_np = self.generate_next_batch(self.test_all_symbol_state_traces,
                                                                  self.test_one_hot_encoding_vectors, batch)

                #
                # Y_batch = Y_batch_np[:, None]
                _accuracy, _y_pred, _y_true = sess2.run(

                    [accuracy, model_pred, model_True],

                    feed_dict={
                        x: X_batch_np,
                        y: Y_batch_np,
                        vs: self.test_variable_state_trace_lengths,
                        vv: self.test_variable_variable_trace_lengths,
                    })
                print(f"Predict : {_y_pred}")
                print("The accuracy is %s" % _accuracy)
                # print(f"this is _y_pred:{_y_pred},{len(_y_pred)}")
                print("Precision", metrics.precision_score(_y_true, _y_pred))
                print("Recall", metrics.recall_score(_y_true, _y_pred))
                print("f1_score", metrics.f1_score(_y_true, _y_pred))
                print("confusion_matrix")
                print(metrics.confusion_matrix(_y_true, _y_pred))
                # fpr, tpr, tresholds = metrics.roc_curve(_y_true, _y_pred)
                
                def result_log(acc,precision,recall,f1):
                    file_path="./result.log"
                    with open(file_path, 'w') as f:
                        f.writelines(f"acc,{acc}\n")
                        f.writelines(f"pre,{precision}\n")
                        f.writelines(f"recall,{recall}\n")
                        f.writelines(f"f1,{f1}")
                        
                result_log(_accuracy,metrics.precision_score(_y_true, _y_pred),metrics.recall_score(_y_true, _y_pred),metrics.f1_score(_y_true, _y_pred))
    
                



class StateTrainingPredic:
    '''
    classdocs
    '''

    def __init__(self, all_symbol_state_traces, one_hot_encoding_vectors, \
                 test_all_symbol_state_traces, test_one_hot_encoding_vectors,
                 num_of_neurons, n_hidden, CLASSES, learning_rate, num_epochs, vocabulary_size,
                 num_of_trace_for_pre_code, batch_size):
        '''
        Constructor
        '''
        self.num_of_neurons = num_of_neurons
        self.n_hidden = n_hidden
        self.CLASSES = CLASSES
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.vocabulary_size = vocabulary_size
        self.num_of_trace_for_pre_code = num_of_trace_for_pre_code
        self.batch_size = batch_size

        # y的batch
        batch_number2 = int(batch_size / num_of_trace_for_pre_code)
        self.all_symbol_state_traces = all_symbol_state_traces
        self.variable_state_trace_lengths = np.array([num_of_trace_for_pre_code for x in range(batch_number2)])

        self.variable_variable_trace_lengths = np.array([num_of_neurons for x in range(batch_size)])
        # y
        self.one_hot_encoding_vectors = one_hot_encoding_vectors

        self.test_all_symbol_state_traces = test_all_symbol_state_traces
        if len(test_all_symbol_state_traces) >= batch_size:
            self.test_variable_state_trace_lengths = np.array([num_of_trace_for_pre_code for x in range(batch_number2)])
        else:
            self.test_variable_state_trace_lengths = np.array([num_of_trace_for_pre_code for x in range(
                (int)(len(test_all_symbol_state_traces) / num_of_trace_for_pre_code))])

        if len(test_all_symbol_state_traces) >= batch_size:
            self.test_variable_variable_trace_lengths = np.array([num_of_neurons for x in range(batch_size)])
        else:
            self.test_variable_variable_trace_lengths = np.array(
                [num_of_neurons for x in range(len(test_all_symbol_state_traces))])
        self.test_one_hot_encoding_vectors = test_one_hot_encoding_vectors

    def generate_next_batch(self, traces, y, batch):
        # print(batch)
        index1 = np.arange(0, len(traces))
        if self.batch_size * batch + self.batch_size <= len(traces):
            index1 = index1[self.batch_size * batch:self.batch_size * batch + self.batch_size]
        else:
            index1 = index1[self.batch_size * batch:]
        # print("1:"+ str(self.batch_size*batch) +":"+str(self.batch_size*batch+self.batch_size))
        traces_list = [traces[i] for i in index1]

        index2 = np.arange(0, len(y))
        batch_number2 = int(self.batch_size / self.num_of_trace_for_pre_code)
        if batch_number2 * batch + batch_number2 <= len(y):
            index2 = index2[batch_number2 * batch:batch_number2 * batch + batch_number2]
        else:
            index2 = index2[batch_number2 * batch:]
        y_list = [y[i] for i in index2]
        # print("2:" + str(batch_number2*batch)+":"+str(batch_number2*batch+batch_number2))

        return np.asarray(traces_list), np.asarray(y_list)

    def train_evaluate(self):

        # one state (a tuple of variable values) from one program will be one data input
        # state 的输入,不定义batchsize的大小
        # 训练
        batches = int(len(self.all_symbol_state_traces) / self.batch_size)
        # 测试
        batches_test = int(len(self.test_all_symbol_state_traces) / self.batch_size)
        # y
        batch_number2 = int(self.batch_size / self.num_of_trace_for_pre_code)

        # 本来这里应该是这样，但是为了训练集和测试集可输入不同的数据，设置为None
        # x = tf.placeholder(tf.int32, [batch_size, num_of_neurons])
        # # the length of each data input before padding
        # # 每条state trace的未经padding的长度
        # vv = tf.placeholder(tf.int32, [batch_size])
        # # the length of each program (number of states) before padding.
        # # 每个代码片段存在多少state traces?
        # vs = tf.placeholder(tf.int32, [batch_number2])
        # # 每个代码片段的类别
        # y = tf.placeholder(tf.int32, [batch_number2, CLASSES])

        x = tf.placeholder(tf.int32, [None, self.num_of_neurons])
        # the length of each data input before padding
        # 每条state trace的未经padding的长度
        vv = tf.placeholder(tf.int32, [None])
        # the length of each program (number of states) before padding.
        # 每个代码片段存在多少state traces?
        vs = tf.placeholder(tf.int32, [None])
        # 每个代码片段的类别
        y = tf.placeholder(tf.int32, [None, self.CLASSES])

        W = tf.Variable(tf.random_normal([self.n_hidden, self.CLASSES]))
        b = tf.Variable(tf.random_normal([self.CLASSES]))

        # Embedding layer
        embeddings = tf.get_variable('embedding_matrix', [self.vocabulary_size, self.n_hidden])
        # 将x重新embedding，（因为要输入到rnn中）
        embedding_rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

        # 此上下文管理器验证（可选）values是否来自同一图形，其实就是层的名字
        with tf.variable_scope("embedding"):
            # 隐藏神经元的个数
            rnn_embedding_cell = rnn.GRUCell(self.n_hidden)
            # 所有数据
            # 输出：每个cell的输出（[batch_size,cell.output_size(n_hideen)]）,最后一个cell的输出（state）
            # 输入：GRU记忆单元cell，输入的数据[batch_size,max_time,embed_size]，sequence_length（list，长度取决于输入了几个数据，代表每个state的长度）
            all_programs_states_outputs_embedding, _ = tf.nn.dynamic_rnn(rnn_embedding_cell, embedding_rnn_inputs,
                                                                         sequence_length=vv, dtype=tf.float32)
            # 重新构建tensor
            # (100,200)
            # 由于这里用的是batch_size，所以当数据量小于batch_size,就会出现问题
            all_programs_states_embedding = tf.gather_nd(all_programs_states_outputs_embedding,
                                                         tf.stack([tf.range(self.batch_size), vv - 1], axis=1))

        # list(100,Tensor(100,200))
        # 每个batch的program的embedding
        # 同理，数据集小于batch_size会有问题
        batch_of_program_state_embeddings = tf.split(all_programs_states_embedding, batch_number2, 0)

        # tensor(100,100,200)=》对应数据10000,program 100
        batch_of_program_state_embeddings = tf.stack(batch_of_program_state_embeddings, 0)

        with tf.variable_scope("prediction"):
            rnn_prediction_cell = rnn.GRUCell(self.n_hidden)
            rnn_prediction_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_prediction_cell] * 2)
            # 100,5,200
            all_programs, _ = tf.nn.dynamic_rnn(rnn_prediction_cell, batch_of_program_state_embeddings,
                                                sequence_length=vs, dtype=tf.float32)
            # 重新构建tensor
            all_programs_final_state = tf.gather_nd(all_programs, tf.stack([tf.range(batch_number2), vs - 1], axis=1))

        # 矩阵相乘,这个其实就是Fully-connected Laye
        # 其实就是 y  = model(x)

        print(f"this is W:{W}")
        print(f"this is all_programs_final_state:{all_programs_final_state}")
        print(f"this is b:{b}")
        prediction = tf.matmul(all_programs_final_state, W) + b
        print(f"this is prediction:{prediction}")

        sess = tf.compat.v1.Session()

        # # Execute the graph and store the value that `e` represents in `result`.
        # result = sess.run(prediction)
        # print(f"this is result:{result}")
        #
        # with tf.compat.v1.Session as sess:
        #     sess.run(tf.global_variables_initializer())
        #     A_v, B_v = sess.run([tf.argmax(prediction,1), prediction])
        #
        # print(f"this is tf.argmax(prediction,1):{ A_v}{ B_v}")

        # 计算auc
        # 比较prediction在列上的最大索引值和y上的是否相同
        # true  false
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        # 降维计算平均值，cast是tensor的数据类型转换
        # 1 0
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # 计算f1
        model_pred = tf.argmax(prediction, 1)
        model_True = tf.argmax(y, 1)

        # 论文最后的softmax层
        # 计算分类问题交叉熵损失函数
        # 这个函数的返回值不是一个数，而是一个向量。如果要求最终的交叉熵损失，我们需要再做一步tf.reduce_sum操作
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        # 自适应矩估计），是一个寻找全局最优点的优化算法，引入了二次梯度校正
        train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999).minimize(loss)

        # 保存模型
        saver = tf.train.Saver()
        saverDir = "log/"
        bestF1intrain = 0

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            # train
            for i in range(self.num_epochs):
                for batch in range(int(batches)):
                    # Demo purpose only for one batch add a for loop to read multiple batches
                    X_batch_np, Y_batch_np = self.generate_next_batch(self.all_symbol_state_traces,
                                                                      self.one_hot_encoding_vectors, batch)
                    # X_batch = np.reshape(X_batch_np,(X_batch_np[0],num_of_neurons))
                    # # X_batch_np = tf.stack(X_batch_np, 0)
                    # Y_batch = np.reshape(Y_batch_np,[int(batch_size/num_of_trace_for_pre_code), CLASSES])
                    _, _loss = sess.run(

                        [train_step, loss],

                        feed_dict={
                            x: X_batch_np,
                            y: Y_batch_np,
                            vs: self.variable_state_trace_lengths,
                            vv: self.variable_variable_trace_lengths,
                        })

                    print("epoch:%s -> training iteration is %s and total_loss: %s " % (i, batch, _loss))

                    for batch in range(int(batches_test)):
                        X_batch_np, Y_batch_np = self.generate_next_batch(self.test_all_symbol_state_traces,
                                                                          self.test_one_hot_encoding_vectors, batch)

                        #
                        # Y_batch = Y_batch_np[:, None]
                        _accuracy, _y_pred, _y_true = sess.run(

                            [accuracy, model_pred, model_True],

                            feed_dict={
                                x: X_batch_np,
                                y: Y_batch_np,
                                vs: self.test_variable_state_trace_lengths,
                                vv: self.test_variable_variable_trace_lengths,
                            })

                        print("The accuracy is %s" % _accuracy)
                        # print(f"this is _y_pred:{_y_pred},{len(_y_pred)}")
                        print("Precision", metrics.precision_score(_y_true, _y_pred))
                        print("Recall", metrics.recall_score(_y_true, _y_pred))
                        print("f1_score", metrics.f1_score(_y_true, _y_pred))
                        # print("confusion_matrix")
                        # print(metrics.confusion_matrix(_y_true, _y_pred))
                        if (metrics.f1_score(_y_true, _y_pred) > bestF1intrain):
                            bestF1intrain = metrics.f1_score(_y_true, _y_pred)
                            saver.save(sess, saverDir + str(bestF1intrain) + ".cpkt")
                            print(f"store model in :{saverDir + str(bestF1intrain)}+.cpkg")

            # test，test的batch和train的不一定一样
            print(f"this is batches_test:{batches_test}")
            if batches_test != 0:
                for batch in range(int(batches_test)):
                    X_batch_np, Y_batch_np = self.generate_next_batch(self.test_all_symbol_state_traces,
                                                                      self.test_one_hot_encoding_vectors, batch)

                    #
                    # Y_batch = Y_batch_np[:, None]
                    _accuracy, _y_pred, _y_true = sess.run(

                        [accuracy, model_pred, model_True],

                        feed_dict={
                            x: X_batch_np,
                            y: Y_batch_np,
                            vs: self.test_variable_state_trace_lengths,
                            vv: self.test_variable_variable_trace_lengths,
                        })

                    print("The accuracy is %s" % _accuracy)
                    # print(f"this is _y_pred:{_y_pred},{len(_y_pred)}")
                    print("Precision", metrics.precision_score(_y_true, _y_pred))
                    print("Recall", metrics.recall_score(_y_true, _y_pred))
                    print("f1_score", metrics.f1_score(_y_true, _y_pred))
                    print("confusion_matrix")
                    print(metrics.confusion_matrix(_y_true, _y_pred))
                    # fpr, tpr, tresholds = metrics.roc_curve(_y_true, _y_pred)

            else:
                X_batch_np = self.test_all_symbol_state_traces
                Y_batch_np = self.test_one_hot_encoding_vectors

                #
                # Y_batch = Y_batch_np[:, None]
                _accuracy, _y_pred, _y_true = sess.run(

                    [accuracy, model_pred, model_True],

                    feed_dict={
                        x: X_batch_np,
                        y: Y_batch_np,
                        vs: self.test_variable_state_trace_lengths,
                        vv: self.test_variable_variable_trace_lengths,
                    })

                print("The accuracy is %s" % _accuracy)
                print("Precision", metrics.precision_score(_y_true, _y_pred))
                print("Recall", metrics.recall_score(_y_true, _y_pred))
                print("f1_score", metrics.f1_score(_y_true, _y_pred))
                print("confusion_matrix")
                print(metrics.confusion_matrix(_y_true, _y_pred))
                # fpr, tpr, tresholds = metrics.roc_curve(_y_true, _y_pred)

        # 重启一个Session
        with tf.Session() as sess2:
            sess2.run(tf.global_variables_initializer())
            saver.restore(sess2, saverDir + str(bestF1intrain) + ".cpkt")
            print(f"load the bestF1model in :{saverDir + str(bestF1intrain)}.cpkt")
            for batch in range(int(batches_test)):
                X_batch_np, Y_batch_np = self.generate_next_batch(self.test_all_symbol_state_traces,
                                                                  self.test_one_hot_encoding_vectors, batch)

                #
                # Y_batch = Y_batch_np[:, None]
                _accuracy, _y_pred, _y_true = sess2.run(

                    [accuracy, model_pred, model_True],

                    feed_dict={
                        x: X_batch_np,
                        y: Y_batch_np,
                        vs: self.test_variable_state_trace_lengths,
                        vv: self.test_variable_variable_trace_lengths,
                    })

                print("The accuracy is %s" % _accuracy)
                # print(f"this is _y_pred:{_y_pred},{len(_y_pred)}")
                print("Precision", metrics.precision_score(_y_true, _y_pred))
                print("Recall", metrics.recall_score(_y_true, _y_pred))
                print("f1_score", metrics.f1_score(_y_true, _y_pred))
                print("confusion_matrix")
                print(metrics.confusion_matrix(_y_true, _y_pred))
                # fpr, tpr, tresholds = metrics.roc_curve(_y_true, _y_pred)

# if __name__ == '__main__':
    # #训练集
    # #traces量为10000
    # train_x = np.random.randint(10, 50, (5000, 100))
    # #labels为1000，也就是代码片段数，因为是one-hot所以是二维
    # train_y = np.random.randint(0, 2, (1000,2))
    # #每个state的长度
    # train_vv = np.random.randint(1, 10, (5000))
    # #每个代码片段数有多少个states
    # train_vs = np.random.randint(1, 10, (1000))
    #
    # #测试集
    # test_x = np.random.randint(10, 50, (1005, 100))
    # test_y = np.random.randint(0, 2, (201,2))
    # # 每个trace的长度
    # test_vv = np.random.randint(1, 10, (1005))
    # # 每个代码片段数有多少个states，在这里面可能不会超过batchsize，先不用思考
    # test_vs = np.random.randint(1, 10, (201))
    # liger1 = StateTraining(train_x, train_y, test_x, test_y)
    # liger1.train_evaluate()

    #目前数据集数量的关系是：
    #x是traces的数量，y是标签数量，之间的关系是x = y* num_of_trace_for_pre_code && y = program_number
    #vv是每个traces的长度，vs是每个标签包含多少个路径，这里是num_of_trace_for_pre_code