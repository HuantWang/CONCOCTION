'''
Created on Aug 11, 2017

@author: wangke
'''

# from tensorflow.contrib import rnn
import numpy as np
import tensorflow as tf

rnn = tf.compat.v1.nn.rnn_cell

#将所有样本对模型进行一次完整训练的次数
num_epochs = 100
learning_rate = 0.0001
n_hidden = 200


vocabulary_size = 10000 # To be changed: input vocabulary
#预测的分类个数
CLASSES = 2 # To be changed: prediction classes
#按理说是将整个训练样本分成若干个batch,但是由于feed_dict的x直接输入的是x,则他应该是把所有数据一次输入了
batch_size = 100 # To be changed
#可以理解为代码片段的个数，因为一个代码片段可能对应了多个traces，这里应该每个代码片段的trace个数是相同的
program_number = 20 # To be changed: number of programs
trace_number = 5

class Training:
    '''
    classdocs
    '''

    def __init__(self, all_program_symbol_traces, trace_lengths, labels, test_all_program_symbol_traces, test_trace_lengths, test_labels):
        '''
        Constructor
        '''
        # 训练集

        # 所有程序的所有traces的拼接，其维度为二维，[batch_size, None]
        self.all_program_symbol_traces = all_program_symbol_traces
        # 每条trace的长度，其维度为一维，[batch_size]，（每条trace不一样，所以不是一个数）
        self.trace_lengths = trace_lengths
        # 每条trace的label，其维度为一维，[batches/traces]，只不过这一在这个代码里batches是所有的数据了
        self.labels = labels

        # 测试集
        self.test_all_program_symbol_traces = test_all_program_symbol_traces
        self.test_trace_lengths = test_trace_lengths
        self.test_labels = test_labels

                
    def train_evaluate(self):
        
        # trace inputs for training: each variable trace forms one data input, meaning one program may consist of multiple traces         
        x = tf.placeholder(tf.int32, [batch_size, None]) 
        # length of each variable trace before padding
        seq_lengths = tf.placeholder(tf.int32, [batch_size]) 
        # labels for training
        #假设batchsize是100，共有20个代码片段，每个代码片段有5个traces，则每个代码片段的y应该是100/5 = 20 ，[batches/traces]
        y = tf.placeholder(tf.int32, [batch_size/trace_number])
        # y = tf.placeholder(tf.int32, [batch_size])

#         keep_prob = tf.constant(1.0)

        W = tf.Variable(tf.random_normal([n_hidden, CLASSES]))        
        b = tf.Variable(tf.random_normal([CLASSES]))
        
        # Embedding layer
        embeddings = tf.get_variable('embedding_matrix', [vocabulary_size, n_hidden])
        rnn_inputs = tf.nn.embedding_lookup(embeddings, x)
    
        # RNN
        cell = rnn.GRUCell(n_hidden)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2)
                
        init_state = cell.zero_state(batch_size, tf.float32)
        rnn_outputs, _ = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=seq_lengths, initial_state=init_state)


#         rnn_inputs = tf.nn.dropout(rnn_inputs, keep_prob)
#         rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)
         
        # remove padding effects
        last_rnn_output = tf.gather_nd(rnn_outputs, tf.stack([tf.range(batch_size), seq_lengths-1], axis = 1))
        
        # again one program may have multiple variable traces so split with the number of programs instead of batches/variables
        # list_of_program_tensors = tf.split(last_rnn_output, program_number, 0)
        list_of_program_tensors = tf.split(last_rnn_output, int(batch_size / trace_number), 0)

        all_programs_tensors = []
        for program_tensor in list_of_program_tensors:

            summed_reduced_program_tensor = tf.reduce_max(program_tensor, 0)
#             states_embedding_each_training_program = tf.reduce_mean(states_embedding_each_training_program, 0)
#             states_embedding_each_training_program = tf.reduce_sum(states_embedding_each_training_program, 0)                                
#             states_embedding_each_training_program = tf.reduce_logsumexp(states_embedding_each_training_program, 0)
            
            all_programs_tensors.append(summed_reduced_program_tensor)

        all_programs_tensors = tf.stack(all_programs_tensors, 0)
      
        prediction = tf.matmul(all_programs_tensors, W) + b                   
        correct_pred = tf.equal(tf.cast(tf.argmax(prediction,1), tf.int32), y)        
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))            
                
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=y))         
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999).minimize(loss)
        
        with tf.Session() as sess:            
            init = tf.global_variables_initializer()
            sess.run(init)
            
            for i in range(num_epochs):
                                      
                # Demo purpose only for one batch add a for loop to read multiple batches 
                _,_loss = sess.run(
        
                    [train_step, loss],

                    feed_dict={     
                        x : self.all_program_symbol_traces,       
                        y : self.labels,
                        seq_lengths : self.trace_lengths,
                    })

                # print("training iteration is %s and total_loss: %s "%(i,_loss))
                print("training epoch is %s and total_loss: %s "%(i,_loss))

            _accuracy = sess.run(
                
                accuracy,
                
                feed_dict={     
                        x : self.test_all_program_symbol_traces,       
                        y : self.test_labels,
                        seq_lengths : self.test_trace_lengths,
                    })
                
            print("The accuracy is %s"%(_accuracy))



if __name__ == '__main__':
    #训练集
    #traces量为100
    train_x = np.random.randint(10, 50, (100, 100))
    #labels为20，也就是代码片段数
    train_y = np.random.randint(0, 1, (20))
    #每个trace的长度
    train_length = np.random.randint(1, 10, (100))

    #测试集
    test_x = np.random.randint(10, 50, (100, 100))
    test_y = np.random.randint(0, 1, (20))
    test_length = np.random.randint(1, 10, (100))
    liger1 = Training(train_x, train_length, train_y, test_x, test_length, test_y)
    liger1.train_evaluate()
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
