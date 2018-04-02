#-*- coding:utf-8 -*-
from generate_poetry import Poetry
from poetry_model import poetryModel
from operator import itemgetter
import tensorflow as tf
import numpy as np
import random


if __name__ == '__main__':
    batch_size = 1
    rnn_size = 128
    num_layers = 2
    poetrys = Poetry()
    words_size = len(poetrys.word_to_id)

    def to_word(prob):
        prob = prob[0]
        indexs, _ = zip(*sorted(enumerate(prob), key=itemgetter(1)))
        rand_num = int(np.random.rand(1)*10);
        index_sum = len(indexs)
        max_rate = prob[indexs[(index_sum-1)]]
        if max_rate > 0.9 :
            sample = indexs[(index_sum-1)]
        else:
            sample = indexs[(index_sum-1-rand_num)]
        return poetrys.id_to_word[sample]

    inputs = tf.placeholder(tf.int32, [batch_size, None])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    model = poetryModel()
    logits,probs,initial_state,last_state = model.create_model(inputs,batch_size,
                                                               rnn_size,words_size,num_layers,False,keep_prob)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,"poetry_model.ckpt")
        next_state = sess.run(initial_state)

        x = np.zeros((1, 1))
        x[0,0] = poetrys.word_to_id['[']
        feed = {inputs: x, initial_state: next_state, keep_prob: 1}
        predict, next_state = sess.run([probs, last_state], feed_dict=feed)
        word = to_word(predict)
        poem = ''
        while word != ']':
            poem += word
            x = np.zeros((1, 1))
            x[0, 0] = poetrys.word_to_id[word]
            feed = {inputs: x, initial_state: next_state, keep_prob: 1}
            predict, next_state = sess.run([probs, last_state], feed_dict=feed)
            word = to_word(predict)
        print (poem)