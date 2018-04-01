#-*- coding:utf-8 -*-
import generate_chat
import seq2seq_model
import tensorflow as tf
import numpy as np
import sys

if __name__ == '__main__':
    source_id_to_word,source_word_to_id,source_vocab_size = generate_chat.get_vocabs(generate_chat.vocab_encode_file)
    target_id_to_word,target_word_to_id,target_vocab_size = generate_chat.get_vocabs(generate_chat.vocab_decode_file)
    to_id = lambda word: source_word_to_id.get(word,generate_chat.UNK_ID)
    with tf.Session() as sess:
        model = seq2seq_model.Seq2SeqModel(source_vocab_size,
                                           target_vocab_size,
                                           generate_chat._buckets,
                                           generate_chat.units_num,
                                           generate_chat.num_layers,
                                           generate_chat.max_gradient_norm,
                                           1,
                                           generate_chat.learning_rate,
                                           generate_chat.learning_rate_decay_factor,
                                           forward_only = True,
                                           use_lstm = True)
        model.saver.restore(sess,"chatbot.ckpt-317000")
        while True:
            sys.stdout.write("ask > ")
            sys.stdout.flush()
            sentence = sys.stdin.readline().strip('\n')
            flag = generate_chat.is_chinese(sentence)
            if not sentence or not flag:
              print("请输入纯中文")
              continue
            sentence_vec = list(map(to_id,sentence))
            bucket_id = len(generate_chat._buckets) - 1
            if len(sentence_vec) > generate_chat._buckets[bucket_id][0]:
                print("sentence too long max:%d" % generate_chat._buckets[bucket_id][0])
                exit(0)
            for i,bucket in enumerate(generate_chat._buckets):
                if bucket[0] >= len(sentence_vec):
                    bucket_id = i
                    break
            encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(sentence_vec, [])]}, bucket_id)
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, True)
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            if generate_chat.EOS_ID in outputs:
                outputs = outputs[:outputs.index(generate_chat.EOS_ID)]
            answer = "".join([tf.compat.as_str(target_id_to_word[output]) for output in outputs])
            print("answer > " + answer)