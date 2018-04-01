#-*- coding:utf-8 -*-
import generate_chat
import seq2seq_model
import tensorflow as tf
import numpy as np
import logging
import logging.handlers

if __name__ == '__main__':

    _,_,source_vocab_size = generate_chat.get_vocabs(generate_chat.vocab_encode_file)
    _,_,target_vocab_size = generate_chat.get_vocabs(generate_chat.vocab_decode_file)
    train_set = generate_chat.read_data(generate_chat.train_encode_vec_file,generate_chat.train_decode_vec_file)
    test_set = generate_chat.read_data(generate_chat.test_encode_vec_file,generate_chat.test_decode_vec_file)
    train_bucket_sizes = [len(train_set[i]) for i in range(len(generate_chat._buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in range(len(train_bucket_sizes))]
    with tf.Session() as sess:
        model = seq2seq_model.Seq2SeqModel(source_vocab_size,
            target_vocab_size,
            generate_chat._buckets,
            generate_chat.units_num,
            generate_chat.num_layers,
            generate_chat.max_gradient_norm,
            generate_chat.batch_size,
            generate_chat.learning_rate,
            generate_chat.learning_rate_decay_factor,
            use_lstm = True)
        ckpt = tf.train.get_checkpoint_state('.')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())
        loss = 0.0
        step = 0
        previous_losses = []
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, False)
            print("step:%d,loss:%f" % (step,step_loss))
            loss += step_loss / 2000
            step += 1
            if step % 1000 == 0:
                print("step:%d,per_loss:%f" % (step,loss))
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                model.saver.save(sess, "./chatbot.ckpt", global_step=model.global_step)
                loss = 0.0
            if step % 5000 == 0:
                for bucket_id in range(len(generate_chat._buckets)):
                    if len(test_set[bucket_id]) == 0:
                        continue
                        encoder_inputs, decoder_inputs, target_weights = model.get_batch(test_set, bucket_id)
                        _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
                        print("bucket_id:%d,eval_loss:%f" % (bucket_id,eval_loss))