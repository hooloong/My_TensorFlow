#-*- coding:utf-8 -*-
from io import open
import random
import sys
import tensorflow as tf

PAD = "PAD"
GO = "GO"
EOS = "EOS"
UNK = "UNK"
START_VOCAB = [PAD, GO, EOS, UNK]

PAD_ID = 0 #填充
GO_ID = 1  #开始标志
EOS_ID = 2 #结束标志
UNK_ID = 3 #未知字符
_buckets = [(10, 15), (20, 25), (40, 50),(80,100)]
units_num = 256
num_layers = 3
max_gradient_norm = 5.0
batch_size = 50
learning_rate = 0.5
learning_rate_decay_factor = 0.97

train_encode_file = "train_encode"
train_decode_file = "train_decode"
test_encode_file = "test_encode"
test_decode_file = "test_decode"
vocab_encode_file = "vocab_encode"
vocab_decode_file = "vocab_decode"
train_encode_vec_file = "train_encode_vec"
train_decode_vec_file = "train_decode_vec"
test_encode_vec_file = "test_encode_vec"
test_decode_vec_file = "test_decode_vec"

def is_chinese(sentence):
    flag = True
    if len(sentence) < 2:
        flag = False
        return flag
    for uchar in sentence:
        if(uchar == '，' or uchar == '。' or
            uchar == '～' or uchar == '?' or
            uchar == '！'):
            flag = True
        elif '一' <= uchar <= '鿿':
            flag = True
        else:
            flag = False
            break
    return flag

def get_chatbot():
    f = open("../../../data/chat.conv","r", encoding="utf-8")
    train_encode = open(train_encode_file,"w", encoding="utf-8")
    train_decode = open(train_decode_file,"w", encoding="utf-8")
    test_encode = open(test_encode_file,"w", encoding="utf-8")
    test_decode = open(test_decode_file,"w", encoding="utf-8")
    vocab_encode = open(vocab_encode_file,"w", encoding="utf-8")
    vocab_decode = open(vocab_decode_file,"w", encoding="utf-8")
    encode = list()
    decode = list()

    chat = list()
    print("start load source data...")
    step = 0
    for line in f.readlines():
        line = line.strip('\n').strip()
        if not line:
            continue
        if line[0] == "E":
            if step % 1000 == 0:
                print("step:%d" % step)
            step += 1
            if(len(chat) == 2 and is_chinese(chat[0]) and is_chinese(chat[1]) and
                not chat[0] in encode and not chat[1] in decode):
                encode.append(chat[0])
                decode.append(chat[1])
            chat = list()
        elif line[0] == "M":
            L = line.split(' ')
            if len(L) > 1:
                chat.append(L[1])
    encode_size = len(encode)
    if encode_size != len(decode):
        raise ValueError("encode size not equal to decode size")
    test_index = random.sample([i for i in range(encode_size)],int(encode_size*0.2))
    print("divide source into two...")
    step = 0
    for i in range(encode_size):
        if step % 1000 == 0:
            print("%d" % step)
        step += 1
        if i in test_index:
            test_encode.write(encode[i] + "\n")
            test_decode.write(decode[i] + "\n")
        else:
            train_encode.write(encode[i] + "\n")
            train_decode.write(decode[i] + "\n")

    vocab_encode_set = set(''.join(encode))
    vocab_decode_set = set(''.join(decode))
    print("get vocab_encode...")
    step = 0
    for word in vocab_encode_set:
        if step % 1000 == 0:
            print("%d" % step)
        step += 1
        vocab_encode.write(word + "\n")
    print("get vocab_decode...")
    step = 0
    for word in vocab_decode_set:
        print("%d" % step)
        step += 1
        vocab_decode.write(word + "\n")

def gen_chatbot_vectors(input_file,vocab_file,output_file):
    vocab_f = open(vocab_file,"r", encoding="utf-8")
    output_f = open(output_file,"w")
    input_f = open(input_file,"r",encoding="utf-8")
    words = list()
    for word in vocab_f.readlines():
        word = word.strip('\n').strip()
        words.append(word)
    word_to_id = {word:i for i,word in enumerate(words)}
    to_id = lambda word: word_to_id.get(word,UNK_ID)
    print("get %s vectors" % input_file)
    step = 0
    for line in input_f.readlines():
        if step % 1000 == 0:
            print("step:%d" % step)
        step += 1
        line = line.strip('\n').strip()
        vec = map(to_id,line)
        output_f.write(' '.join([str(n) for n in vec]) + "\n")

def get_vectors():
    gen_chatbot_vectors(train_encode_file,vocab_encode_file,train_encode_vec_file)
    gen_chatbot_vectors(train_decode_file,vocab_decode_file,train_decode_vec_file)
    gen_chatbot_vectors(test_encode_file,vocab_encode_file,test_encode_vec_file)
    gen_chatbot_vectors(test_decode_file,vocab_decode_file,test_decode_vec_file)

def get_vocabs(vocab_file):
    words = list()
    with open(vocab_file,"r", encoding="utf-8") as vocab_f:
        for word in vocab_f:
            words.append(word.strip('\n').strip())
    id_to_word = {i: word for i, word in enumerate(words)}
    word_to_id = {v: k for k, v in id_to_word.items()}
    vocab_size = len(id_to_word)
    return id_to_word,word_to_id,vocab_size

def read_data(source_path, target_path, max_size=None):
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set