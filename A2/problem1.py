#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

### Create a dictionary that maps words to indices which lists each word in the corpus exactly once.

def map_words_to_indices(input_file, output_file):
    word_index_dict = {}
    #read brown_vocab_100.txt into word_index_dict
    with open(input_file, 'r') as f:
        for line_index, line in enumerate(f):
            word = line.rstrip()
            word_index_dict[word] = str(line_index)

    #write word_index_dict to word_to_index_100.txt
    with open(output_file, 'w') as wf:
        wf.write(str(word_index_dict))

    print(word_index_dict['all'])
    print(word_index_dict['resolution'])
    print(len(word_index_dict))


input_file = 'brown_vocab_100.txt'
output_file = 'word_to_index_100.txt'
map_words_to_indices(input_file, output_file)



def map_words_to_order(corpus):
    word_order_map = {}  # dictionary to store word-order mapping
    sentence_start = "<s>"
    sentence_end = "</s>"
    sentence_delimiter = " "
    sentences = corpus.split(sentence_end)  # split corpus into sentences

    for sentence in sentences:
        sentence = sentence.strip()  # remove leading/trailing whitespaces
        if sentence.startswith(sentence_start):
            sentence = sentence[len(sentence_start):]  # remove sentence start tag
        if sentence.endswith(sentence_delimiter):
            sentence = sentence[:-len(sentence_delimiter)]  # remove sentence delimiter
        words = sentence.split(sentence_delimiter)  # split sentence into words

        for i, word in enumerate(words):
            if word not in word_order_map:
                word_order_map[word] = len(word_order_map) + 1  # assign next order to new word

    return word_order_map

