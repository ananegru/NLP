#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from generate import GENERATE

#### Question 2-------------------------------------------------------------------
def map_words_to_indices(input_file, output_file):
    word_index_dict = {}
    #read brown_vocab_100.txt into word_index_dict
    with open(input_file, 'r') as f:
        for line_index, line in enumerate(f):
            word = line.rstrip()
            #if word not in ("<s>", "</s>"):
            word_index_dict[word] = str(line_index)

    # write word_index_dict to word_to_index_100.txt
    with open(output_file, 'w') as wf:
        wf.write(str(word_index_dict))
    return word_index_dict

input_file = 'brown_vocab_100.txt'
output_file = 'word_to_index_100.txt'
word_index_dict = map_words_to_indices(input_file, output_file)


def initialize_vector_of_counts(file, word_index_dict):
    sentences = []
    for line in file:
        words = [word.lower() for word in line.strip().split()]
        sentences.append(words)
    #initialize the numpy vector of counts with zeros.
    length_counts = len(word_index_dict)
    counts = np.zeros(length_counts)

    #Iterate through the sentences and increment counts for each of the words they contain
    for sentence in sentences:
        for word in sentence:            
            if word in word_index_dict:
                count_idx = int(word_index_dict[word])
                counts[count_idx] += 1

    return counts

def build_mle_unigram(file, word_index_dict):
    counts = initialize_vector_of_counts(file, word_index_dict)
    #Normalize Counts and create Probabilities
    probs = counts / np.sum(counts)
    np.savetxt("unigram_probs.txt", probs)
    return probs

file = open("brown_100.txt")
probs = build_mle_unigram(file, word_index_dict)


#Check desired output
probs_file = np.loadtxt("unigram_probs.txt")
print(probs_file[0])
print(probs_file[-1])

#### Question 6-------------------------------------------------------------------
def calculate_sentence_probability(corpus, output_file):
    # Iterate through each sentence in the toy corpus
    for sentence in corpus:
        # Split the sentence into words and calculate the joint probability
        words = sentence.strip().lower().split()
        sent_prob = 1
        for word in words:
            if word in word_index_dict:
                index = int(word_index_dict[word])
                word_prob = probs_file[index]
                sent_prob *= word_prob
        # Write the joint probability of the sentence to the output file
        # output_file.write(str(sent_prob) + "\n")

        #Overwrite output with perplexity of the sentence
        sent_len = len(words)  # Include the end-of-sentence token
        perplexity = 1 / (pow(sent_prob, 1.0 / sent_len))
        output_file.write(str(perplexity) + "\n")

output_file = open("unigram_eval.txt", "w")
toy_corpus = open("toy_corpus.txt")
calculate_sentence_probability(toy_corpus, output_file)


#### Question 7-------------------------------------------------------------------
# # # Convert string keys to integers
word_index_dict = {value: int(key) for value, key in word_index_dict.items()}
def generate_sentences(amount, file_name, word_index_dict):
    with open(file_name, "w") as file:
        # Loop to generate 10 sentences
        for i in range(amount):
            generated_sentence = GENERATE(word_index_dict, probs, "unigram", 10, "<s>")
            # Write the generated sentence to the file
            file.write("Generated Sentence " + str(i + 1) + ": " + generated_sentence + "\n")
generate_sentences(10, 'unigram_generation.txt', word_index_dict)


