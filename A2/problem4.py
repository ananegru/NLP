



"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""


import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import random
import codecs

#### Question 4-------------------------------------------------------------------
def map_words_to_indices(input_file, output_file):
    word_index_dict = {}
    #read brown_vocab_100.txt into word_index_dict
    with open(input_file, 'r') as f:
        for line_index, line in enumerate(f):
            word = line.rstrip()
            #if word != '<s>' or word != '</s>':
            word_index_dict[word] = str(line_index)
    
    #write word_index_dict to word_to_index_100.txt
    with open(output_file, 'w') as wf:
        wf.write(str(word_index_dict))
    return word_index_dict

input_file = 'brown_vocab_100.txt'
output_file = 'word_to_index_100.txt'
word_index_dict = map_words_to_indices(input_file, output_file)

def initialize_matrix_of_counts_smoothed(file, word_index_dict, alpha):
    V = len(word_index_dict)
    counts = np.zeros((V, V))
    previous_word = '<s>'
    # sentences = []
    total_count = 0
    for line in file:
        words = [word.lower() for word in line.strip().split()]
        for word in words:
            # print(word)
            if word in word_index_dict:
                #increment counts for combination of word and previous word
                counts[int(word_index_dict[previous_word]), int(word_index_dict[word])] += 1
                total_count += 1
                #set previous_word to current word
                previous_word = word
    # Add α to every cell of the counts matrix
    counts += alpha
    return counts

def write_out_probabilities(probs):
    prob_the_all = probs[int(word_index_dict['all']), int(word_index_dict['the'])]
    prob_jury_the = probs[int(word_index_dict['the']), int(word_index_dict['jury'])]
    prob_campaign_the = probs[int(word_index_dict['the']), int(word_index_dict['campaign'])]
    prob_calls_anonymous = probs[int(word_index_dict['anonymous']), int(word_index_dict['calls'])]

    # Write probabilities to the file
    with open("smooth_probs.txt", "w") as f:
        f.write(f"p(the | all) = {prob_the_all}\n")
        f.write(f"p(jury | the) = {prob_jury_the}\n")
        f.write(f"p(campaign | the) = {prob_campaign_the}\n")
        f.write(f"p(calls | anonymous) = {prob_calls_anonymous}\n")

def build_smoothed_mle_bigram(file, word_index_dict, alpha):
    counts = initialize_matrix_of_counts_smoothed(file, word_index_dict, alpha)
    probs = normalize(counts, norm='l1', axis=1)
    write_out_probabilities(probs)
    return probs

alpha = 0.1
file = open("brown_100.txt")
smooth_probs = build_smoothed_mle_bigram(file, word_index_dict, alpha)


#### Question 6-------------------------------------------------------------------
def calculate_sentence_probability(corpus, output_file, probs):
    # Iterate through each sentence in the toy corpus
    for sentence in corpus:
        # Split the sentence into words and calculate the joint probability
        words = sentence.strip().lower().split()
        sent_prob = 1
        previous_word = '<s>'
        for word in words:
            if word != '</s>':
                if word in word_index_dict:
                    word_idx = int(word_index_dict[word])
                    previous_word_idx = int(word_index_dict[previous_word])
                    word_prob = probs[previous_word_idx, word_idx]
                    sent_prob *= word_prob
                    previous_word = word

        #Overwrite output with perplexity of the sentence
        sent_len = len(words) + 1 # Include the end-of-sentence token
        perplexity = 1 / (pow(sent_prob, 1.0 / sent_len))
        output_file.write(str(perplexity) + "\n")

output_file = open("smoothed_eval.txt", "w")
toy_corpus = open("toy_corpus.txt")
calculate_sentence_probability(toy_corpus, output_file, smooth_probs)


#### Question 7-------------------------------------------------------------------
# Convert string keys to integers
word_index_dict = {value: int(key) for value, key in word_index_dict.items()}
def generate_sentences(amount, file_name, word_index_dict):
    with open(file_name, "w") as file:
        # Loop to generate 10 sentences
        for i in range(amount):
            generated_sentence = GENERATE(word_index_dict, smooth_probs, "bigram", 10, "the")
            # Write the generated sentence to the file
            file.write("Generated Sentence " + str(i + 1) + ": " + generated_sentence + "\n")
generate_sentences(10, 'smoothed_generation.txt', word_index_dict)


