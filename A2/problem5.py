import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import random
import codecs

#### Question 5-------------------------------------------------------------------
def map_words_to_indices(input_file, output_file):
    word_index_dict = {}
    with open(input_file, 'r') as f:
        for line_index, line in enumerate(f):
            word = line.rstrip()
            word_index_dict[word] = str(line_index)
    with open(output_file, 'w') as wf:
        wf.write(str(word_index_dict))
    return word_index_dict

input_file = 'brown_vocab_100.txt'
output_file = 'word_to_index_100.txt'
word_index_dict = map_words_to_indices(input_file, output_file)

def initialize_matrix_of_counts_smoothed(file, word_index_dict, V):
    V = len(word_index_dict)
    # Update counts to include trigram counts
    counts = np.zeros((V, V, V))
    previous_word_1 = '<s>'
    previous_word_2 = '<s>'
    total_count = 0
    for line in file:
        words = [word.lower() for word in line.strip().split()]
        for word in words:
            if word in word_index_dict:
                # Increment counts for combination of word, previous_word_1, and previous_word_2
                counts[int(word_index_dict[previous_word_2]), int(word_index_dict[previous_word_1]), int(word_index_dict[word])] += 1
                total_count += 1
                # Set previous_word_2 to previous_word_1 and previous_word_1 to current word
                previous_word_2 = previous_word_1
                previous_word_1 = word
    return counts

def calculate_probabilities(counts, smoothing, V):
    # Calculate probabilities manually
    probs = np.zeros((V, V, V))
    for i in range(V):
        for j in range(V):
            # Sum the counts for each word combination to get the total count for the current trigram
            total_count = np.sum(counts[i, j, :])
            if smoothing == 'unsmoothed':
                if total_count > 0:
                    # Normalize the counts to probabilities
                    probs[i, j, :] = (counts[i, j, :]) / (total_count)
            if smoothing == 'smoothed':
                if total_count > 0:
                    # Normalize the counts to probabilities
                    probs[i, j, :] = (counts[i, j, :] + alpha) / (total_count + alpha * V)
    return probs 

def write_out_probabilities(probs, outputfile):
    # Get the indices of the words "in", "the", and "time" from the word_index_dict
    in_word = word_index_dict["in"]
    the = word_index_dict["the"]
    past = word_index_dict["past"]
    time = word_index_dict["time"]
    said = word_index_dict["said"]
    jury = word_index_dict["jury"]
    recommended = word_index_dict["recommended"]
    that = word_index_dict["that"]
    agriculture = word_index_dict["agriculture"]
    komma = word_index_dict[","]
    teacher = word_index_dict["teacher"]

    p_past_given_in_the = probs[int(in_word), int(the), int(past)]
    p_time_given_in_the = probs[int(in_word), int(the), int(time)]
    p_said_given_the_jury = probs[int(the), int(jury), int(said)]
    p_recommended_given_the_jury = probs[int(the), int(jury), int(recommended)]
    p_that_given_jury_said = probs[int(jury), int(said), int(that)]
    p_komma_given_teacher_agriculture= probs[int(teacher), int(agriculture), int(komma)]

    # Write probabilities to the file
    with open(outputfile, "w") as f:
        f.write(f"p(past | in, the) = {p_past_given_in_the}\n")
        f.write(f"p(time | in, the) = {p_time_given_in_the}\n")
        f.write(f"p(said | the, jury) = {p_said_given_the_jury}\n")
        f.write(f"p(recommended | the, jury) = {p_recommended_given_the_jury}\n")
        f.write(f"p(that | jury, said) = {p_that_given_jury_said}\n")
        f.write(f"p(, | agriculture, teacher) = {p_komma_given_teacher_agriculture}\n")

    return p_past_given_in_the


def build_mle_trigram(file, word_index_dict, alpha):
    V = len(word_index_dict)
    counts = initialize_matrix_of_counts_smoothed(file, word_index_dict, V)
    probs = calculate_probabilities(counts, 'unsmoothed', V)
    probs_smoothed = calculate_probabilities(counts, 'smoothed', V)
    p_past_given_in_the = write_out_probabilities(probs, 'trigram_probs.txt')
    smoothed_p_past_given_in_the = write_out_probabilities(probs_smoothed, 'smoothed_trigram_probs.txt')
    
    #Check desired output:
    print("Unsmoothed: Probability of 'time' given 'in' and 'the':", p_past_given_in_the)
    print("Smoothed: Probability of 'time' given 'in' and 'the':", smoothed_p_past_given_in_the)

alpha = 0.1
file = open("brown_100.txt")
build_mle_trigram(file, word_index_dict, alpha)
