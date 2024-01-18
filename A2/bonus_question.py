# Run time 13.1s
# Import libraries
import nltk
from nltk.corpus import brown
from math import log2

# Retrieve all bigrams in the Brown corpus using the 'nltk.bigrams' function
brown_bigrams = list(nltk.bigrams(brown.words()))

# Count the frequency of each word and each bigram in the corpus using the 'nltk.FreqDist' function
word_freq = nltk.FreqDist(brown.words()) # C(w) is the absolute frequency
bigram_freq = nltk.FreqDist(brown_bigrams)

# Define the minimum frequency of a word
#min_freq = 30 # This produces a different result from the current one with min 10
min_freq = 10 # The minimum requested from the assignment

# Calculate the Pointwise Mutual Information (PMI) for each bigram
# Loop over each bigram in the corpus and calculate the PMI if both words in the bigram meet the frequency boundary
pmi = {}
N = len(brown_bigrams)
for bigram in bigram_freq: # Equation for PMI 
    w1, w2 = bigram # Successive pairs (w1, w2) of words in the Brown corpus
    if word_freq[w1] >= min_freq and word_freq[w2] >= min_freq:
        p_w1 = word_freq[w1] / N # N is the size of the corpus
        p_w2 = word_freq[w2] / N
        p_w1w2 = bigram_freq[bigram] / N
        pmi[bigram] = log2(p_w1w2 / (p_w1 * p_w2))

# Print the top 20 and bottom 20 pairs based on PMI value 
# Sort the bigrams by PMI value and print the top 20 and bottom 20 pairs. Print in descending order by using 'reverse:True'
top_20_pairs = sorted(pmi.items(), key=lambda x: x[1], reverse=True)[:20]
bottom_20_pairs = sorted(pmi.items(), key=lambda x: x[1])[:20]

# Print top 20: in this list, most words are common and used by humans
print("The top 20 word pairs with highest PMI value:")
for pair, value in top_20_pairs:
    print(pair[0], pair[1], value)

#output:
#The top 20 word pairs with highest PMI value:
#Hong Kong 16.687742245964497
#Viet Nam 16.147173864601793
#Simms Purdew 16.059711023351454
#Pathet Lao 16.059711023351454
#El Paso 15.825245769714432
#7th Cavalry 15.825245769714432
#Herald Tribune 15.79465744988101
#o Shu 15.754856441823033
#WTV antigen 15.66912656779715
#Gray Eyes 15.632600691772035
#Puerto Rico 15.562211363880639
#Internal Revenue 15.562211363880639
#decomposition theorem 15.33981894254419
#Saxon Shore 15.314283850437052
#anionic binding 15.310672596884674
#Export-Import Bank 15.289192869474222
#carbon tetrachloride 15.261816430542916
#unwed mothers 15.240283268993275
#Common Market 15.240283268993275
#Beverly Hills 15.224341725124255

# Print bottom 20: in this list, it contains grammar mistakes mostly and punctuation signs 
print("\nThe top Bottom 20 word pairs with lowest PMI value:")
for pair, value in bottom_20_pairs:
    print(pair[0], pair[1], value)

#output:
#The top Bottom 20 word pairs with lowest PMI value:
#. , -11.275521042744144
#the . -10.379948574066933
#and . -10.212224594789445
#of of -10.130649571683692
#the in -10.043150535330767
#a . -9.860862711997012
#the , -9.621352433965733
#and and -9.390332006733546
#the is -9.07860152224684
#the and -8.973093485289878
#of to -8.643013826887588
#, ; -8.12730071235218
#the I -8.122737965627413
#of for -8.101731699386105
#? the -7.985597795371627
#the not -7.900112577242017
#to was -7.759278246204564
#of he -7.67253688811393
#he of -7.67253688811393
#in of -7.660622837807196