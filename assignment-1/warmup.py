#Here are some libraries you're likely to use. You might want/need others as well.
import re
import sys
from random import random
from math import log
from collections import defaultdict
from functools import reduce
import math

# input all probs

probs=defaultdict(float)

probs['a##'] = 0.2
probs['b##'] = 0.8
probs['###'] = 0.0
probs['a#a'] = 0.2
probs['b#a'] = 0.7
probs['##a'] = 0.1
probs['a#b'] = 0.15
probs['b#b'] = 0.75
probs['##b'] = 0.1
probs['aaa'] = 0.4
probs['baa'] = 0.5
probs['#aa'] = 0.1
probs['aab'] = 0.6
probs['bab'] = 0.3
probs['#ab'] = 0.1
probs['aba'] = 0.25
probs['bba'] = 0.65
probs['aba'] = 0.1
probs['abb'] = 0.5
probs['bbb'] = 0.4
probs['#bb'] = 0.1



def entropy(w, N):
    n_grams = [w[i:i+N] for i in range(len(w)-N)]

    # convert each ngram to its corresponding probability
    w_probs = map(lambda ng: probs[ng], n_grams)

    return -1/N * math.log(reduce(lambda x, y: x*y, w_probs), 2)

def perplexity(w, N):
    n_grams = [w[i:i+N] for i in range(len(w)-N)]

    # convert each ngram to its corresponding probability
    w_probs = map(lambda ng: probs[ng], n_grams)

    # calculate term under sq root
    temp = 1/reduce(lambda x, y: x*y, w_probs) 
    return temp ** (1/N)

print(perplexity('##abaab#', 3))
print(2**entropy('##abaab#', 3))

tri_counts=defaultdict(int) #counts of all trigrams in input

#this function currently does nothing.
def preprocess_line(line):
    return line


#here we make sure the user provides a training filename when
#calling this program, otherwise exit with a usage error.
if len(sys.argv) != 2:
    print("Usage: ", sys.argv[0], "<training_file>")
    sys.exit(1)

infile = sys.argv[1] #get input argument: the training file

#This bit of code gives an example of how you might extract trigram counts
#from a file, line by line. If you plan to use or modify this code,
#please ensure you understand what it is actually doing, especially at the
#beginning and end of each line. Depending on how you write the rest of
#your program, you may need to modify this code.
with open(infile) as f:
    for line in f:
        line = preprocess_line(line) #doesn't do anything yet.
        for j in range(len(line)-(3)):
            trigram = line[j:j+3]
            tri_counts[trigram] += 1

#Some example code that prints out the counts. For small input files
#the counts are easy to look at but for larger files you can redirect
#to an output file (see Lab 1).
print("Trigram counts in ", infile, ", sorted alphabetically:")
for trigram in sorted(tri_counts.keys()):
    print(trigram, ": ", tri_counts[trigram])
print("Trigram counts in ", infile, ", sorted numerically:")
for tri_count in sorted(tri_counts.items(), key=lambda x:x[1], reverse = True):
    print(tri_count[0], ": ", str(tri_count[1]))


