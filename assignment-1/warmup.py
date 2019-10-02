# Here are some libraries you're likely to use. You might want/need others as well.
import re
import sys
from collections import defaultdict
import utils
# input all probs by hand :/

probs = defaultdict(float)

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

###############

print(utils.perplexity('##abaab#', 3, probs))
print(2**utils.entropy('##abaab#', 3, probs))
