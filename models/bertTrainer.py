"""bertTrainer.py
    File intent- create a script that runs the bert model with different inputs to find the optimal parameters
"""

import bertModel 


### parameters
trials = 1

batch_size_step = 2
init_batch_size = 8

max_len_step = 2
init_max_len = 16

epoch_step = 5
epoch_count = 10

results = []

for trial in trials:
    
    results.append(bertModel())