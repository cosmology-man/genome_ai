# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 00:18:54 2023

@author: ashaa
"""

import random
import matplotlib.pyplot as plt
import time
start_time = time.time()
import numpy as np

def random_sequence(length):
    sequence = np.random.choice([-3, 1, -1, 3], size=length)
    return sequence


def double_cut(sequence):
    
    tmp = sequence
    
    for i in range(10):
        
        cut_index = random.randint(1, len(sequence)-1)
        
        bruh = tmp[0:cut_index]
        moment = tmp[cut_index:]
        tmp = (np.append(moment,bruh, axis=None)) 
        
    return tmp











def replace_chunks(seq):
    # randomly choose the percentage of sequence to modify
    pct_modified = np.random.uniform(0.25, 0.95)

    # calculate the number of changes to make and the average length of chunks
    seq_len = len(seq)
    avg_chunk_len = int(seq_len * pct_modified / np.random.randint(1, 10))
    num_changes = int(seq_len * pct_modified / avg_chunk_len)

    # generate the start indices of each chunk to be modified
    max_start_idx = seq_len - num_changes * avg_chunk_len
    chunk_start_indices = np.random.randint(0, max_start_idx+1, size=num_changes)

    # generate the new chunks to replace the original chunks
    new_chunks = np.random.choice([-3, -1, 3, 1], size=num_changes*avg_chunk_len).reshape(num_changes, avg_chunk_len)

    # modify the original sequence with the new chunks
    modified_seq = seq.copy()
    for i in range(num_changes):
        modified_seq[chunk_start_indices[i]:chunk_start_indices[i]+avg_chunk_len] = new_chunks[i]

    return modified_seq#double_cut(modified_seq)













"""

def replace_chunks(sequence):
    sequence_length = len(sequence)

    while True:
        # Determine the percentage of the sequence to replace
        replace_percentage_factor = random.uniform(0.0, 1.0)
        replace_length = int(sequence_length * replace_percentage_factor)

        # Determine the number of chunks to replace
        max_chunks = replace_length // 2
        num_chunks_factor = random.uniform(0.5, 1.5)
        if max_chunks == 0:
            num_chunks = 0
        else:
            num_chunks = int(num_chunks_factor * (replace_length // max_chunks))

        # Determine the maximum length of each chunk
        max_chunk_length_factor = random.uniform(0.5, 1.5)
        max_chunk_length = int(max_chunk_length_factor * (replace_length // num_chunks)) if num_chunks > 0 else 0

        # Create a copy of the sequence
        new_sequence = sequence.copy()

        changes_made = False

        for i in range(num_chunks):
            # Determine the maximum length of the current chunk
            chunk_length_factor = random.uniform(0.5, 1.5)
            max_chunk_length_cur = int(chunk_length_factor * max_chunk_length) if max_chunk_length > 0 else 0

            # Determine the start index of the current chunk
            max_chunk_start_index = sequence_length - replace_length
            if max_chunk_start_index <= 0:
                break
            start_index = random.randint(0, max_chunk_start_index)

            # Determine the end index of the current chunk
            end_index = start_index + max_chunk_length_cur

            # Generate a new random sequence of the same length as the chunk
            new_length = end_index - start_index
            new_chunk = random_sequence(new_length)

            # Insert the new sequence into the original sequence and delete the old chunk
            new_sequence = sequence[:start_index] + new_chunk + sequence[end_index:]
            sequence_length = len(new_sequence)

            # Determine the distance to the next chunk
            chunk_distance_factor = random.uniform(0.5, 1.5)
            max_chunk_distance_cur = int(chunk_distance_factor * max_chunk_length_cur) if max_chunk_length_cur > 0 else 0

            # Determine the starting index for the next chunk
            start_index += max_chunk_distance_cur

            changes_made = True

        if changes_made:
            return new_sequence

"""













"""
def replace_chunks(sequence):
    sequence_length = len(sequence)

    # Determine the percentage of the sequence to replace
    replace_percentage_ranges = [
        (0.05, 0.5)
    ]
    replace_percentage_min, replace_percentage_max = random.choice(replace_percentage_ranges)
    replace_percentage_factor = random.uniform(replace_percentage_min, replace_percentage_max)
    replace_length = int(sequence_length * replace_percentage_factor)

    # Determine the number of chunks to replace
    max_chunks = replace_length // 2
    num_chunks_factor = random.uniform(0.5, 1.5)
    num_chunks = int(num_chunks_factor * (replace_length // max_chunks))

    # Determine the maximum length of each chunk
    max_chunk_length_factor = random.uniform(0.5, 1.5)
    max_chunk_length = int(max_chunk_length_factor * (replace_length // num_chunks))

    # Create a copy of the sequence
    new_sequence = sequence.copy()

    for i in range(num_chunks):
        # Determine the maximum length of the current chunk
        chunk_length_factor = random.uniform(0.5, 1.5)
        max_chunk_length_cur = int(chunk_length_factor * max_chunk_length)

        # Determine the start index of the current chunk
        max_chunk_start_index = replace_length - max_chunk_length_cur
        if max_chunk_start_index <= 0:
            break
        start_index = random.randint(0, max_chunk_start_index)

        # Determine the end index of the current chunk
        end_index = start_index + max_chunk_length_cur

        # Generate a new random sequence of the same length as the chunk
        new_length = end_index - start_index
        new_chunk = generate_random_sequence(new_length)

        # Replace the chunk with the new sequence
        new_sequence[start_index:end_index] = new_chunk

        # Determine the distance to the next chunk
        chunk_distance_factor = random.uniform(0.5, 1.5)
        max_chunk_distance_cur = int(chunk_distance_factor * max_chunk_length_cur)

        # Determine the starting index for the next chunk
        start_index += max_chunk_distance_cur

    return new_sequence
"""







def similarity_percentage(original_sequence, replaced_sequence):
    if len(original_sequence) != len(replaced_sequence):
        raise ValueError('Sequences must be of equal length.')
    
    num_similar = sum([1 for i in range(len(original_sequence)) if original_sequence[i] == replaced_sequence[i]])
    
    return (num_similar / len(original_sequence)) * 100

"""
plt.figure()
seq = random_sequence(10000)
mod = replace_chunks(seq)

plt.plot(seq)
plt.plot(mod)
plt.show()"""

out_sequence = np.zeros((500000, 10000), dtype = np.int16)
out_modified_sequence = np.zeros((500000, 10000), dtype = np.int16)
out_rando = np.zeros((500000, 10000), dtype = np.int16)
out_other_rando = np.zeros((500000, 10000), dtype = np.int16)
out_similarity = np.zeros((500000), dtype = np.int16)
for i in range(500000):
        rando = random_sequence(10000)
        other_rando = random_sequence(10000)
        og = random_sequence(10000)
        modified_og = double_cut(replace_chunks(og))
        similarity = similarity_percentage(og, replace_chunks(modified_og))
        
        out_sequence[i]=og
        out_modified_sequence[i] = modified_og
        out_rando[i] = rando
        out_other_rando[i] = other_rando
        out_similarity[i] = similarity
        
np.save('/users/ashaa/code/genome_ai/sequence_1.npy', out_sequence)
np.save('/users/ashaa/code/genome_ai/changed_sequence_1.npy', out_modified_sequence)
np.save('/users/ashaa/code/genome_ai/rando_1.npy', out_rando)
np.save('/users/ashaa/code/genome_ai/other_rando_1.npy', out_rando)
np.save('/users/ashaa/code/genome_ai/out_similarity_1.npy', out_similarity)

"""
out_sequence = []
out_modified_sequence = []
out_random = []

for i in range(1500000):
    rando = random_sequence(10000)
    seq = random_sequence(10000)
    out_sequence.append(seq) 
    out_modified_sequence.append(replace_chunks(seq))
    out_random.append(rando)

np.save('/users/ashaa/code/genome_ai/sequence.npy', out_sequence)
np.save('/users/ashaa/code/genome_ai/changed_sequence.npy', out_modified_sequence)
np.save('/users/ashaa/code/random_sequence_compare.npy', out_random)

sequence = np.load('/users/ashaa/code/genome_ai/sequence.npy')
changed_sequence = np.load('/users/ashaa/code/genome_ai/changed_sequence.npy')
#for i in range(len(sequence)):
#    print(similarity_percentage(sequence[i], changed_sequence[i]))
    """
print("My program took", time.time() - start_time, "to run")









