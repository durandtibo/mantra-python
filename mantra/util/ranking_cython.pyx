
import cython
cimport cython

import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
# DTYPE = np.int32
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int32_t DTYPE_t

#@cython.boundscheck(False) # turn off bounds-checking for entire function
def ssvm_ranking_feature_map(np.ndarray[np.float64_t, ndim=2] patterns, np.ndarray[DTYPE_t, ndim=1] labels, np.ndarray[DTYPE_t, ndim=1] ranking):

    cdef int feature_dim = patterns.shape[1]
    cdef int number_of_examples = patterns.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] count = np.zeros(number_of_examples, dtype=np.int32)
    cdef int i,j,y_ij
    for i in range(number_of_examples):
        if labels[i] == 1:
            for j in range(number_of_examples):
                if labels[j] == 0:
                    y_ij = -1
                    if ranking[i] > ranking[j]:
                        y_ij = 1
                    if y_ij == 1:
                        count[i] += 1
                        count[j] -= 1
                    else:
                        count[i] -= 1
                        count[j] += 1

    cdef np.ndarray[np.float64_t, ndim=1] psi = np.zeros(feature_dim, dtype=np.float64)
    for i in range(number_of_examples):
        psi += count[i] * patterns[i]
    # psi /= float(x.num_pos * x.num_neg)
    return psi


#@cython.boundscheck(False)
def find_optimum_neg_locations_cython(int num_pos, int num_neg, np.ndarray[DTYPE_t, ndim=1] labels, np.ndarray[np.float64_t, ndim=1] positive_example_score, np.ndarray[np.float64_t, ndim=1] negative_example_score, np.ndarray[np.uint32_t, ndim=1] example_index_map):

    cdef float max_value = 0.0
    cdef float current_value = 0.0
    cdef int max_index = -1
    cdef np.ndarray[np.uint32_t, ndim=1] optimum_loc_neg_example = np.zeros(num_neg, dtype=np.uint32)
    cdef int j,k
    cdef float num_pos_f = float(num_pos)
    cdef float num_neg_f = float(num_neg)

    # for every jth negative image
    for j in range(1, num_neg+1):
        max_value = 0
        max_index = num_pos + 1
        # k is what we are maximising over. There would be one k_max for each negative image j
        current_value = 0
        for k in reversed(range(1, num_pos+1)):
            current_value += (1.0 / num_pos_f) * ((1.0*j / (j + k)) - ((j - 1.0) / (j + k - 1.0))) - (2.0 / (num_pos_f * num_neg_f)) * (positive_example_score[k-1] - negative_example_score[j-1])
            if current_value > max_value:
                max_value = current_value
                max_index = k
            optimum_loc_neg_example[j-1] = max_index

    ranking = encode_ranking_cython(labels, positive_example_score, negative_example_score, example_index_map, optimum_loc_neg_example)
    return ranking


#@cython.boundscheck(False)
def encode_ranking_cython(np.ndarray[DTYPE_t, ndim=1] labels, np.ndarray[np.float64_t, ndim=1] positive_example_score, np.ndarray[np.float64_t, ndim=1] negative_example_score, np.ndarray[np.uint32_t, ndim=1] example_index_map, np.ndarray[np.uint32_t, ndim=1] optimum_loc_neg_example):

    cdef int number_of_examples = labels.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] ranking = np.zeros(number_of_examples, np.int32)
    cdef int i, j, i_prime, j_prime, oi_prime, oj_prime

    for i in range(number_of_examples):
        for j in range(i+1, number_of_examples):
            if i == j:
                # do nothing
                number_of_examples
            elif labels[i] == labels[j]:
                if labels[i] == 1:
                    if positive_example_score[example_index_map[i]] > positive_example_score[example_index_map[j]]:
                        ranking[i] += 1
                        ranking[j] -= 1
                    elif positive_example_score[example_index_map[j]] > positive_example_score[example_index_map[i]]:
                        ranking[i] -= 1
                        ranking[j] += 1
                    else:
                        if i < j:
                            ranking[i] += 1
                            ranking[j] -= 1
                        else:
                            ranking[i] -= 1
                            ranking[j] += 1

                else:
                    if negative_example_score[example_index_map[i]] > negative_example_score[example_index_map[j]]:
                        ranking[i] += 1
                        ranking[j] -= 1
                    elif negative_example_score[example_index_map[j]] > negative_example_score[example_index_map[i]]:
                        ranking[i] -= 1
                        ranking[j] += 1
                    else:
                        if i < j:
                            ranking[i] += 1
                            ranking[j] -= 1
                        else:
                            ranking[i] -= 1
                            ranking[j] += 1

            elif labels[i] == 1 and labels[j] == 0:
                i_prime = example_index_map[i] + 1
                j_prime = example_index_map[j] + 1
                oj_prime = optimum_loc_neg_example[j_prime-1]

                if (oj_prime - i_prime - 0.5) > 0:
                    ranking[i] += 1
                    ranking[j] -= 1
                else:
                    ranking[i] -= 1
                    ranking[j] += 1

            elif labels[i] == 0 and labels[j] == 1:
                i_prime = example_index_map[i] + 1
                j_prime = example_index_map[j] + 1
                oi_prime = optimum_loc_neg_example[i_prime - 1]

                if (j_prime - oi_prime + 0.5) > 0:
                    ranking[i] += 1
                    ranking[j] -= 1
                else:
                    ranking[i] -= 1
                    ranking[j] += 1

    return ranking


def average_precision_cython(np.ndarray[DTYPE_t, ndim=1] gt_labels, np.ndarray[np.float64_t, ndim=1] scores):

    cdef int i, j, label
    cdef int number_of_examples = gt_labels.shape[0]

    # Stores rank of all examples
    cdef np.ndarray[DTYPE_t, ndim=1] ranking = np.zeros(number_of_examples, dtype=np.int32)
    # Stores list of images sorted by rank. Higher rank to lower rank
    cdef np.ndarray[DTYPE_t, ndim=1] sorted_examples = np.zeros(number_of_examples, dtype=np.int32)

    # Converts rank matrix to rank list
    for i in range(number_of_examples):
        ranking[i] = 1
        for j in range(number_of_examples):
            if scores[i] > scores[j]:
                ranking[i] += 1
        sorted_examples[number_of_examples - ranking[i]] = i

    # Computes prec@i
    cdef float pos_count = 0.
    cdef float total_count = 0.
    cdef float precision_at_i = 0.
    for i in range(number_of_examples):
        label = gt_labels[sorted_examples[i]]
        if label == 1:
            pos_count += 1
        total_count += 1
        if label == 1:
            precision_at_i += pos_count / total_count
    precision_at_i /= pos_count
    return precision_at_i


def generate_ranking_from_labels_cython(np.ndarray[DTYPE_t, ndim=1] labels):

    cdef int number_of_examples = labels.shape[0]
    cdef int i, j
    cdef np.ndarray[DTYPE_t, ndim=1] ranking = np.zeros(number_of_examples, dtype=np.int32)

    # Initializes ranking
    for i in range(number_of_examples):
        for j in range(i+1, number_of_examples):
            if labels[i] == 1:
                ranking[i] += 1
                ranking[j] -= 1
            else:
                ranking[i] -= 1
                ranking[j] += 1

    return ranking
