from collections import Counter, defaultdict
import numpy

# def index_all_ngrams(tunes, ngram_sizes):
#     ngrams_by_size = {}
#     for ngram_size in ngram_sizes:
#         ngrams_by_size[ngram_size] = index_ngrams(tunes, ngram_size)
        
#     return ngrams_by_size

def index_ngrams(tunes, ngram_size):
    total = Counter()
    ngram_occurrences = defaultdict(list)
    all_tune_ngrams = []
    all_tune_ngram_counts = []
    for tune_id, tune in enumerate(tunes):
        tune_ngrams = get_ngrams(tune, ngram_size)
        tune_ngram_counts = Counter(tune_ngrams)
        total += tune_ngram_counts
        all_tune_ngrams.append(tune_ngrams)
        all_tune_ngram_counts.append(tune_ngram_counts)
        for pos, ngram in enumerate(tune_ngrams):
            ngram_occurrences[ngram].append({
                'tune_id': tune_id,
                'pos': pos
            })
    return total, ngram_occurrences, all_tune_ngrams, all_tune_ngram_counts

def get_ngrams(intervals, ngram_size):
    """
    Get list of ngrams (tuples) with length `ngram_size` found in `intervals`.
    """
    ngrams = [
        tuple(intervals[pos : pos + ngram_size])
        for pos in range(len(intervals) - ngram_size + 1)
    ]
    return ngrams

def find_plagiarism(training_tunes, generated_tunes, ngram_size):
    # ngram_sizes = [2]
    training_ngrams = index_ngrams(training_tunes, ngram_size)
    generated_ngrams = index_ngrams(generated_tunes, ngram_size)
    plagiarisms = []
    for tune, tune_ngrams in zip(generated_tunes, generated_ngrams[2]):
        occurrences_in_training = {}
        for ngram in tune_ngrams:
            occurrences_in_training[ngram] = training_ngrams[0].get(ngram, 0)
        plagiarisms.append(occurrences_in_training)

    return plagiarisms

def find_plagiarism2(training_tunes, generated_tunes, ngram_size):
    # ngram_sizes = [2]
    training_ngrams = index_ngrams(training_tunes, ngram_size)
    generated_ngrams = index_ngrams(generated_tunes, ngram_size)
    overlap_matrix = numpy.zeros((len(generated_tunes), len(training_tunes)))
    overlaps = defaultdict(dict)
    # go over all generated tunes
    for generated_tune_id, generated_tune_ngram_counts in enumerate(generated_ngrams[3]):
        # tune_ngrams - list of ngrams in this tune
        # find overlap with each of the training tunes
        for training_tune_id, training_tune_ngram_counts in enumerate(training_ngrams[3]):
            overlapping_ngrams = generated_tune_ngram_counts.keys() & training_tune_ngram_counts.keys()
            overlap_matrix[generated_tune_id, training_tune_id] = len(overlapping_ngrams)
            overlaps[generated_tune_id][training_tune_id] = overlapping_ngrams
        
    return overlap_matrix, overlaps


def find_plagiarism3(training_tunes, generated_tunes):
    """
    For each pair find of tunes find the largest overlapping substring
    """
    ngram_sizes = range(1, 20)
    all_training_ngrams = {}
    all_generated_ngrams = {}
    print('Indexing ngrams')
    for ngram_size in ngram_sizes:
        print(ngram_size)
        all_training_ngrams[ngram_size] = index_ngrams(training_tunes, ngram_size)
        all_generated_ngrams[ngram_size] = index_ngrams(generated_tunes, ngram_size)
    overlap_matrix = numpy.zeros((len(generated_tunes), len(training_tunes)))
    overlaps = defaultdict(dict)
    # go over all generated tunes
    # for generated_tune_id, generated_tune_ngram_counts in enumerate(generated_ngrams[3]):
    for generated_tune_id in range(len(generated_tunes)):
        print('Computing overlap for generated tune %s' % generated_tune_id)
        # tune_ngrams - list of ngrams in this tune
        # find overlap with each of the training tunes
        # for training_tune_id, training_tune_ngram_counts in enumerate(generated_ngrams[3]):
        for training_tune_id in range(len(training_tunes)):
            for ngram_size in ngram_sizes:
                training_tune_ngrams = all_training_ngrams[ngram_size][3][training_tune_id].keys()
                generated_tune_ngrams = all_generated_ngrams[ngram_size][3][generated_tune_id].keys()
                overlapping_ngrams = training_tune_ngrams & generated_tune_ngrams
                if len(overlapping_ngrams) > 0:
                    overlap_matrix[generated_tune_id, training_tune_id] = ngram_size
                    overlaps[generated_tune_id][training_tune_id] = overlapping_ngrams
        
    return overlap_matrix, overlaps