import music21
import pprint
import textdistance
import itertools
from multiprocessing import Pool

FILENAME = 'allabcwrepeats_parsed'
FILE_LIMIT = 10000

def compare_strings(idxa, a, idxb, b):
    a = 'M:{}'.format(a.split('M:')[1])
    b = 'M:{}'.format(b.split('M:')[1])
    score = textdistance.damerau_levenshtein.normalized_similarity(a, b)
    print(idxa, idxb, score)

if __name__ == '__main__':
    with open(FILENAME, encoding='utf-8') as f:
        data = f.read()
    # Files are delimited by a blank line (2 '\n's in a row )
    files = data.split('\n\n')
    # Files after being parsed by music21
    if not FILE_LIMIT:
        FILE_LIMIT = len(files)
    combinations = itertools.combinations(range(FILE_LIMIT), 2)
    comparisons = [(idxa, files[idxa], idxb, files[idxb]) for idxa, idxb in combinations]
    p = Pool(32)
    p.starmap(compare_strings, comparisons)