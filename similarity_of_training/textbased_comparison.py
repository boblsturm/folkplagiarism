import music21
import pprint
import textdistance
import itertools
from multiprocessing import Pool

FILENAME = '../allabcwrepeats_parsed'
FILE_LIMIT = 10000

def compare_strings(idxa, a, idxb, b):
    a = 'M:{}'.format(a.split('M:')[1])
    b = 'M:{}'.format(b.split('M:')[1])
    score = textdistance.damerau_levenshtein.normalized_similarity(a, b)
    return (idxa, idxb, score)

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
    subset_size = 1000
    subsets = [comparisons[x:x+subset_size] for x in range(0, len(comparisons), subset_size)]
    p = Pool(32)
    with open("outout.py", "w") as out:
        out.write('output = []\n')
        for idx, subset in enumerate(subsets):
            out.write('# subset_{}\n'.format(idx))
            out.write('output.extend({})\n'.format(p.starmap(compare_strings, subset)))