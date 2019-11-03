import music21
import pprint

def compute_intervals(filename, file_limit):
    with open(filename, encoding='utf-8') as f:
        data = f.read()
    # Files are delimited by a blank line (2 '\n's in a row )
    files = data.split('\n\n')
    # Files after being parsed by music21
    all_intervals = []
    for idx, f in enumerate(files[:file_limit]):
        f = 'M:{}'.format(f.split('M:')[1])
        try:
            s = music21.converter.parseData(f)
            notes = s.flat.notes
            intervals = []
            for n1, n2 in zip(notes[:-1], notes[1:]):
                i = music21.interval.Interval(n1, n2)
                # As long as one of the notes is not a rest
                if hasattr(i, 'semitones'):
                    intervals.append(i.semitones)
            all_intervals.append(intervals)
        except:
            """print('crashed music21 -- {}'.format(f))"""
    return all_intervals

if __name__ == '__main__':
    files = 1000
    generated = compute_intervals('output_folkrnnv2.txt', 1000)
    print(generated)
    original = compute_intervals('allabcwrepeats_parsed', 1000)
    print(original)