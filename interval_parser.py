import music21
import pprint

if __name__ == '__main__':
    with open('output_folkrnnv2.txt') as f:
        data = f.read()
    # Files are delimited by a blank line (2 '\n's in a row )
    files = data.split('\n\n')
    # Files after being parsed by music21
    m21files = []
    for f in files[:100]:
        # print(f)
        try:
            s = music21.converter.parseData(f)
            notes = s.flat.notes
            # print(list(notes))
            intervals = []
            for n1, n2 in zip(notes[:-1], notes[1:]):
                i = music21.interval.Interval(n1, n2)
                intervals.append(i.semitones)
            m21files.append(intervals)
        except:
            print('crashed music21 -- {}'.format(f))
    print(m21files)