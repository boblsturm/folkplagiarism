import re
import pandas as pd

FILENAME = 'allabcwrepeats_parsed'

with open(FILENAME, encoding='utf-8') as f:
    data = f.read()
# Files are delimited by a blank line (2 '\n's in a row )
files = data.split('\n\n')
# Files after being parsed by music21
dictionary = {
    'title': [],
    'time_signature': [],
    'key': [],
    'abcdata': []
}
for f in files:
    regexp = r'^(T:)?(?P<title>.*)?(\nM:)?(?P<time_signature>.*)?(\nK:)?(?P<key>.*)?(\n)?(?P<abcdata>.*)?$'
    m = re.match(regexp, f, re.M)
    d = m.groupdict()
    [dictionary[k].append(v) for k,v in d.items()]
df = pd.DataFrame.from_dict(dictionary)
