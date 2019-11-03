def get_metadata_and_text(line):
    txt = line.split('\n')[-1]
    metadata, _ = line.split(txt)
    return metadata, txt

def replace_letter(s):
    if s[0].isalpha():
        return 'X'
    else:
        return s

def replace_letters(txt):
    toks = txt.split(' ')
    new_toks = [replace_letter(t) for t in toks]
    return ' '.join(new_toks)

def rhythm_only(line):
    metadata, txt = get_metadata_and_text(line)
    txt = txt.strip()
    txt = replace_letters(txt)
    return "".join([metadata, txt])

for f in ["output_folkrnnv2.txt", "original_data.txt"]:
    lines = open(f).read().split('\n\n')[0:-1]
    rhythm_lines = []
    for i,l in enumerate(lines):
        try:
            rhythm_lines.append(rhythm_only(l))
        except:
            print(i)
    big_str = "\n\n".join(rhythm_lines)
    filename = f.split(".txt")[0] + "_rhythm_only" + ".txt"
    with open(filename, "w") as outfile:
        outfile.write(big_str)
