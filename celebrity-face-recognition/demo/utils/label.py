def read_label(filepath):
    label_dict = {}
    with open(filepath, 'r') as f:
        while True:
            s = f.readline()[:-1]
            if s == '':
                break
            s = s.split()
            name = ''.join(s[:-1])
            idx = int(s[-1])
            label_dict[idx] = name

    return label_dict, len(label_dict)
