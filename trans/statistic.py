sum_len = 0
min_len = 10000000
max_len = 0
total_len = 0
avg_len = 0

with open('../dataset/snli/pair_train.tsv', 'r', encoding='utf8') as fr:
    line = fr.readline()
    while line != '':
        label, s1, s2 = line.split('\t')
        l1 = len(s1.split(' '))
        l2 = len(s2.split(' '))

        min_len = l1 if l1 < min_len else min_len
        min_len = l2 if l2 < min_len else min_len
        max_len = l1 if l1 > max_len else max_len
        max_len = l2 if l2 > max_len else max_len

        line = fr.readline()
        sum_len += l1
        sum_len += l2
        total_len += 2

    avg_len = sum_len / total_len
    print('max len', max_len)
    print('min len', min_len)
    print('avg len', avg_len)
