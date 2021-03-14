import matplotlib.pyplot as plt


min_len = 10000000
max_len = 0
num = 0
sum_len = 0
length_list = 30 * [0]
stride = 5
label_0 = 0

with open('./lcqmc-clean/train.tsv', 'r', encoding='utf8') as fr:
    line = fr.readline()
    while line != '':
        s1, s2, r = line.split('\t')
        if r == '0\n':
            label_0 += 1
        l1 = len(s1)
        l2 = len(s2)
        min_len = l1 if l1 < min_len else min_len
        min_len = l2 if l2 < min_len else min_len
        max_len = l1 if l1 > max_len else max_len
        max_len = l2 if l2 > max_len else max_len
        num += 1
        if num % 10000 == 0:
            print(num)
        sum_len += l1
        sum_len += l2

        if l1 < 10000:
            length_list[int(l1 / stride)] += 1
        if l2 < 10000:
            length_list[int(l2 / stride)] += 1

        line = fr.readline()

print(min_len)
print(max_len)
print(num)
print(label_0)
print('avg', sum_len / (num * 2))

length = [str(i) for i in range(0, 30)]
plt.bar(length, length_list)
plt.show()


def document_statistics():
    sum_index = 0
    sum_len = 0
    max_len = 0
    min_len = 1000000000
    avg_len = 0
    length_list = 5*[0]
    stride = 2000
    with open('./snli/train.tsv', encoding='utf-8') as f:
        line = f.readline()
        i= 1
        while line != '':
            sum_index += 1
            pid, url, title, content = line.split('\t')
            content = content.split(' ')
            length = len(content)
            sum_len += length
            if length > max_len:
                max_len = length
            if length < min_len:
                min_len = length
            if length < 10000:
                length_list[int(length / stride)] += 1
            if i % 10000 == 0:
                print(i)
            i += 1
            line = f.readline()
    avg_len = sum_len / i

    print('sum_index', sum_index)
    print('sum_len', sum_len)
    print('max_len', max_len)
    print('min_len', min_len)
    print('avg_len', avg_len)
    length = [str(i) for i in range(0, 5)]
    plt.bar(length, length_list)
    plt.show()
