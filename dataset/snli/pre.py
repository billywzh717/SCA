import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

with open('test.tsv', 'r') as fr:
    with open('test_no_stop_word.tsv', 'w') as fw:
        line = fr.readline().replace('\n', '')
        while line != '':
            label, s1, s2 = line.split('\t')
            s1s = s1.split(' ')
            s2s = s2.split(' ')

            l = []
            for word in s1s:
                word = word.lower()
                if word not in stop_words:
                    l.append(word)
            s1 = ' '.join(l)

            l = []
            for word in s2s:
                word = word.lower()
                if word not in stop_words:
                    l.append(word)
            s2 = ' '.join(l)
            fw.write(label + '\t' + s1 + '\t' + s2 + '\n')
            line = fr.readline().replace('\n', '')
