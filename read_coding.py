import codecs
f = codecs.open('mega_samples.txt', encoding='cp1251')

for line in f:
    print(repr(line))
