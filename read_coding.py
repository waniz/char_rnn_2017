import codecs
f = codecs.open('suto_chechov.txt', encoding='cp1251')

for line in f:
    print(repr(line))
