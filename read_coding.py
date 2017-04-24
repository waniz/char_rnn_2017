import codecs
f = codecs.open('autogenerator.txt', encoding='cp1251')

for line in f:
    print(repr(line))
