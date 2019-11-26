from emonet import EmoNet
em = EmoNet()
# p = em.predict(text='Sat in a gym surrounded by big , sweaty Yorkshire men !', with_dist=True)
p = em.predict(path='C:/Users/hazadeh/WorkStations/PycharmProjects/emotion/emonet/data/test.tsv', with_dist=True)
for line in p:
    print(line)
