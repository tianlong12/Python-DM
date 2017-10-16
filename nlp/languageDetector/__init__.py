#encoding=utf-8
from sklearn.model_selection import train_test_split
from LanguageDetector import LanguageDetector
in_f = open('data.csv')
lines = in_f.readlines()
in_f.close()
dataset = [(line.strip()[:-3], line.strip()[-2:]) for line in lines]
x, y = zip(*dataset)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
language_detector = LanguageDetector()
language_detector.fit(x_train, y_train)
print(language_detector.predict('viennent de signer la convention de kinshasa sur le contrôle'))
print(language_detector.score(x_test, y_test))