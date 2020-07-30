import pandas as pd
from nltk.corpus import stopwords
import string

max_iter=1000

#Read csv
df = pd.read_csv(r'C:\Users\migue_000\Desktop\Faculdade\3º Ano\6º Semestre\Inteligência Artificial\projeto\spam.csv', engine='python')

#Change Dataset
df.drop_duplicates(inplace=True)  # remove duplicates
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df = df.rename(columns={"v1":"evaluation", "v2":"message"})
#print(df)

#Change text
def transform_message(msg):
    # remove punctuation
    noPunctuation = [char for char in msg if char not in string.punctuation]
    noPunctuation = ''.join(noPunctuation)

    # remove stopwords
    finalWords = [word for word in noPunctuation.split() if word.lower() not in stopwords.words('english')]
    return finalWords

#convert text to token matrix
from sklearn.feature_extraction.text import CountVectorizer
messagesBow = CountVectorizer(analyzer=transform_message).fit_transform(df['message'])   #messages bag of words
print("Messages Bag of Words shape: ", messagesBow.shape)

#Data division (80% training, 20% testing)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(messagesBow, df['evaluation'], test_size=0.2, random_state=0)

print("\n========== Perceptron ==========\n")

#Perceptron classifier
print("Perceptron classifier:")
from sklearn.linear_model import Perceptron
classifier = Perceptron(random_state=0).fit(x_train,y_train)
print(classifier.predict(x_train))
print(y_train.values)

#Perceptron model training
print("\nModel trainning: ")
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
prediction = classifier.predict(x_train)
print(classification_report(y_train, prediction),"\n")
print("Trainning Confusion matrix: \n", confusion_matrix(y_train,prediction))
print("Model accuracy in training: ", accuracy_score(y_train, prediction),"%")

#Perceptron model testing
print("\nModel testing: ")
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
prediction = classifier.predict(x_test)
print(classification_report(y_test, prediction),"\n")
print("Testing Confusion matrix: \n", confusion_matrix(y_test,prediction))
print("Model accuracy in testing: ", accuracy_score(y_test, prediction),"%")