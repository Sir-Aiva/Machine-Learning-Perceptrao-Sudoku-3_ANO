import pandas as pd
from nltk.corpus import stopwords
import string

#Read csv
df = pd.read_csv(r'C:\Users\migue_000\Desktop\Faculdade\3º Ano\6º Semestre\Inteligência Artificial\projeto\spam.csv', engine='python')

#Change Dataset
df.drop_duplicates(inplace=True)  # remove duplicates
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df = df.rename(columns={"v1":"evaluation", "v2":"message"})
#print(df)

# nltk.download('stopwords')  correr 1 vez

#Change text
def transform_message(msg):
    # remove punctuation
    noPunctuation = [char for char in msg if char not in string.punctuation]
    noPunctuation = ''.join(noPunctuation)

    # remove stopwords
    finalWords = [word for word in noPunctuation.split() if word.lower() not in stopwords.words('english')]
    return finalWords

print("\n========== Data Preparation ==========\n")

# show token list
print("Token list: \n",df['message'].head().apply(transform_message),"\n")

#convert text to token matrix
from sklearn.feature_extraction.text import CountVectorizer
messagesBow = CountVectorizer(analyzer=transform_message).fit_transform(df['message'])   #messages bag of words
print("Messages Bag of Words shape: ", messagesBow.shape, "\n")

#Data division (80% training, 20% testing)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(messagesBow, df['evaluation'], test_size=0.2, random_state=0)

print("\n========== Naive Bayes ==========\n")

#Naive Bayes classifier
print("Naive Bayes classifier:")
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(x_train, y_train)
print(classifier.predict(x_train))
print(y_train.values)

#model evaluation (trainning)
print("\nModel trainning: ")
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
prediction = classifier.predict(x_train)
print(classification_report(y_train, prediction),"\n")
print("Trainning Confusion matrix: \n", confusion_matrix(y_train,prediction))
print("Model accuracy in training: ", accuracy_score(y_train, prediction),"%")


#model evaluation (testing)
print("\nModel testing:")
#print(classifier.predict(x_test))
#print(y_test.values)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
prediction = classifier.predict(x_test)
print(classification_report(y_test, prediction),"\n")
print("Testing Confusion matrix: \n", confusion_matrix(y_test,prediction))
print("Model accuracy in testing: ", accuracy_score(y_test, prediction),"%")
