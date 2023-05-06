import pandas as pd
dataset = pd.read_csv("C:\\Users\\tejak\\OneDrive\\Documents\\Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred)
print(y_test)
# B Checking with Logistic Regression
from sklearn.linear_model import LogisticRegression

classifier_logistic_regression = LogisticRegression()
classifier_logistic_regression.fit(X_train, y_train)
y_pred1 = classifier_logistic_regression.predict(X_test)
## C K-NEAREST NEIGHBOURS

from sklearn.neighbors import KNeighborsClassifier  ##KNN

classifier_KNN = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier_KNN.fit(X_train, y_train)
y_pred2=classifier_KNN.predict(X_test)
# D Support Vector Machine
from sklearn.svm import SVC  ##svm

classifier_SVC = SVC(kernel='linear')
classifier_SVC.fit(X_train, y_train)
y_pred3=classifier_SVC.predict(X_test)
# E Kernel SVM
from sklearn.svm import SVC

classifier_SVC_rbf = SVC(kernel='rbf')
classifier_SVC_rbf.fit(X_train, y_train)
y_pred4=classifier_SVC_rbf.predict(X_test)
## F  DECISION TREE

from sklearn.tree import DecisionTreeClassifier  ##decision_tree

classifier_descision_tree = DecisionTreeClassifier(criterion='entropy')
classifier_descision_tree.fit(X_train, y_train)
y_pred5=classifier_descision_tree.predict(X_test)
## G RANDOM FOREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier  ##Random_Forest

classifier_RF = RandomForestClassifier(n_estimators=750, criterion='entropy')
classifier_RF.fit(X_train, y_train)
y_pred6=classifier_RF.predict(X_test)
# H Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier ##Stochastic Gradient Descent
clf1 = SGDClassifier(loss="hinge", penalty="l2", max_iter=150)
clf1.fit(X, y)
y_pred7=clf1.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
cm1=confusion_matrix(y_test,y_pred1)
print(cm1)
cm2=confusion_matrix(y_test,y_pred2)
print(cm2)

#print(cm1)
a=accuracy_score(y_test, y_pred)
print(a)
b=accuracy_score(y_test,y_pred1)
print(b)
c=accuracy_score(y_test,y_pred2)
print(c)
d=accuracy_score(y_test,y_pred3)
print(d)
e=accuracy_score(y_test,y_pred4)
print(e)
f=accuracy_score(y_test,y_pred5)
print(f)
g=accuracy_score(y_test,y_pred6)
print(g)
h=accuracy_score(y_test,y_pred7)
print(h)
import json
filename = "data1.json"
try:
    with open(filename, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    data = {"list1": [], "list2": []}
list1 = data["list1"]
list2 = data["list2"]
# Take user input for the new review
# s=str(input("Enter the review"))
# review = re.sub('[^a-zA-Z]', ' ', s)
# review = review.lower()
# review = review.split()
# ps = PorterStemmer()
# all_stopwords = stopwords.words('english')
# all_stopwords.remove('not')
# review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
# review = ' '.join(review)
# print(review)
#
# # Use the trained CountVectorizer to convert the new review text into numerical features
# new_X = cv.transform([review]).toarray()
#
# # Use the trained classifier to predict the sentiment of the new review
# new_y_pred = clf1.predict(new_X)
# # p.write(type(review))
# #review = " "
#
# # Print the predicted sentiment
# # if review.__contains__("worst"):
# # st.write("Negative review")
# if (review.__contains__("good") and review.__contains__("not")):
#     print("Negative review")
# elif (review.__contains__("bad") and review.__contains__("not") and review.__contains__("food")):
#     print("Positive review")
# elif (review.__contains__("not") and new_y_pred[0] == 1):
#     print("Negative review")
#
#     # list2.append(a)
# elif (review.__contains__("not") and new_y_pred[0] == 0):
#     print("Positive review")
#     # list1.append(a)
# elif new_y_pred[0] == 1:
#     print("Positive review")
#     # list1.append(a)
#
# else:
#     print("Negative review")
import streamlit as st

# Take user input for the new review
new_review = st.text_input("Enter your review:")
a=new_review


# Create a button for the user to submit their review
submit_button = st.button("Submit")

if submit_button:
    # Preprocess the new review text in the same way as the training data
    if len(new_review) == 0:
        st.write("Please enter  the review!!!")
    else:
        review = re.sub('[^a-zA-Z]', ' ', new_review)
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        #st.write(review)

        # Use the trained CountVectorizer to convert the new review text into numerical features
        new_X = cv.transform([review]).toarray()
        # st.write(new_X)

        # Use the trained classifier to predict the sentiment of the new review
        new_y_pred = clf1.predict(new_X)
        #st.write(type(review))
        review=" "


        # Print the predicted sentiment
        #if review.__contains__("worst"):
            #st.write("Negative review")
        if (new_review.__contains__("good") and new_review.__contains__("not")):
            st.write("Negative review")
        elif (review.__contains__("bad") and review.__contains__("not") and review.__contains__("food")) :
            st.write("Positive review")
        elif (review.__contains__("not") and new_y_pred[0] == 1):
            st.write("Negative review")

            #list2.append(a)
        elif (review.__contains__("not") and new_y_pred[0] == 0):
            st.write("Positive review")
            #list1.append(a)
        elif new_y_pred[0] == 1:
            st.write("Positive review")
            #list1.append(a)

        else:
            st.write("Negative review")
            #list2.append(a)


with open(filename, "w") as f:
    json.dump(data, f)

# Print the contents of list3
# st.write(list1)
# st.write(list2)



