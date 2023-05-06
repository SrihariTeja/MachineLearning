# MachineLearning
In this project a dataset is taken which contains two columns one is the reviews given to the restaurant and the second column labels the reviews as the positive review or the negative review.
After taking the dataset I have used the NLP concept to train the system. 
After taking the dataset firstly each and every review will undergo stemming process where every review is preprocessed so that all the irrelevant words which are not useful 
for the training are removed.
After the stemming process all the remaining words are converted to array of numericals. By using the concept of Bag of Words all the words are completed to numerical values.
The size of the Bag of words vector is 22000, which almost covers the daily using english words.
Here in this model the size of the vector taken is 1500 because this dataset covers the 1500 words only.
Then by using the different classification techniques the system is trained. Among all those techniques the best technique is taken based on thier accuracy score.
For this model the best technique is Gradient Descent.
Then by using the streamlit an interface is created where the user will enter the review and then the review will be classified as the positive or negative review.
