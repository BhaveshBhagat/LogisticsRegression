import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn import metrics
import numpy as np

def main():
    ''' Reading the Dataset using pandas '''
    logisticsdata = pd.read_csv('C:/Users/Bhagat/Documents/Python/ML/DataSets/logistic regression dataset-Social_Network_Ads.csv')
    print(logisticsdata)

    ''' Encode the categorical varibale using LabelEncoder '''
    
    label_encoder = LabelEncoder()
    Gender = label_encoder.fit_transform(logisticsdata['Gender'])

    ''' adding the gender feature into dataset '''
    logisticsdata = pd.concat([logisticsdata , pd.DataFrame(Gender , columns = ['Gender'])], axis =1)

    ''' spliting the dataset into training and testing sets '''
    xtrain = logisticsdata.iloc[:200,[2,3,5]]
    ytrain = logisticsdata.iloc[:200,4]

    xtest = logisticsdata.iloc[201:,[2,3,5]]
    ytest = logisticsdata.iloc[201:,4]

    ''' Using StandardScaler we have to scale down the feature value for better accuracy '''
    st_scale = StandardScaler()
    x_scale = st_scale.fit_transform(xtrain)

    xt_scale = st_scale.fit_transform(xtest)

    ''' Creating the model object and train model using the training dataset '''
    logmodel = LogisticRegression()

    logmodel.fit(x_scale,ytrain)

    '''  Predicting the values for test dataset '''
    
    ypred = logmodel.predict(xt_scale)


    # Printing the test predictated output values and showing the confusion matrix and accuracy score
    print(ypred)

    print('Confusion Matrix : ' ,metrics.confusion_matrix(ytest,ypred) , '\n Accuracy Score : ' , metrics.accuracy_score(ytest,ypred))

   
main()   
