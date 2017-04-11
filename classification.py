'''
Gregory V. Cuesta
gvc2108
4/2/2017

COMS W4701 AI
Assigment #3
Problem #3

written in Python 3
'''

import sys
import csv
import pandas as pd
from sklearn import svm, grid_search
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
def main():

    '''
    Preprocessing
    '''
    
    # check that proper arguments were entered in command line
    if len(sys.argv) != 3:
        print('ERROR: two arguments required')
        print('usage: python3 problem2_3.py <input_csv> <output_csv>')
        sys.exit()
        
    # load dataset and feature scale
    df = pd.read_csv(sys.argv[1])
    df['A'] = scale(df['A'])
    df['B'] = scale(df['B'])

    X = df.iloc[:, [0, 1]].values
    y = df.iloc[:, 2].values

    # split the dataset into training and test sets (60/40)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, stratify = y)
    
    outputList = []

    
    '''
    SVM with Linear Kernel
    '''
    
    cfr = svm.SVC(kernel = 'linear')
    parameters = {'C':[0.1, 0.5, 1, 5, 10, 50, 100]}
    svmLinGS = grid_search.GridSearchCV(cfr, parameters)
    svmLinGS.fit(X_train, y_train) 
    print('\n*** SVM with Linear Kernel ***\n')
    print('SVM Linear C Best Estimator: ', svmLinGS.best_estimator_.C)
    print('SVM Linear Best Score: ', svmLinGS.best_score_)
    
    svmLinPredictions = svmLinGS.predict(X_test)   
    svmLinScores = cross_val_score(svmLinGS, X_train, y_train, cv = 5)
    svmLinBestScore = round(max(svmLinScores), 4)
    svmLinTestScore = accuracy_score(y_test, svmLinPredictions)
    print('SVM Linear Scores: ', svmLinScores)
    print('SVM Linear Best Score: ', svmLinBestScore)
    print('SVM Linear Test Score: ', svmLinTestScore)

    cm = confusion_matrix(y_test, svmLinPredictions)
    print('SVM Linear Confusion Matrix: \n', cm)
    
    outputList.append(['svm_linear', svmLinBestScore, svmLinTestScore])
 
 
    '''
    SVM with Polynomial Kernel
    ''' 
    cfr = svm.SVC(kernel = 'poly')
    parameters = {'C':[0.1, 1, 3], 'degree':[4, 5, 6], 'gamma':[0.1, 1]}
    svmPolyGS = grid_search.GridSearchCV(cfr, parameters)
    svmPolyGS.fit(X_train, y_train)
    print('\n*** SVM with Polynomial Kernel ***\n')
    print('SVM Polynomial C Best Estimator: ', svmPolyGS.best_estimator_.C)
    print('SVM Polynomial degree Best Estimator: ', svmPolyGS.best_estimator_.degree) 
    print('SVM Polynomial gamma Best Estimator: ', svmPolyGS.best_estimator_.gamma)    
    print('SVM Polynomial Best Score: ', svmPolyGS.best_score_)
    
    svmPolyPredictions = svmPolyGS.predict(X_test)   
    svmPolyScores = cross_val_score(svmPolyGS, X_train, y_train, cv = 5)
    svmPolyBestScore = round(max(svmPolyScores), 4)
    svmPolyTestScore = accuracy_score(y_test, svmPolyPredictions)
    print('SVM Polynomial Scores: ', svmPolyScores)
    print('SVM Polynomial Best Score: ', svmPolyBestScore)
    print('SVM Polynomial Test Score: ', svmPolyTestScore)

    cm = confusion_matrix(y_test, svmPolyPredictions)
    print('SVM Polynomial Confusion Matrix: \n', cm)
    
    outputList.append(['svm_polynomial', svmPolyBestScore, svmPolyTestScore])

 
    '''
    SVM with RBF Kernel
    '''  
    cfr = svm.SVC(kernel = 'rbf')
    parameters = {'C':[0.1, 0.5, 1, 5, 10, 50, 100], 'gamma':[0.1, 0.5, 1, 3, 6, 10]}
    svmRBFGS = grid_search.GridSearchCV(cfr, parameters)
    svmRBFGS.fit(X_train, y_train)
    print('\n*** SVM with RBF Kernel ***\n')
    print('SVM RBF C Best Estimator: ', svmRBFGS.best_estimator_.C)
    print('SVM RBF gamma Best Estimator: ', svmRBFGS.best_estimator_.gamma)    
    print('SVM RBF Best Score: ', svmRBFGS.best_score_)
    
    svmRBFPredictions = svmRBFGS.predict(X_test)   
    svmRBFScores = cross_val_score(svmRBFGS, X_train, y_train, cv = 5)
    svmRBFBestScore = round(max(svmRBFScores), 4)
    svmRBFTestScore = accuracy_score(y_test, svmRBFPredictions)
    print('SVM RBF Scores: ', svmRBFScores)
    print('SVM RBF Best Score: ', svmRBFBestScore)
    print('SVM RBF Test Score: ', svmRBFTestScore)

    cm = confusion_matrix(y_test, svmRBFPredictions)
    print('SVM RBF Confusion Matrix: \n', cm)
    
    outputList.append(['svm_rbf', svmRBFBestScore, svmRBFTestScore])
    

    '''
    Logistic Regression
    '''  
    cfr = LogisticRegression()
    parameters = {'C':[0.1, 0.5, 1, 5, 10, 50, 100]}
    lrGS = grid_search.GridSearchCV(cfr, parameters)
    lrGS.fit(X_train, y_train)
    print('\n*** Logistic Regression ***\n')
    print('Logistic Regression C Best Estimator: ', lrGS.best_estimator_.C)    
    print('Logistic Regression Best Score: ', lrGS.best_score_)
    
    lrPredictions = lrGS.predict(X_test)   
    lrScores = cross_val_score(lrGS, X_train, y_train, cv = 5)
    lrBestScore = round(max(lrScores), 4)
    lrTestScore = accuracy_score(y_test, lrPredictions)
    print('Logistic Regression Scores: ', lrScores)
    print('Logistic Regression Best Score: ', lrBestScore)
    print('Logistic Regression Test Score: ', lrTestScore)

    cm = confusion_matrix(y_test, lrPredictions)
    print('Logistic Regression Confusion Matrix: \n', cm)
    
    outputList.append(['logistic', lrBestScore, lrTestScore])
    
    
    '''
    k-Nearest Neighbors
    '''  
    cfr = KNeighborsClassifier()
    parameters = {'n_neighbors': list(range(1,50)), 'leaf_size': list(range(5, 60, 5))}
    knnGS = grid_search.GridSearchCV(cfr, parameters)
    knnGS.fit(X_train, y_train)
    print('\n*** K-Nearest Neighbors ***\n')
    print('K-Nearest Neighbors n_neighbors Best Estimator: ', knnGS.best_estimator_.n_neighbors) 
    print('K-Nearest Neighbors leaf_size Best Estimator: ', knnGS.best_estimator_.leaf_size) 
    print('K-Nearest Neighbors Best Score: ', knnGS.best_score_)
    
    knnPredictions = knnGS.predict(X_test)   
    knnScores = cross_val_score(knnGS, X_train, y_train, cv = 5)
    knnBestScore = round(max(knnScores), 4)
    knnTestScore = accuracy_score(y_test, knnPredictions)
    print('K-Nearest Neighbors Scores: ', knnScores)
    print('K-Nearest Neighbors Best Score: ', knnBestScore)
    print('K-Nearest Neighbors Test Score: ', knnTestScore)

    cm = confusion_matrix(y_test, knnPredictions)
    print('K-Nearest Neighbors Confusion Matrix: \n', cm)
    
    outputList.append(['knn', knnBestScore, knnTestScore])
    
    
    '''
    Decision Trees
    '''  
    cfr = DecisionTreeClassifier()
    parameters = {'max_depth': list(range(1, 50)), 'min_samples_split':list(range(2, 10))}
    dtGS = grid_search.GridSearchCV(cfr, parameters)
    dtGS.fit(X_train, y_train)
    print('\n*** Decision Trees ***\n')
    print('Decision Trees max_depth Best Estimator: ', dtGS.best_estimator_.max_depth) 
    print('Decision Trees min_samples_split Best Estimator: ', dtGS.best_estimator_.min_samples_split) 
    print('Decision Trees Best Score: ', dtGS.best_score_)
    
    dtPredictions = dtGS.predict(X_test)   
    dtScores = cross_val_score(dtGS, X_train, y_train, cv = 5)
    dtBestScore = round(max(dtScores), 4)
    dtTestScore = accuracy_score(y_test, dtPredictions)
    print('Decision Trees Scores: ', dtScores)
    print('Decision Trees Best Score: ', dtBestScore)
    print('Decision Trees Test Score: ', dtTestScore)

    cm = confusion_matrix(y_test, dtPredictions)
    print('Decision Trees Confusion Matrix: \n', cm)
    
    outputList.append(['decision_tree', dtBestScore, dtTestScore])


    '''
    Random Forest
    '''  
    cfr = RandomForestClassifier()
    parameters = {'max_depth': list(range(1, 50)), 'min_samples_split':list(range(2, 10))}
    rfGS = grid_search.GridSearchCV(cfr, parameters)
    rfGS.fit(X_train, y_train)
    print('\n*** Random Forest ***\n')
    print('Random Forest max_depth Best Estimator: ', rfGS.best_estimator_.max_depth) 
    print('Random Forest min_samples_split Best Estimator: ', rfGS.best_estimator_.min_samples_split) 
    print('Random Forest Best Score: ', rfGS.best_score_)
    
    rfPredictions = dtGS.predict(X_test)   
    rfScores = cross_val_score(dtGS, X_train, y_train, cv = 5)
    rfBestScore = round(max(dtScores), 4)
    rfTestScore = accuracy_score(y_test, dtPredictions)
    print('Random Forest Scores: ', rfScores)
    print('Random Forest Best Score: ', rfBestScore)
    print('Random Forest Test Score: ', rfTestScore)

    cm = confusion_matrix(y_test, rfPredictions)
    print('Random Forest Confusion Matrix: \n', cm)
    
    outputList.append(['random_forest', rfBestScore, rfTestScore])


    '''
    Post-processing
    '''
    outputFileName = sys.argv[2]
    outputFile = open(outputFileName, 'w')
    writer = csv.writer(outputFile)
    for line in outputList:
        writer.writerow(line)  
    outputFile.close()

    print('\nProgram complete. \n\"output3.csv\" file created.\n')
    
    return 0

def scale(df_column):
    
    columnMean = df_column.mean()
    columnSD = df_column.std()
    
    scaleFormula = (df_column - columnMean) / columnSD
    
    return scaleFormula

main()