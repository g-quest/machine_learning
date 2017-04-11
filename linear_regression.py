'''
Gregory V. Cuesta
gvc2108
4/2/2017

COMS W4701 AI
Assigment #3
Problem #2

written in Python 3
'''

import sys
import csv
import pandas as pd

def main():

    # check that proper arguments were entered in command line
    if len(sys.argv) != 3:
        print('ERROR: two arguments required')
        print('usage: python3 problem2_3.py <input_csv> <output_csv>')
        sys.exit()
    
    # data prep and normalization
    df = pd.read_csv(sys.argv[1], header = None)
    df[0] = scale(df[0])
    df[1] = scale(df[1])
    df.insert(0, 'intercept', 1)
    df.columns = ['intercept', 'age', 'weight', 'height']
    nRows = len(df) - 1 # exclude header
    
    outputList = []
    
    ''' perform gradient descent for each required learning rate '''
    
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    
    for alpha in alphas:
        
        # initialize coefficients
        b_0 = 0.0
        b_age = 0.0
        b_weight = 0.0
        
        n_iters = 0
        while (n_iters < 100):
            
            sumErrIntercept = 0
            sumErrAge = 0
            sumErrWeight = 0
            sumSqErr = 0
            for row in df.itertuples():
                
                intercept = row[1]
                age = row[2]
                weight = row[3]
                y = row[4]
                
                err = error(b_0, b_age, b_weight, age, weight, y)
                
                errIntercept = err * intercept
                sumErrIntercept = sumErrIntercept + errIntercept
                
                errAge = err * age
                sumErrAge = sumErrAge + errAge
                
                errWeight = err * weight
                sumErrWeight = sumErrWeight + errWeight
                
                sqError = err ** 2
                sumSqErr = sumSqErr + sqError

            b_0      = b_0      - (alpha * (1.0 / nRows) * sumErrIntercept)
            b_age    = b_age    - (alpha * (1.0 / nRows) * sumErrAge)
            b_weight = b_weight - (alpha * (1.0 / nRows) * sumErrWeight)
            
            ''' uncomment to see change in costs '''
#            iterCost = cost(nRows, sumSqErr)
#            print(iterCost)
            
            n_iters = n_iters + 1
        
        finalValues = [alpha, 100, b_0, b_age, b_weight]
        outputList.append(finalValues)
        
    ''' perform gradient descent for chosen learning rate and iterations '''   

    # chose this number because it's the last before it starts diverging
    # costs started increasing at 1.15396
    chosenAlpha = 1.15395
    
    # initialize coefficients
    b_0 = 0.0
    b_age = 0.0
    b_weight = 0.0
    
    chosen_n_iters = 0
    while (chosen_n_iters < 1000):
   
        sumErrIntercept = 0
        sumErrAge = 0
        sumErrWeight = 0
        sumSqErr = 0
        for row in df.itertuples():
            
            intercept = row[1]
            age = row[2]
            weight = row[3]
            y = row[4]
            
            err = error(b_0, b_age, b_weight, age, weight, y)
            
            errIntercept = err * intercept
            sumErrIntercept = sumErrIntercept + errIntercept
            
            errAge = err * age
            sumErrAge = sumErrAge + errAge
            
            errWeight = err * weight
            sumErrWeight = sumErrWeight + errWeight
            
            sqError = err ** 2
            sumSqErr = sumSqErr + sqError
        
        b_0      = b_0      - (chosenAlpha * (1.0 / nRows) * sumErrIntercept)
        b_age    = b_age    - (chosenAlpha * (1.0 / nRows) * sumErrAge)
        b_weight = b_weight - (chosenAlpha * (1.0 / nRows) * sumErrWeight)
        
        ''' uncomment to see change in costs '''
#        iterCost = cost(nRows, sumSqErr)
#        print(iterCost)
        
        chosen_n_iters = chosen_n_iters + 1
        
    finalValues = [chosenAlpha, 1000, b_0, b_age, b_weight]
    outputList.append(finalValues)
    
    # create output file
    outputFileName = sys.argv[2]
    outputFile = open(outputFileName, 'w')
    writer = csv.writer(outputFile)
    for line in outputList:
        writer.writerow(line)  
    outputFile.close()

    print('\nProgram complete. \n\"output2.csv\" file created.\n')
    
    return 0

def scale(df_column):
    
    columnMean = df_column.mean()
    columnSD = df_column.std()
    
    scaleFormula = (df_column - columnMean) / columnSD
    
    return scaleFormula

def error(b_0, b_age, b_weight, age, weight, y):
    
    y_hat = b_0 + (b_age * age) + (b_weight * weight)
    
    return (y_hat - y)
    
def cost(nRows, sqErrorSum):
    
    empRisk = (1 / (2 * nRows)) * sqErrorSum
    
    return empRisk
    
main()