# written in Python 3

import sys
import csv

def main():
    
    # check that proper arguments were entered in command line
    if len(sys.argv) != 3:
        print('ERROR: two arguments required')
        print('usage: python3 perceptron_learning.py perceptron_learning_input.csv <output_csv>')
        sys.exit()
    
    # load dataset and convert strings to integers
    inputFile = open(sys.argv[1], 'r')
    reader = csv.reader(inputFile)
    data = []   
    for row in reader:
        row = [int(x) for x in row]
        data.append(row)    
    inputFile.close()   
    
    # initialize weights and bias
    w_1 = 0.001
    w_2 = 0.001
    b = 0.001
    
    outputList = []
    while (1):
        
        iterCost = 0
        for line in data:
            
            x_1 = line[0]
            x_2 = line[1]
            
            y = line[2]
            y_hat = getPredict(b, w_1, w_2, x_1, x_2)
            cost = y - y_hat
            w_1 = round(w_1 + (cost * x_1))
            w_2 = round(w_2 + (cost * x_2))
            b   = round(b   + (cost))
            
            weights = [w_1, w_2, b]
            iterCost = iterCost + cost
            
            ''' uncoment to see change in costs within each iteration '''
#            print(cost)
            
        ''' uncomment to see change in total iteration cost '''
#        print(iterCost)
            
        outputList.append(weights)

        # perceptron reaches convergence when weights are no longer changed
        if (iterCost == 0):
            break
        
    # create output file
    outputFileName = sys.argv[2]
    outputFile = open(outputFileName, 'w')
    writer = csv.writer(outputFile)
    for line in outputList:
        writer.writerow(line)  
    outputFile.close()
    
    print('\nConvergence reached. \n\"output1.csv\" file created.\n')

    return 0

def getPredict(b, w_1, w_2, x_1, x_2):
    
    weightedSum = b + (w_1 * x_1) + (w_2 * x_2)
    
    if weightedSum >= 0:
        return 1.0
    else:
        return -1.0

main()

