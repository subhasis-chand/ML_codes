import numpy as np
from logisticRegression import LogisticRegression
import sklearn as skl
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def main():
    rawData = load_breast_cancer()
    trainingData = np.matrix(rawData.data, dtype='float')
    x = trainingData[:, 0:10]
    x = x / x.max(axis=0)
    y = np.matrix(rawData.target, dtype='float').T
    print("Data loaded successfully...")
    shuffle = False
    while True:
        print("Do you want to shuffle the data while spliting for training and testing? press y or n... q for quit...")
        userInput = input()
        if userInput == 'y' or userInput == 'Y':
            shuffle = True
            break
        elif userInput == 'n' or userInput == 'N':
            break
        elif userInput == 'q' or userInput == 'Q':
            print("Quitting the program...")
            exit()
        else:
            print("invalid input... Please provide a valid input.")
    
    animation = False
    while True:
        print("Do you want to show the animation for gradient descent? press y or n... q for quit...")
        userInput = input()
        if userInput == 'y' or userInput == 'Y':
            animation = True
            break
        elif userInput == 'n' or userInput == 'N':
            break
        elif userInput == 'q' or userInput == 'Q':
            print("Quitting the program...")
            exit()
        else:
            print("invalid input... Please provide a valid input.")

            
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=shuffle)
    print("Training and testing data generated...")

    logReg = LogisticRegression(x_train, y_train)
    logReg.train(animation=animation, alpha=0.001)
    y_pred = logReg.test(x_test)
    score = logReg.scores(y_test)
    print("precision: ", score['precision'])
    print("recall: ", score['recall'])
    print("f1 score: ", score['f1'])
    print("accuaracy: ", score['accuaracy'])
    print("confusionMatrix: ", score['confusionMatrix'])

if __name__ == "__main__":
    main()
         
































