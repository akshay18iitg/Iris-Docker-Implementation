from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pickle
import os


class TrainerConfig:
    def __init__(self):
        self.model_path = os.path.join('artifacts','model.pkl')

class Trainer:

    def __init__(self):
        self.config = TrainerConfig()
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target

    def initialize_training(self):

        X_train,X_test,y_train,y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)

        model = RandomForestClassifier(n_estimators=10)

        model.fit(X_train,y_train)


        predicted = model.predict(X_test)

        score = accuracy_score(predicted,y_test)

        os.makedirs('./artifacts',exist_ok=True)

        with open(self.config.model_path,'wb') as f:
            pickle.dump(model,f)

        return score


if __name__ == '__main__':
    trainer = Trainer()
    print(trainer.initialize_training())

