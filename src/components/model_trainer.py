import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,
                              GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class modelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

#reponsible for model training
class modelTrainer:
    def __init__(self):
        self.model_trainer_config = modelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Spliting training and test input data")
            X_train, y_train, X_test, y_test = (train_array[:,:-1],
                                                train_array[:,-1],
                                                test_array[:,:-1],
                                                test_array[:,-1]
                                            )
            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "GradientBoosting Regressor":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "KNeighbors Regressor":KNeighborsRegressor(),
                "XGB Regressor":XGBRegressor(),
                "CatBoost Regressor":CatBoostRegressor(verbose = False),
                "AdaBoost Regressor":AdaBoostRegressor()
            }
            
            params = {
                "Decision Tree":{
                    "criterion":['squared_error', 'friedman_mse','absolute_error', 'poisson'],
                    #  "spliter":['best','random'],
                    #  "max_feature":['sqrt','log2']
                },
                "Random Forest":{
                    "criterion":['squared_error', 'friedman_mse','absolute_error', 'poisson'],
                    # "max_feature":['sqrt','log2']
                    'n_estimators':[8,16,32,64,128,256]
                    
                },
                'GradientBoosting Regressor':{
                    # "loss":['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squarted_error','friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators':[8,16,32,64,128,256]
                },
                'Linear Regression':{},
                'KNeighbors Regressor':{
                    'n_neighbors':[5,7,9,11]
                    # 'weights':['uniform','distance']
                    # 'algorithm':['ball_tree','kd_tree','brute']
                },
                'XGB Regressor':{
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    # 'loss':['linear', 'square', 'exponential'],
                    'n_estimators':[8,16,32,64,128,256]
                },
                'CatBoost Regressor':{
                    'depth':[6,8,10],
                    'learning_rate':[.01,.05,.1],
                    'iterations':[30,50,100]
                },
                'AdaBoost Regressor':{
                    'learning_rate':[.1,.01,.05,.001],
                    # 'loss':['linear', 'square', 'exponential'],
                    'n_estimators':[8,16,32,64,128,256]  
                }
            }
            model_report = evaluate_models(X_train = X_train,
                                          y_train =y_train,
                                          X_test = X_test,
                                          y_test = y_test,
                                          models = models,
                                          params = params)
            # to get the best model name from dict
            best_model_score = max(sorted(model_report.values()))
            
            # to get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score <0.6:
                raise CustomException("No best mnodel flound")
            logging.info("Best found model on both training and testing dataset")
            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj = best_model)
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return f"R2 score: {r2_square}"

        except Exception as e:
            raise CustomException(e,sys)
        

