import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_tranformation import DataTransformation, DataTransformationConfig

@dataclass
class dataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'data.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestionConfig = dataIngestionConfig()
        
    def initiate_data_ingeston(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('read the dataset as dataframe')
            
            os.makedirs(os.path.dirname(self.ingestionConfig.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestionConfig.raw_data_path, index=False, header= True)
            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestionConfig.train_data_path, index=False, header= True)
            test_set.to_csv(self.ingestionConfig.test_data_path, index=False, header=True)
            
            logging.info('Inhestion of the data is completed')
            return (
                self.ingestionConfig.train_data_path,
                self.ingestionConfig.test_data_path
                )
        except Exception as e:
            raise CustomeException(e,sys)
        
if __name__ == '__main__':
    obj  =DataIngestion()
    train_data, test_data = obj.initiate_data_ingeston()
    
    
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
