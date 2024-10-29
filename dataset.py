import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

if __name__ == '__main__':
    api = KaggleApi()
    api.authenticate()


    # Baixando a competição com a API do Kaggle
    api.competition_download_files('hms-harmful-brain-activity-classification')


    print('Data source extraction complete.')