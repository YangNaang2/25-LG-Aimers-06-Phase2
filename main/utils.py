"""
실험 관련 utility 함수들을 정의한 모듈입니다.
"""
# built-in library
import os
import random
from datetime import datetime

# external library
import numpy as np
import pandas as pd


# global variables
SAVE_DIR = "../results"


def set_seed(seed: int = 2025) -> None:
    """실험 재현을 위해 seed를 설정하는 함수입니다.

    Args:
        seed (int, optional): seed로 사용할 정수형 값. Defaults to 2025.
    """
    random.seed(seed)
    os.environ['PYTHONASHSEED'] = str(seed)
    np.random.seed(seed)


def make_submission_file(y_pred: np.ndarray, dir_name: str, model_name: str = None,
                         val_roc_auc: float = None, sub_path: str = "../data/sample_submission.csv") -> None:
    """제출용 csv 파일을 생성하는 함수입니다.

    csv 파일 이름의 형식은 다음과 같습니다:
        {YYYYMMDD}_0_{validation ROC-AUC score의 소수점 이하 숫자}_{model_name}.csv

    csv 파일 이름 예시:
        20250201_0_5000_sample.csv
            - 2025년 2월 1일에 생성
            - validation ROC-AUC score는 0.5000
            - 사용한 모델의 이름은 sample

    저장 경로는 다음과 같습니다:
        ../results/{dir_name}/20250201_0_5000_sample.csv

    Args:
        y_pred (np.ndarray): test data에 대한 모델의 예측 결과입니다.
        dir_name (str): csv 파일을 저장할 디렉토리의 이름입니다.
        model_name (str, optional): 실험에 사용한 모델명입니다. csv 파일명을 생성할 때 사용합니다. Defaults to None.
        val_roc_auc (float, optional): validation data에 대한 ROC-AUC 값입니다. Defaults to None.
        sub_path (str, optional): 
            - default submission 파일이 저장되어있는 경로입니다. 기본 양식을 불러오기 위해 사용합니다.
            - Defaults to "../data/sample_submission.csv".
    """
    # submission file 양식을 불러와서, 예측 결과를 'probability' column에 저장
    df_sub = pd.read_csv(sub_path)
    df_sub['probability'] = y_pred

    # 제출용 csv 파일의 이름 생성
    csv_name = datetime.now().strftime("%Y%m%d")

    if val_roc_auc: csv_name += ("_0_" + str(val_roc_auc)[2:])
    else: csv_name += "_none"

    if model_name: csv_name += ("_" + model_name + ".csv")
    else: csv_name += "_none.csv"

    # csv 파일을 저장할 디렉토리 생성
    dir_path = os.path.join(SAVE_DIR, dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # csv 파일 저장
    df_sub.to_csv(os.path.join(dir_path, csv_name), index=False)
