"""
Reference:
Machine learning predicts live‑birth occurrence before in‑vitro fertilization treatment
"""
# custom modules
from preprocess.base import *


# TODO: 전처리 함수를 모두 모아 한 번에 전처리하는 함수 구현
def pp_all_v1(tr_data_path: str = None, tt_data_path: str = None) -> tuple:
    """Refernce 논문을 기반으로 한 전처리 함수의 첫 번째 버전입니다.

    전처리는 다음과 같은 순서대로 진행됩니다.
        0. 사용할 categorical_features, numerical_features 각각 정의
        1. 학습 및 테스트 데이터 로드
        2. 정의한 features를 제외한 나머지 features 제거
        3. 범주형 데이터를 수치형 데이터로 변환

    Args:
        tr_data_path (str, optional): _description_. Defaults to None.
        tt_data_path (str, optional): _description_. Defaults to None.

    Returns:
        tuple: _description_
    """
    categorical_features = [
        '시술 당시 나이', 'IVF 시술 횟수', 'IVF 임신 횟수', 'IVF 출산 횟수', 
    ]
    numerical_features = [
        '남성 주 불임 원인', '남성 부 불임 원인',
        '여성 주 불임 원인', '여성 부 불임 원인',
        '부부 주 불임 원인', '부부 부 불임 원인', '불명확 불임 원인',
        '불임 원인 - 난관 질환', '불임 원인 - 남성 요인',
        '불임 원인 - 배란 장애', '불임 원인 - 자궁경부 문제',
        '불임 원인 - 자궁내막증', '불임 원인 - 정자 농도',
        '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태', 
        '이식된 배아 수', '해동 난자 수', '수집된 신선 난자 수',
        '파트너 정자와 혼합된 난자 수', '동결 배아 사용 여부', '신선 배아 사용 여부',
    ]
    
    tr_df, tt_df = load_data(tr_data_path, tt_data_path)
    tr_df, tt_df = drop_features(
        tr_df, tt_df, categorical_features + numerical_features
    )
    tr_df, tt_df = categorical_to_numerical(
        tr_df, tt_df, categorical_features
    )

    return (tr_df, tt_df)    
    

# Test Code
if __name__ == "__main__":
    pass