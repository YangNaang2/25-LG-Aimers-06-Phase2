"""
데이터 로드, 불필요한 피처 삭제 등
기본적인 역할을 수행하는 함수들을 정의한 모듈입니다.
"""
# external library
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(tr_data_path: str = None, tt_data_path: str = None) -> tuple:
    """학습 및 테스트 데이터를 불러옵니다.

    Args:
        tr_data_path (str, optional): 학습 데이터의 경로입니다. Defaults to None.
        tt_data_path (str, optional): 테스트 데이터의 경로입니다. Defaults to None.

    Returns:
        tuple
            : (pd.DataFrame, pd.DataFrame)
            : 불러온 학습 및 테스트 데이터를 반환합니다.
    """
    print("학습 및 테스트 데이터를 불러옵니다...")
    # 예외처리
    if not tr_data_path:
        print("[ERROR] 학습 데이터의 경로를 입력해주세요.")
        return
    
    if not tt_data_path:
        print("[ERROR] 테스트 데이터의 경로를 입력해주세요.")
        return

    tr_data = pd.read_csv(tr_data_path)
    tt_data = pd.read_csv(tt_data_path)
    print("<<<성공>>>!")

    return (tr_data, tt_data)


def drop_features(tr_df: pd.DataFrame, tt_df: pd.DataFrame,
                  used_features: list[str] = None) -> tuple:
    """used_features를 제외한 나머지 features를 삭제 후 반환합니다.

    Args:
        tr_df (pd.DataFrame): 학습 데이터입니다.
        tt_df (pd.DataFrame): 평가 데이터입니다.
        used_features (list[str], optional): 
            - 사용하고자 하는 features입니다. Defaults to None.

    Returns:
        tuple: (pd.DataFrame, pd.DataFrame)
    """
    print("used_features를 제외한 나머지 features를 삭제합니다...")
    # 예외처리
    if not used_features:
        print("[ERROR] 사용할 features를 입력하지 않았습니다.")
        return
    
    # used_features를 제외한 나머지 features를 삭제
    not_used_features = list(set(tt_df.columns.values) - set(used_features))

    print(f"총 {len(not_used_features)} 개의 feature를 제거합니다...")
    tr_df = tr_df.drop(columns=not_used_features, axis=1)
    tt_df = tt_df.drop(columns=not_used_features, axis=1)
    print("<<<성공>>>!")

    return (tr_df, tt_df)


# TODO: 결측치 처리
def categorical_to_numerical(tr_df: pd.DataFrame, tt_df: pd.DataFrame,
                          categorical_features: list[str]) -> tuple:
    print("범주형 데이터를 수치형 데이터로 변환합니다...")
    if not categorical_features:
        print("[ERROR] 범주형 features를 입력해주세요.")
        return

    # 원본 데이터 복사
    x_tr = tr_df.copy()
    x_tt = tt_df.copy()

    for f in categorical_features:
        print(f"현재 {f} feature에 대한 변환을 수행 중입니다...")
        if x_tr[f].dtype.name == 'object':
            le = LabelEncoder()

            # 학습 및 평가 데이터를 합쳐서 label encoding
            tr_f = list(x_tr[f].values)
            tt_f = list(x_tt[f].values)

            # map 만들기
            le.fit(tr_f + tt_f)

            # map 출력(확인용)
            print("label encoding map을 출력합니다.")
            print(dict(zip(le.classes_, le.transform(le.classes_))))

            # mapping
            x_tr[f] = le.transform(tr_f)
            x_tt[f] = le.transform(tt_f)
            print("<<<성공>>>!")
        
        else:
            print("[ERROR] 알 수 없는 에러가 발생했습니다.")
            return
        
    print(f"{categorical_features} 모두 변환에 성공했습니다.")
    return (x_tr, x_tt)