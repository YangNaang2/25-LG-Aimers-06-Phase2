"""
데이터 로드, 불필요한 피처 삭제 등
기본적인 역할을 수행하는 함수들을 정의한 모듈입니다.
"""
# built-in library
import random

# external library
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


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


def categorical_to_numerical(tr_df: pd.DataFrame, tt_df: pd.DataFrame,
                          categorical_features: list[str]) -> tuple:
    """categorical_features에 할당된 범주형 변수들을 수치형 변수들로 변환 후 반환합니다.

    Args:
        tr_df (pd.DataFrame): 학습 데이터입니다.
        tt_df (pd.DataFrame): 평가 데이터입니다.
        categorical_features (list[str]): 범주형 변수들의 이름을 담고 있는 리스트입니다.

    Returns:
        tuple: (pd.DataFrame, pd.DataFrame)
    """
    print("범주형 데이터를 수치형 데이터로 변환합니다...")
    if not categorical_features:
        print("[ERROR] 범주형 features를 입력해주세요.")
        return

    # 원본 데이터 복사
    x_tr = tr_df.copy()
    x_tt = tt_df.copy()

    # str(object) type으로 형변환
    for col in categorical_features:
        x_tr[col] = x_tr[col].astype(str)
        x_tt[col] = x_tt[col].astype(str)

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


def split_train_and_validation(tr_data: pd.DataFrame, val_size: float = 0.2, 
                               seed: int = 2025, method: str = 'sklearn') -> tuple:
    """주어진 data를 train / validation set으로 나누는 함수입니다.

    Args:
        tr_data (pd.DataFrame): split 할 data 입니다.
        val_size (float, optional): validation data의 비율입니다. Defaults to 0.2.
        seed (int, optional): sampling 시 사용할 seed 값입니다. Defaults to 2025.

    Returns:
        tuple: (x_train, y_train, x_validation, y_validation)
    """
    if method == 'sklearn':
        x_tr, x_val, y_tr, y_val = train_test_split(tr_data.drop(columns=['임신 성공 여부'], axis=1),
                                                    tr_data['임신 성공 여부'],
                                                    test_size=val_size,
                                                    random_state=seed,
                                                    shuffle=True)
    elif method == 'custom':
        x_tr = tr_data.copy()
        x_val = tr_data.copy()

        for code in tr_data['시술 시기 코드'].unique():
            index_lst = tr_data[tr_data['시술 시기 코드'] == code].index.values
            random.shuffle(index_lst)

            tr_index_lst = index_lst[int(len(index_lst) * val_size):]
            val_index_lst = index_lst[:int(len(index_lst) * val_size)]

            x_tr = x_tr.drop(index=val_index_lst)
            x_val = x_val.drop(index=tr_index_lst)

        y_tr = x_tr["임신 성공 여부"]
        x_tr = x_tr.drop(columns=["임신 성공 여부"], axis=1)

        y_val = x_val["임신 성공 여부"]
        x_val = x_val.drop(columns=['임신 성공 여부'], axis=1)
        
    return (x_tr, y_tr, x_val, y_val)


def pp_baseline(tr_data_path: str = None, tt_data_path: str = None,
                val_size: float = 0.2, seed: int = 2025) -> tuple:
    """대회 주최 측에서 제공한 기본 전처리 방식으로 동작하는 함수입니다.

    Args:
        tr_data_path (str, optional): 학습 데이터의 경로입니다. Defaults to None.
        tt_data_path (str, optional): 평가 데이터의 경로입니다. Defaults to None.
        val_size (float, optional): 평가 데이터의 크기입니다. Defaults to 0.2.
        seed (int, optional): 실험 재현을 위해 설정할 시드 값입니다. Defaults to 2025

    Returns:
        tuple: (pd.DataFrame, pd.DataFrame)
    """
    categorical_columns = [
        "시술 시기 코드", "시술 당시 나이", "시술 유형", "특정 시술 유형", "배란 자극 여부",
        "배란 유도 유형", "단일 배아 이식 여부", "착상 전 유전 검사 사용 여부", 
        "착상 전 유전 진단 사용 여부", "남성 주 불임 원인", "남성 부 불임 원인", "여성 주 불임 원인",
        "여성 부 불임 원인", "부부 주 불임 원인", "부부 부 불임 원인", "불명확 불임 원인",
        "불임 원인 - 난관 질환", "불임 원인 - 남성 요인", "불임 원인 - 배란 장애",
        "불임 원인 - 여성 요인", "불임 원인 - 자궁경부 문제", "불임 원인 - 자궁내막증",
        "불임 원인 - 정자 농도", "불임 원인 - 정자 면역학적 요인", "불임 원인 - 정자 운동성",
        "불임 원인 - 정자 형태", "배아 생성 주요 이유", "총 시술 횟수", "클리닉 내 총 시술 횟수",
        "IVF 시술 횟수", "DI 시술 횟수", "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
        "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수", "난자 출처", "정자 출처", "난자 기증자 나이",
        "정자 기증자 나이", "동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부",
        "대리모 여부", "PGD 시술 여부", "PGS 시술 여부"
        ]
    
    numerical_columns = [
        "임신 시도 또는 마지막 임신 경과 연수", "총 생성 배아 수", "미세주입된 난자 수",
        "미세주입에서 생성된 배아 수", "이식된 배아 수", "미세주입 배아 이식 수", "저장된 배아 수",
        "미세주입 후 저장된 배아 수", "해동된 배아 수", "해동 난자 수", "수집된 신선 난자 수",
        "저장된 신선 난자 수", "혼합된 난자 수", "파트너 정자와 혼합된 난자 수", 
        "기증자 정자와 혼합된 난자 수", "난자 채취 경과일", "난자 해동 경과일",
        "난자 혼합 경과일", "배아 이식 경과일", "배아 해동 경과일"
        ]
    
    tr_df, tt_df = load_data(tr_data_path, tt_data_path)
    tr_df, tt_df = drop_features(
        tr_df, tt_df, categorical_columns + numerical_columns
    )
    tr_df, tt_df = categorical_to_numerical(tr_df, tt_df, categorical_columns)

    # 수치형 컬럼들에 대해 결측치 처리
    tr_df[numerical_columns] = tr_df[numerical_columns].fillna(0)
    tt_df[numerical_columns] = tt_df[numerical_columns].fillna(0)

    # split
    x_tr, y_tr, x_val, y_val = split_train_and_validation(tr_df, val_size,
                                                          seed, method='custom')

    return ((x_tr, y_tr), (x_val, y_val), tt_df)

