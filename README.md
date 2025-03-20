# 25-LG-Aimers-06-Phase2
### 📢 대회 개요
- **주제**: 난임 환자 대상 임신 성공 여부 예측 AI 온라인 해커톤
- **기간**: 2025.02.01 ~ 02.27 (약 4주)
- **주최**: LG AI Research
<br>

### 👪 팀원
- 김건희
- 김성원
- 박도연
- 박수영
- 양진우
<br>


### 🔬 최종 전략
- **구현 코드:** [02_autogluon.ipynb](https://github.com/nstalways/25-LG-Aimers-06-Phase2/blob/main/main/02_autogluon.ipynb)
  - `AutoGluon` 기반 모델링
<br>

### 🏅 대회 결과
|  | ROC-AUC | 등수 |
| :-: | :-: | :-: |
| public | 0.74157 | 113 / 794 |
| private | 0.742 | 72 / 794 |
<br>

### 추가 내용
오프라인 해커톤에 진출한 팀들의 코드를 살펴보니
AutoGluon을 사용하였지만 그 전에 feature engineering을 통해 일부 특성은 지우고 파생 변수를 만들어 진행하였다. 
점수 차이는 0.002
