# Instacart Reorder Prediction
고객이 이전에 구매한 상품 중 **다음(마지막) 주문에 다시 담을 상품**을 예측하는 프로젝트입니다.
Kaggle Instacart 공개 구매 데이터를 사용했고, LightGBM과 사용자 단위 교차검증으로 평가했습니다.

## Highlights

- **Baseline 대비 개선:** 단순 규칙(`buy_rate`) AUC 0.773 → LightGBM 0.834
- **Top-K 활용 관점:** 사용자별 상위 10개 추천만으로 재구매 상품의 약 **58%** 포착 (Recall@10 = 0.583)
- **검증 설계:** `user_id` 기준 GroupKFold(5)로 사용자 간 누수 방지
- **가설 검증:** 주문 시점 피처 제거(ablation), 실제 경과일·aisle 피처 추가 실험을 수행했고, 개선이 미미함을 확인해 대표 모델은 17개 피처로 유지

## Problem

- **Task:** (user, product) 쌍이 사용자의 마지막 주문에 포함되는지 이진 분류
- **Candidate:** 사용자가 과거 주문에서 한 번이라도 구매한 상품 (신규 상품 추천은 범위 밖)
- **Why:** 재구매 가능성이 높은 상품을 고객별로 순위화하면 리마인드, 추천, 재고 계획에 활용할 수 있습니다.

## Data

[Instacart Market Basket Analysis (Kaggle)](https://www.kaggle.com/c/instacart-market-basket-analysis/data)

| 항목 | 값 |
|---|---|
| 사용자 수 (train set 대상) | 131,209 |
| 과거(prior) 주문 상품 행 수 | 20,641,991 |
| 후보 (user, product) 쌍 | 8,474,661 |
| 양성 비율 | 9.78% |

- 라벨: 사용자의 `train` 주문에 해당 상품이 포함되면 1, 아니면 0
- 상품·사용자 통계는 `prior` 주문만으로 계산했습니다. 라벨이 되는 `train` 주문의 구매 내용은 피처 계산에 쓰지 않았습니다.

## Features

| 그룹 | 피처 |
|---|---|
| User-Product | `times_bought`, `last_order`, `avg_cart_pos`, `buy_rate`, `orders_since_last`, `cnt_last3`, `avg_gap`, `due_ratio` |
| User | `user_orders`, `user_reorder_ratio`, `user_basket`, `user_avg_days` |
| Product | `prod_count`, `prod_reorder` |
| Order context | `target_days`, `target_dow`, `target_hour` (예측 대상 주문의 이전 주문 후 경과일, 요일, 시간) |
| Extended | `cnt_last5`, `first_order_ratio`, `user_prod_share` |

- **8 features:** `times_bought`, `last_order`, `avg_cart_pos`, `user_orders`, `buy_rate`, `orders_since_last`, `prod_count`, `prod_reorder`
- **Full (17):** 8 features + 9개 (대표 모델)
- **Extended (20):** Full + 3개

## Method

- **Model:** LightGBM (`n_estimators=200`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=50`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=3.5`)
- **Validation:** `GroupKFold(5)`, `user_id` 기준 분할 (같은 사용자가 train과 validation에 동시에 들어가지 않음)
- **Baseline:** `buy_rate` 단독 점수의 AUC (과거 주문 중 해당 상품을 산 비율)
- **Metrics:** AUC, AP(Average Precision), F1, Precision, Recall, MCC (분류 임계값 0.5), 사용자별 Precision@K / Recall@K
- **Interpretation:** SHAP (TreeExplainer, 2,000개 샘플)

## Results

5-fold 평균입니다.

| 모델 | AUC | AP |
|---|---|---|
| Baseline (`buy_rate` 단독) | 0.773 | - |
| 8 features | 0.827 | 0.401 |
| **Full (17 features)** | **0.834** | **0.427** |
| Extended (20 features) | 0.834 | 0.428 |

Full 모델의 임계값 0.5 기준 성능입니다.

| F1 | Precision | Recall | MCC |
|---|---|---|---|
| 0.444 | 0.402 | 0.496 | 0.379 |

**해석**

- 단순 규칙(`buy_rate`)보다 AUC가 **+0.061** 높습니다 (0.773 → 0.834).
- 양성 비율이 약 9.8%이므로 랜덤 AP 기준선은 약 0.098입니다. Full 모델의 AP 0.427은 이의 **약 4.4배**입니다.
- 핵심 피처 8개만으로 Full 모델의 AUC에 근접합니다 (0.827 vs 0.834).
- Extended에서 추가한 3개 피처는 성능을 개선하지 못했습니다 (AUC 0.834로 동일).
- 절대 F1(0.444)은 높지 않습니다. 다음 장바구니 구성은 그날의 상황에 따라 변동이 큰 문제여서 구매 이력만으로 설명할 수 있는 부분에 한계가 있다고 판단합니다. 그래서 단일 임계값 지표보다 아래의 Top-K 관점이 실제 활용에 더 적합합니다.

### Threshold

OOF 예측(5-fold)을 모아 임계값 0.10~0.88을 탐색한 결과, F1 최적 임계값은 **0.50**이었습니다 (F1 0.444). `scale_pos_weight=3.5`가 이미 클래스 불균형을 보정하고 있어 기본값과 일치한 것으로 해석합니다.

### Top-K (사용자별 상위 K개 추천)

| K | Precision@K | Recall@K |
|---|---|---|
| 5 | 0.390 | 0.410 |
| 10 | 0.309 | 0.583 |
| 20 | 0.231 | 0.763 |

- 랜덤 기준 precision(양성 비율 9.8%) 대비 K=5에서 약 4.0배, K=10에서 약 3.2배입니다.
- Recall@K는 마지막 주문에 재구매 상품이 하나라도 있는 사용자(122,607 / 131,209명, 93.4%)만 대상으로 계산했습니다. 나머지 사용자는 후보 정의상 정답을 맞힐 수 없습니다.
- 사용자별 상위 10개만 제시해도 재구매 상품의 약 58%를 포함합니다.

### Ablation & Feature Experiments

Top-K는 사용자별 상위 10개, 임계값은 0.5 기준이며 5-fold OOF 예측으로 계산했습니다.

| 실험 | 피처 수 | AUC | AP | F1 | P@10 | R@10 |
|---|---|---|---|---|---|---|
| A. Full (대표 모델) | 17 | 0.834 | 0.427 | 0.444 | 0.309 | 0.583 |
| B. Full − `target_*` | 14 | 0.831 | 0.416 | 0.437 | 0.308 | 0.582 |
| C. Full + 실제 경과일 3개 | 20 | 0.835 | 0.430 | 0.446 | 0.310 | 0.585 |
| D. Full + aisle/department 8개 | 25 | 0.835 | 0.428 | 0.445 | 0.310 | 0.585 |
| E. Full + 경과일 + aisle/dept | 28 | 0.836 | 0.431 | 0.447 | 0.311 | 0.587 |
| F. B + `avg_gap_days` + aisle/dept | 23 | 0.833 | 0.418 | 0.439 | 0.309 | 0.583 |

**Findings**

- **`target_*` (예측 대상 주문의 시점 정보):** 제거하면 AP가 0.011 감소하지만 Top-K는 거의 변하지 않습니다 (P@10 −0.001, R@10 −0.001). 이 피처는 사용자당 값이 하나라서 사용자 내부의 상품 순위는 바꾸지 못하고, 사용자 간 확률 수준을 맞추는 데 기여하는 것으로 해석합니다. 주문 시점에 알 수 있는 정보이므로 누수는 아닙니다.
- **실제 경과일, aisle/department 피처:** 개선 폭은 AUC +0.001~0.002, R@10 +0.002~0.004로 작았습니다. 기존의 주문 수 기반 피처가 구매 주기 정보를 이미 포착하고 있는 것으로 보여, 대표 모델은 더 단순한 Full(17)로 유지했습니다.
- 단일 시드, 5-fold 평균이므로 0.002 이하의 차이는 노이즈일 수 있습니다.

## Limitations

- 후보를 과거 구매 상품으로 한정했기 때문에 신규 상품 추천은 다루지 않습니다.
- `scale_pos_weight` 사용으로 예측 확률은 보정되지 않았습니다. 순위 기반 활용(Top-K)에는 영향이 없지만 확률값 자체를 해석할 때는 주의가 필요합니다.
- Instacart의 `days_since_prior_order`는 30일에서 잘려 있어(cap) 누적 경과일은 긴 공백을 과소추정합니다.
- 사용자 단위 분할이므로 시간 경과에 따른 성능 변화(시간 순서 분할)는 평가하지 않았습니다.
- 모든 비교는 단일 시드(`random_state=42`)이며 신뢰구간을 계산하지 않았습니다.
- 트리 수와 학습률 등 하이퍼파라미터는 튜닝하지 않았습니다.

## Next Steps

- [x] 임계값 튜닝 (최적 0.50 확인)
- [x] Top-K 평가
- [x] `target_*` ablation
- [x] 실제 경과일, aisle/department 피처 추가 (효과 미미)
- [ ] 확률 보정 (`scale_pos_weight` 제거 후 비교)
- [ ] 하이퍼파라미터 튜닝 (early stopping)
- [ ] 시간 순서 기반 검증

## Repository Structure

```
.
├── prediction.ipynb     # 전체 파이프라인 (피처 생성, CV, SHAP, 추가 실험)
├── assets/              # README용 이미지 (shap_summary.png)
├── requirements.txt
└── README.md
```

## How to Run

```bash
git clone https://github.com/erdbwjd00/instacart-reorder-prediction.git
cd instacart-reorder-prediction
pip install -r requirements.txt
```

1. Kaggle에서 데이터를 받아 레포 루트에 압축을 풉니다 (`orders.csv`, `order_products__prior.csv`, `order_products__train.csv`, `products.csv`, `aisles.csv`, `departments.csv`).
2. `prediction.ipynb`를 엽니다.
3. 메모리가 부족하면 노트북의 `N = None`을 `N = 20000` 등으로 바꿔 일부 사용자만 사용하세요.
4. Restart & Run All

데이터 파일은 용량과 라이선스 때문에 레포에 포함하지 않았습니다.
