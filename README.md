# Instacart Reorder Prediction
고객이 이전에 구매한 상품 중 **다음(마지막) 주문에 다시 담을 상품**을 예측하는 프로젝트입니다.
Kaggle Instacart 공개 구매 데이터를 사용했고, LightGBM과 사용자 단위 교차검증으로 평가했습니다.

## Problem
- **Task:** (user, product) 쌍이 사용자의 마지막 주문에 포함되는지 이진 분류
- **Candidate:** 사용자가 과거 주문에서 한 번이라도 구매한 상품 (신규 상품 추천은 범위 밖)
- **Why:** 재구매 가능성이 높은 상품을 고객별로 순위화하면 리마인드/추천/재고 계획에 활용 가능

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
- **Full:** 8 features + 9개 (총 17개)
- **Extended:** Full + 3개 (총 20개)

## Method

- **Model:** LightGBM (`n_estimators=200`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=50`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=3.5`)
- **Validation:** `GroupKFold(5)`, `user_id` 기준 분할 (같은 사용자가 train과 validation에 동시에 들어가지 않음)
- **Baseline:** `buy_rate` 단독 점수의 AUC (과거 주문 중 해당 상품을 산 비율)
- **Metrics:** AUC, AP(Average Precision), F1, Precision, Recall, MCC (분류 임계값 0.5)
- **Interpretation:** SHAP (TreeExplainer, 2,000개 샘플)

## Results

5-fold 평균입니다.

| 모델 | AUC | AP |
|---|---|---|
| Baseline (`buy_rate` 단독) | 0.773 | - |
| 8 features | 0.827 | 0.401 |
| Full (17 features) | 0.834 | 0.427 |
| Extended (20 features) | 0.834 | 0.428 |

Full 모델의 임계값 0.5 기준 성능입니다.

| F1 | Precision | Recall | MCC |
|---|---|---|---|
| 0.444 | 0.402 | 0.496 | 0.379 |

**해석**

- 단순 규칙(`buy_rate`)보다 AUC가 **+0.061** 높습니다 (0.773 → 0.834).
- 양성 비율이 약 9.8%이므로 랜덤 AP 기준선은 약 0.098입니다. Full 모델의 AP 0.427은 이의 **약 4.4배**입니다.
- 핵심 피처 8개만으로 Full 모델의 AUC에 근접합니다 (0.827 vs 0.834).
- Extended에서 추가한 3개 피처는 성능을 개선하지 못했습니다 (AUC 0.834로 동일). 사용자-상품 구매 이력 기반 피처는 이미 충분히 포착된 것으로 보입니다.

## Limitations

- 후보를 과거 구매 상품으로 한정했기 때문에 신규 상품 추천은 다루지 않습니다.
- 분류 임계값을 0.5로 고정했고, `scale_pos_weight` 사용으로 예측 확률은 보정되지 않았습니다.
- `target_*` 피처는 예측 대상 주문의 시점 정보입니다. 주문 시점에 알 수 있는 정보지만, 이 피처가 없을 때의 성능 비교(ablation)는 아직 하지 않았습니다.
- 상품의 실제 마지막 구매 후 경과일(일 단위), aisle/department 정보는 사용하지 않았습니다.
- 시간 순서 분할이 아니라 사용자 단위 분할이므로 시간 경과에 따른 성능 변화는 평가하지 않았습니다.

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
