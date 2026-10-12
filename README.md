## market-prediction
Kaggle Instacart 공개 구매 데이터로 고객의 재구매 여부를 예측

### Method
- Features: 구매 빈도, 최근 구매 시점, 상품 인기도 등
- Model: LightGBM, GroupKFold(5) CV

### Result
- 8 features: AUC 0.827
- Full features: AUC 0.834
- SHAP으로 기여도 파악
