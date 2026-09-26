# Intent Classifier -- Held-out Evaluation

- Held-out accuracy: **0.622** (37 examples)
- Trained on 108 examples

## Accuracy by language

- amharic: 0.444  (n=9)
- amharic_latin: 0.667  (n=9)
- english: 0.684  (n=19)

## Classification report

```
              precision    recall  f1-score   support

        info       0.47      0.82      0.60        11
    register       0.73      0.89      0.80         9
      assist       0.75      0.38      0.50         8
    escalate       1.00      0.33      0.50         9

    accuracy                           0.62        37
   macro avg       0.74      0.60      0.60        37
weighted avg       0.72      0.62      0.60        37

```

## Confusion matrix (rows=true, cols=predicted)

| true \ pred | info | register | assist | escalate |
|---|---|---|---|---|
| info | 9 | 2 | 0 | 0 |
| register | 1 | 8 | 0 | 0 |
| assist | 5 | 0 | 3 | 0 |
| escalate | 4 | 1 | 1 | 3 |
