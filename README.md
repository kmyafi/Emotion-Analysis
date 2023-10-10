# Analisis & Klasifikasi Emosi Data Tweet Berbahasa Indonesia dengan Metode Ekstraksi Fitur
Multiclass classification task to perform emotion analysis of tweet using NLP

> **Objectives :**
> 1. Know effective ways to identify and classify emotions in Indonesian *tweets*.
> 2. Obtain a model with good performance and accuracy in classifying emotions in Indonesian *tweets* on Twitter.
> 3. Get prediction results for someone's emotions based on *tweets* that have been published.
>
> **Credits :**
> 1. Annisa Fitria Anwar Damanik
> 2. Nasywa Safira Ardanty
> 3. Kamal Muftie Yafi
> 4. Rifa Nayaka Utami

## Set
- **Dataset:** Emotion dataset from Indonesian tweet obtained from [Data Analysis Competition (DAC) Informatics Festival (IFest) Unpad](https://www.ifestunpad.id/data-analysis);
- **Slang:** Modified Kamus Alay based on [Kamus Alay (Colloquial Indonesian Lexicon)](https://github.com/nasalsabila/kamus-alay);
- **Feature Extraction:** `Bag-of-Words`, `TF-IDF`;
- **Classifier:** `Naive Bayes`, `SVM`, `Logistic Regression`, `Decision Tree`.

## About The Data
This dataset was formed from Indonesian tweet containing five emotion values, namely fear, joy, love, sadness, and anger. The total data in this dataset is 5,153 with the last 1,000 data unlabeled to predict. Each label has a varied amount of data distribution, including 654 data for fear, 1,002 data for joy, 498 data for love, 1,123 data for sadness, and 876 data for anger.

| **Emotion Label** | Fear |  Joy   | Love  | Sadness | Anger |
| ----------------- | :--: | :----: | :---: | :-----: | :---: |
| **Total Data**    | 654  | 1,002  | 498   |  1,123  |  876  |

## Algorithm included
- [x] Text cleaning/preprocessing
- [x] Non-standard word replacement
- [x] Feature extraction: *BoW*, *TF-IDF*
- [x] Classification: *Naive Bayes*, *SVM*, *Logistic Regression*, *Decision Tree*
- [x] Predicting

## Requirements
- `pandas`
- `regex`
- `matplotlib`
- `unidecode`
- `html`
- `unicodedata`
- `emoji`
- `string`
- `sklearn`
- `tqdm`
- `random`

## Modelling
### 1. Bag of Words (BoW) as Feature
#### **Naive Bayes**
Confusion matrix
```
Predicted    0    1   2   3    4
Actual                          
0          158   23   2   8   23
1           61  115   4   5   12
2           29    8  57   2    3
3           41    5   0  76    9
4           41    9   0   3  114
```
Classification report
```
              precision    recall  f1-score   support

           0       0.48      0.74      0.58       214
           1       0.72      0.58      0.64       197
           2       0.90      0.58      0.70        99
           3       0.81      0.58      0.68       131
           4       0.71      0.68      0.70       167

    accuracy                           0.64       808
   macro avg       0.72      0.63      0.66       808
weighted avg       0.69      0.64      0.65       808
```

#### **SVC**
Confusion matrix
```
Predicted    0    1   2   3    4
Actual                          
0          142   41   6   1   24
1           47  127   7   3   13
2           15   15  64   1    4
3           42   10   1  70    8
4           50   16   0   1  100
```
Classification report
```
              precision    recall  f1-score   support

           0       0.48      0.66      0.56       214
           1       0.61      0.64      0.63       197
           2       0.82      0.65      0.72        99
           3       0.92      0.53      0.68       131
           4       0.67      0.60      0.63       167

    accuracy                           0.62       808
   macro avg       0.70      0.62      0.64       808
weighted avg       0.66      0.62      0.63       808
```

#### **Logistic Regression**
Confusion matrix
```
Predicted    0    1   2   3    4
Actual                          
0          137   33   9   9   26
1           36  125  10   7   19
2           16   13  66   2    2
3           20   10   2  90    9
4           40   16   1   7  103
```
Classification report
```
              precision    recall  f1-score   support

           0       0.55      0.64      0.59       214
           1       0.63      0.63      0.63       197
           2       0.75      0.67      0.71        99
           3       0.78      0.69      0.73       131
           4       0.65      0.62      0.63       167

    accuracy                           0.64       808
   macro avg       0.67      0.65      0.66       808
weighted avg       0.65      0.64      0.65       808
```

#### **Decision Tree**
Confusion matrix
```
Predicted   0   1   2   3   4
Actual                       
0          83  43  16  16  56
1          38  91  17  12  39
2          18  10  65   5   1
3          15  10   4  92  10
4          36  31   4   9  87
```
Classification report
```
              precision    recall  f1-score   support

           0       0.44      0.39      0.41       214
           1       0.49      0.46      0.48       197
           2       0.61      0.66      0.63        99
           3       0.69      0.70      0.69       131
           4       0.45      0.52      0.48       167

    accuracy                           0.52       808
   macro avg       0.54      0.55      0.54       808
weighted avg       0.52      0.52      0.52       808
```

### 2. TFIDF Vectorizer as Feature
#### **Naive Bayes**
Confusion matrix
```
Predicted    0    1  2   3   4
Actual                        
0          186   18  0   0  10
1           94  101  0   0   2
2           73   19  6   0   1
3           93    3  0  31   4
4           83    8  0   0  76
```
Classification report
```
              precision    recall  f1-score   support

           0       0.35      0.87      0.50       214
           1       0.68      0.51      0.58       197
           2       1.00      0.06      0.11        99
           3       1.00      0.24      0.38       131
           4       0.82      0.46      0.58       167

    accuracy                           0.50       808
   macro avg       0.77      0.43      0.43       808
weighted avg       0.71      0.50      0.47       808
```

#### **SVC**
Confusion matrix
```
Predicted    0    1   2   3    4
Actual                          
0          154   32   6   0   22
1           45  134   4   2   12
2           20   12  65   1    1
3           38   10   0  74    9
4           50   16   0   1  100
```
Classification report
```
              precision    recall  f1-score   support

           0       0.50      0.72      0.59       214
           1       0.66      0.68      0.67       197
           2       0.87      0.66      0.75        99
           3       0.95      0.56      0.71       131
           4       0.69      0.60      0.64       167

    accuracy                           0.65       808
   macro avg       0.73      0.64      0.67       808
weighted avg       0.70      0.65      0.66       808
```

#### **Logistic Regression**
Confusion matrix
```
Predicted    0    1   2   3    4
Actual                          
0          144   34   6   5   25
1           36  139   4   3   15
2           17   11  67   1    3
3           29   10   0  83    9
4           41   16   0   2  108
```
Classification report
```
              precision    recall  f1-score   support

           0       0.54      0.67      0.60       214
           1       0.66      0.71      0.68       197
           2       0.87      0.68      0.76        99
           3       0.88      0.63      0.74       131
           4       0.68      0.65      0.66       167

    accuracy                           0.67       808
   macro avg       0.73      0.67      0.69       808
weighted avg       0.69      0.67      0.67       808
```

#### **Decision Tree**
Confusion matrix
```
Predicted   0   1   2   3   4
Actual                       
0          98  44  21  16  35
1          51  77  17  13  39
2          14  11  64   5   5
3          20   9   3  92   7
4          63  22   4  10  68
```
Classification report
```
              precision    recall  f1-score   support

           0       0.40      0.46      0.43       214
           1       0.47      0.39      0.43       197
           2       0.59      0.65      0.62        99
           3       0.68      0.70      0.69       131
           4       0.44      0.41      0.42       167

    accuracy                           0.49       808
   macro avg       0.52      0.52      0.52       808
weighted avg       0.49      0.49      0.49       808
```

### Comparison
|   Model   | F1 Score | Accuracy |
| :-------- | :------: | :------: |
| NB-BoW    |  64.36   |   64.36  |
| SVM-BoW   |  62.25   |   62.25  |
| LR-BoW    |  64.48   |   64.48  |
| DT-BoW    |  51.73   |   51.73  |
| NB-TfIdf  |  49.50   |   49.50  |
| SVM-TfIdf |  65.22   |   65.22  |
| LR-TfIdf  |  66.96   |   66.96  |
| DT-TfIdf  |  49.38   |   49.38  |
