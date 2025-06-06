# BTP-keyword_spotting-in-speech
2025-06-06 13:21:36,271 - DataIngestion - INFO - Loaded 64721 audio files with 30 classes
2025-06-06 13:21:36,366 - DataIngestion - INFO - Data split into training and testing sets
2025-06-06 13:21:36,367 - KeywordSpottingPipeline - INFO - Number of classes: 30
2025-06-06 13:35:36,743 - ModelTrainer - INFO - Epoch 1/10, Classification Loss: 3.1799, Denoising Loss: 2167.7709, Accuracy: 8.95%
2025-06-06 13:35:36,743 - ModelTrainer - INFO - Test Accuracy: 9.25%
2025-06-06 13:39:18,218 - DataIngestion - INFO - Loaded 64721 audio files with 30 classes
2025-06-06 13:39:18,310 - DataIngestion - INFO - Data split into training and testing sets
2025-06-06 13:39:18,312 - KeywordSpottingPipeline - INFO - Number of classes: 30
2025-06-06 13:39:20,134 - ModelTrainer - INFO - Device in use cpu
2025-06-06 13:56:55,161 - DataIngestion - INFO - Loaded 64721 audio files with 30 classes
2025-06-06 13:56:55,250 - DataIngestion - INFO - Data split into training and testing sets
2025-06-06 13:56:55,251 - KeywordSpottingPipeline - INFO - Number of classes: 30
2025-06-06 13:56:59,169 - ModelTrainer - INFO - Device in use cuda
2025-06-06 14:01:11,847 - ModelTrainer - INFO - Epoch 1/10, Classification Loss: 3.2450, Denoising Loss: 2071.2359, Accuracy: 7.69%
2025-06-06 14:01:11,848 - ModelTrainer - INFO - Test Accuracy: 10.34%
2025-06-06 14:04:57,503 - ModelTrainer - INFO - Epoch 2/10, Classification Loss: 2.8358, Denoising Loss: 221.5153, Accuracy: 15.87%
2025-06-06 14:04:57,503 - ModelTrainer - INFO - Test Accuracy: 15.44%
2025-06-06 14:08:43,175 - ModelTrainer - INFO - Epoch 3/10, Classification Loss: 2.4315, Denoising Loss: 16.9666, Accuracy: 24.65%
2025-06-06 14:08:43,175 - ModelTrainer - INFO - Test Accuracy: 27.80%
2025-06-06 14:12:29,625 - ModelTrainer - INFO - Epoch 4/10, Classification Loss: 1.9175, Denoising Loss: 15.6102, Accuracy: 40.28%
2025-06-06 14:12:29,625 - ModelTrainer - INFO - Test Accuracy: 35.89%
2025-06-06 14:16:16,045 - ModelTrainer - INFO - Epoch 5/10, Classification Loss: 1.4868, Denoising Loss: 15.5646, Accuracy: 53.48%
2025-06-06 14:16:16,045 - ModelTrainer - INFO - Test Accuracy: 51.54%
2025-06-06 14:20:03,779 - ModelTrainer - INFO - Epoch 6/10, Classification Loss: 1.1719, Denoising Loss: 15.2745, Accuracy: 63.29%
2025-06-06 14:20:03,779 - ModelTrainer - INFO - Test Accuracy: 67.08%
2025-06-06 14:23:53,713 - ModelTrainer - INFO - Epoch 7/10, Classification Loss: 0.9423, Denoising Loss: 15.0793, Accuracy: 70.56%
2025-06-06 14:23:53,713 - ModelTrainer - INFO - Test Accuracy: 57.98%
2025-06-06 14:27:43,615 - ModelTrainer - INFO - Epoch 8/10, Classification Loss: 0.8138, Denoising Loss: 15.0306, Accuracy: 74.48%
2025-06-06 14:27:43,615 - ModelTrainer - INFO - Test Accuracy: 67.18%
2025-06-06 14:31:32,452 - ModelTrainer - INFO - Epoch 9/10, Classification Loss: 0.6993, Denoising Loss: 14.9418, Accuracy: 78.10%
2025-06-06 14:31:32,452 - ModelTrainer - INFO - Test Accuracy: 78.42%
2025-06-06 14:35:21,483 - ModelTrainer - INFO - Epoch 10/10, Classification Loss: 0.6311, Denoising Loss: 14.8596, Accuracy: 80.08%
2025-06-06 14:35:21,483 - ModelTrainer - INFO - Test Accuracy: 79.22%
2025-06-06 14:35:21,513 - ModelTrainer - INFO - Model saved to /home/dinesh/Documents/GitHub/BTP-keyword_spotting-in-speech/model1.pth
2025-06-06 14:36:02,136 - ModelEvaluator - INFO - Evaluation Results: Accuracy: 79.21%
2025-06-06 14:36:02,139 - ModelEvaluator - INFO - Confusion Matrix:
[[193   9   6   9   1   3   2   2  17   4   0   2   2   2  46   4   1   1
   19   2   0   0   1   1   0   0  12   2   2   0]
 [  4 270   0   8   0   5   0  13   8   0   0   0   0   2  11   0   2   0
    5   0   0   0   0  10   2   2   1   0   2   1]
 [  1   0 288   1   0   1   1   0  11   2  10   0   0   0   4   0   0   0
    1   1   0   3  12   0   0   1   6   0   4   0]
 [  0   2  10 221  33   0   3   8  40   0   0   0   0   4  22   0   1   0
    0   0   0   0   3   0   0   0   1   0   1   0]
 [  0   1   2   2 308   1   0   0  45   1   0   0   0   3  91   0   1   0
    0   1   0   1   1   0   0   2   2   1   0   9]
 [  1   2   7   0   1 413   0   1   4   2   1   1   0   0   0   0   1   0
    3   0   0  14   0   2   0   9   5   1   1   1]
 [  1   1  11   7   7   0 318  15   4   0   2   0   1  33  11   0  10   5
    9   2   3   1   9   3   0   1   1  11   2   3]
 [  0   1   0   0   0   1   0 429   5   0   0   0   0   0   1   0   2   4
    0   0   0   1   9   6   1   5   1   0   2   7]
 [  0   0   4   3   8   1   0   5 331   0   0   0   0   1 104   0   1   0
    0   0   0   0   3   3   0   3   1   0   2   4]
 [  0   0   2   0   0   2   0   0   2 301  11   0   0   0   0   0   0   0
    1   6   0   3   5   1   2   1  10   0   0   1]
 [  0   0   5   0   0   0   0   0   3   4 327   0   0   0   6   0   0   0
    0   0   0   0   0   0   0   2   3   0   0   0]
 [  1   0   6   0   0   1   2   4   2   0   0 334   0   0   6   1   0  13
   51   0   0   1   0   0   0   4   5   5  33   2]
 [  0   0   0   0   7   0   3   0   0   1   0   0 257  29  14   0   0  24
    1   3   0   0   0   0   0   0   1   5   0   4]
 [  0   1   1   2  10   0   3   1   2   1   0   0   3 367  49   0   2   9
    6   0   0   0   3   1   0   0   3   6   0   3]
 [  0   1   0   0   1   0   0   0  12   0   0   1   0   0 450   0   0   0
    0   0   0   0   3   0   0   1   2   0   1   3]
 [  1   1  17   1   1   1   8   6   2   0  17   3   0   0   3 343  10   0
    0   0   0   0   9   0   0   1  33   0  14   0]
 [  3  11   0  17   4   0  71  56  15   0   0   2   2  21   6   3 245   5
    0   1   0   0   4   2   0   2   2   1   0   0]
 [  0   0   0   0   1   1   2   3   1   0   0   0   1  22   9   0   0 403
   18   0   0   0   1   1   0   1   3   3   2   2]
 [  1   0   1   0   0   2   2   1   0   1   0  11   2  14  10   0   1  10
  403   0   1   0   3   0   0   0   5   3   2   0]
 [  0   0   1   0   2   0   1   0   6  10   1   0   4   0   1   0   0   0
    0 419   3   3   8   0   0   5   1   1   0  10]
 [  0   0   1   0   0   1   0  12   0   0   0   0   0   0   2   0   1   1
    0   3 210   3   7   3   4   6   0   0   4  89]
 [  0   0   6   0   0   4   0   0   0   4   7   1   0   0   0   0   1   1
    3   2   1 423   6   1   1   2   8   1   1   1]
 [  0   0  10   2   0   0   1   4   2   0   0   0   0   0   3   1   0   1
    0   0   0   3 444   0   0   3   1   1   0   0]
 [  0   0   2   0   0   2   1   4   3   0   0   0   0   1   0   0   0   0
    5   0   1   1   2 314 112  16   1   0   1   5]
 [  0   0   2   0   0   2   0   0   0   0   0   0   0   0   0   0   1   0
    0   0   0   0   0  19 302  15   1   0   0   5]
 [  0   0   1   0   0   1   0   3  25   0   1   1   0   0   1   0   1   0
    0   0   0   0   2   1   7 403   1   0   2  25]
 [  0   0  20   1   0   0   2   0   0   5   4   0   0   0   6   3   2   0
    1   0   0   2  35   0   0   2 392   0   0   0]
 [  0   0   1   0   3   0   1   0   0   0   0   0   1   5  64   0   0   9
    0   0   0   0   1   0   0   0   0 263   0   1]
 [  0   0   1   0   1   1   1   1   0   0   5   2   0   0  21   0   0   2
    0   0   2   1   0   0   0   3   0   0 433   2]
 [  0   0   0   0   0   0   0   2   1   0   0   0   0   0   6   0   0   0
    1   3   2   0   2   1   1   3   3   0   0 450]]
2025-06-06 14:36:02,140 - ModelEvaluator - INFO - Classification Report:
              precision    recall  f1-score   support

           0       0.94      0.56      0.70       343
           1       0.90      0.78      0.84       346
           2       0.71      0.83      0.77       347
           3       0.81      0.63      0.71       349
           4       0.79      0.65      0.72       472
           5       0.93      0.88      0.90       470
           6       0.75      0.68      0.71       471
           7       0.75      0.90      0.82       475
           8       0.61      0.70      0.65       474
           9       0.90      0.86      0.88       348
          10       0.85      0.93      0.89       350
          11       0.93      0.71      0.81       471
          12       0.94      0.74      0.83       349
          13       0.73      0.78      0.75       473
          14       0.48      0.95      0.63       475
          15       0.97      0.73      0.83       471
          16       0.87      0.52      0.65       473
          17       0.83      0.85      0.84       474
          18       0.76      0.85      0.81       473
          19       0.95      0.88      0.91       476
          20       0.94      0.61      0.74       347
          21       0.92      0.89      0.91       474
          22       0.77      0.93      0.85       476
          23       0.85      0.67      0.75       471
          24       0.70      0.87      0.78       347
          25       0.82      0.85      0.83       475
          26       0.78      0.83      0.80       475
          27       0.87      0.75      0.81       349
          28       0.85      0.91      0.88       476
          29       0.72      0.95      0.82       475

    accuracy                           0.79     12945
   macro avg       0.82      0.79      0.79     12945
weighted avg       0.82      0.79      0.79     12945

2025-06-06 14:36:02,310 - KeywordSpottingPipeline - INFO - Pipeline execution completed
