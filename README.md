# Sentiment-Analysis-Using-BERT-Logistic-Regression-Random-Forest
### The goal of this is to perform sentiment analysis on a Twitter dataset where the sentiment can be positive, negative, or neutral. You will explore different machine learning algorithms, apply various pre-processing techniques, and train models to achieve
- better accuracy. Special emphasis will be placed on using BERT for this task.
### Dataset:
- You are provided with a Twitter sentiment analysis dataset that includes tweets and their corresponding sentiment labels (positive,negative, neutral).
- The dataset has three sentiments namely, negative(-1), neutral(0), and positive(+1). It contains two fields for the tweet and label
### Exploring Machine Learning Algorithms:
- Train at least two traditional machine learning models (e.g., Logistic Regression, SVM, Random Forest) on the pre-processed data.
- Evaluate their performance using appropriate metrics such as accuracy, precision, recall, and F1-score.

### Using BERT for Sentiment Analysis:
- Load a pre-trained BERT model (e.g., `bert-base-uncased`) from the Hugging Face Transformers library.
- Add a classification layer on top of BERT to predict the sentiment labels.
- Fine-tune the BERT model on the training set. Use the validation set to tune hyperparameters and prevent overfitting.
- Implement techniques such as learning rate scheduling and gradient clipping to stabilize training.

### Evaluation:
- Evaluate the fine-tuned BERT model on the test set using the same metrics as above.
- Generate a confusion matrix to visualize the model’s performance across different sentiment classes.
- Compare the performance of the BERT model with the traditional machine learning models.
