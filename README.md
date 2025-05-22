Text Sentiment Analysis – Project Summary

What I Learned About Text Processing:
-------------------------------------
- Text data is often messy and unstructured, requiring cleaning and normalization before analysis.
- Common text preprocessing steps include:
  - Removing punctuation and special characters.
  - Converting all text to lowercase.
  - Removing stopwords (common words like "and", "the", "is" that don't add meaning).
- Tokenization splits text into individual words for further analysis.
- The TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer is effective for converting text into numerical features that machine learning models can understand.
- Cleaning and proper preprocessing of text significantly improves the performance of sentiment analysis models.

Model Accuracy:
---------------
- The Logistic Regression model on my dataset achieved an accuracy of **66.67%**.

Examples of Tested Custom Reviews:
----------------------------------
1. Review: "The movie was boring and too long."
   - Prediction: NEGATIVE

2. Review: "Amazing performance and storyline!"
   - Prediction: POSITIVE

3. Review: "I loved the soundtrack and the visuals were stunning."
   - Prediction: POSITIVE

4. Review: "The plot was confusing and the ending made no sense."
   - Prediction: NEGATIVE

5. Review: "Outstanding cast and a gripping story."
   - Prediction: POSITIVE

6. Review: "Bad acting and lazy writing."
   - Prediction: NEGATIVE

These examples show that the model can recognize both positive and negative sentiments from new, unseen movie reviews.
