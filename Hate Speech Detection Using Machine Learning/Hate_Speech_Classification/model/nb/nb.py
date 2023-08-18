naivebayes = MultinomialNB()
naivebayes.fit(X_train_cv1, y_train)
print('naive bayes done')

# Create a list of F1 score of all models
f1_score_data = {'F1 Score':[f1_score(naivebayes.predict(X_test_cv1), y_test)]}

# Create DataFrame with the model names as column labels
df_f1 = pd.DataFrame(f1_score_data, index=['Naive Bayes'])
