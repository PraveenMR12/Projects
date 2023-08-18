def cv_tf_train_test(df_done, label, vectorizer, ngram):
    ''' Train/Test split'''
    # Split the data into X and y data sets
    X = df_done.comment_text
    y = df_done[label]

    # Split our data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    ''' Count Vectorizer/TF-IDF '''
    # Create a Vectorizer object and remove stopwords from the table
    cv1 = vectorizer(ngram_range=(ngram), stop_words='english')
    X_train_cv1 = cv1.fit_transform(X_train) # Learn the vocabulary dictionary and return term-document matrix
    X_test_cv1 = cv1.transform(X_test)      # Learn a vocabulary dictionary of all tokens in the raw documents.

    ''' Initialize all model objects and fit the models on the training data '''
    svm = LinearSVC()
    svm.fit(X_train_cv1, y_train)
    print('SVM done')

    # Create a list of F1 score of all models
    f1_score_data = {'F1 Score':[f1_score(svm.predict(X_test_cv1), y_test)]}

    # Create DataFrame with the model names as column labels
    df_f1 = pd.DataFrame(f1_score_data, index=['Support Vector Machine'])

    return df_f1
