# Reuters News Data Classification using TF-IDF and Topic Modeling

The Thomson Reuters GHC Machine Learning/Natural Language Challenge: predict the news category based on the news content

# Code

### News_Classification.ipynb

1. Feature Engineering
```
    A. Tokenization
    B. Punctuation & Stopwords Removal
    C. Lemmatization
```

    
2. Text to Feature
```
    A. TF-IDF
    B. LDA Topic Modeling
    C. Word Embedding (Word2Vec/GloVe)
    D. Ensemble: TF-IDF + LDA
```

        
3. Training and Hyperparameter Tuning (Ranked by GridSearchCV Best Accuracy Score)
```
    A. SVM:  0.8947833775419982
    B. Stochastic Gradient Descen: 0.8890994063407857
    C. Logistic Regression: 0.8880889225716811
    D. Naive Bayes 0.8769736011115321
    E. XGBoost: 0.8676266262473159
    F. KNN: 0.8556271314892004
    G. Random Forest: 0.8505747126436781
```
    
  


### WordEmbedding.py
	A module to create word embedding for news data. 
	Source: word2vec-google-news-300; glove-wiki-gigaword-300

	To replace the BoW with Word Embeddings, simply import the module and create a WordEmbedding object.
	Three options to use the word embedding vectors: 
		1. Mean
		2. Sum
		3. IDF Weighted Mean


