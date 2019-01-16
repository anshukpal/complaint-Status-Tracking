# complaint-Status-Tracking
While handling customer complaints, it is hard to track the status of the complaint. To automate this process, build models that can automatically predict the complaint status (how the complaint was resolved) based on the complaint submitted by the consumer and other related meta-data.  Data Description The dataset consists of one file files: train.csv  Data Dictionary  Complaint-ID: Complaint Id Date received: Date on which the complaint was received Transaction-Type: Type of transaction involved Complaint-reason: Reason of the complaint Consumer-complaint-summary: Complaint filed by the consumer - Present in three languages :  English, Spanish, French Company-response: Public response provided by the company (if any) Date-sent-to-company: Date on which the complaint was sent to the respective department Complaint-Status: Status of the complaint (Target Variable) Consumer-disputes:If the consumer raised any disputes

# Analysis

1. Data Understanding, Analysis and Preparation
	a. There are total 5 unique complaint status. There is lot imbalance in data - almost 34k records for Closed with explanation status - which accounts for almost 80% of the overall training data.
	b. The status has been factorized and categorized
	c. Understanding the status prediction to be made, complaint summary would be only available at the first time as part of the complaint support process, hence that can be be used for prediction of status. Others like Consumer-disputes,Company-response,Transaction-Type,Date-received,Date-sent-to-company will not be available to the support at the initial stage. 
	d. Complaint-reason can be used for prediction- Note: Column was used to merge with complaint summary column, but there seemed to no improvemnt in accuracy
	e. Tokenization/string cleaning for dataset was done.
	f. Withing the training set - 70% was used as training and rest 30% for validation

2. Modelling
	a. Used Pipeline for getting all algos
	b. The unique words form complaint summary was used using the wieghted tf-idf - Got all the words.
	c. Algos used:
		1. LinearSVC (LinearSVC) - 76.32% (validation accuracy)
		2. Naive Bayes (MultinomialNB) - 79% (validation accuracy)
		3. Linear SVM (SGDClassifier)  - 79% (validation accuracy)
		4. Logistic Regression - 72% (validation accuracy)
		5. XGBoost - 80% (although 1% higher, but the amount of resources it takes up, not worth it)
	c. Used Keras - 79.8% (used Simple Model and LSTM)


