import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from collections import Counter


import spacy

from xgboost import XGBClassifier

from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score



df = pd.read_csv('train.csv',encoding='utf-8-sig')
#df = df[pd.notnull(df['Consumer-complaint-summary'])]
#df = df[pd.notnull(df['Consumer-disputes'])]
#df = df[pd.notnull(df['Company-response'])]
#df = df[pd.notnull(df['Complaint-reason'])]
#df = df[pd.notnull(df['Transaction-Type'])]
df['Complaint-Details'] = df['Consumer-complaint-summary']
# df['Complaint-Status-Category'] = df['Complaint-Status'].factorize()[0]
#df['Transaction-Type-Category'] = df['Transaction-Type'].factorize()[0]

del df['Consumer-complaint-summary']
del df['Consumer-disputes']
del df['Company-response']
del df['Transaction-Type']
del df['Complaint-reason']
del df['Date-received']
del df['Date-sent-to-company']

# print(Counter(df["Complaint-Status-Category"]))

df['Complaint-Status'] = df['Complaint-Status'].astype('category')
# col = ['Complaint-ID','Complaint-Status', 'Consumer-complaint-summary','Consumer-disputes','Company-response','Complaint-reason','Transaction-Type','Complaint-Details','Complaint-Status-Category']
# df = df[col]
# df.columns = ['Complaint-ID','Complaint-Details','Complaint-Status-Category']


#df = df.values

import re 
def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\n", "", string)    
    string = re.sub(r"\r", "", string) 
    string = re.sub(r"[0-9]", "digit", string)
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()


X = []
for i in range(df.shape[0]):
    X.append(clean_str(df.iloc[i][2]))
Y = np.array(df["Complaint-Status"])



# X = df['Complaint-Details']
# Y = df['Complaint-Status-Category']

# X = df[:,[2,5]]
# Y = df[:,7]

# print(df[:,[2,5]])
# print(df[:,7])

# print(df.shape)
# print(X.shape)
# print(Y.shape)


# X = X.values.reshape(X.shape[1:])
# X = X.transpose()

#Creating training and Test Data
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=5)

# model = Pipeline([('vectorizer', CountVectorizer()),('tfidf', TfidfTransformer()),
#  ('clf', OneVsRestClassifier(LinearSVC(class_weight="balanced")))])
# #the class_weight="balanced" option tries to remove the biasedness of model towards majority sample
# parameters = {'vectorizer__ngram_range': [(1, 1), (1, 2),(2,2)],
#                'tfidf__use_idf': (True, False)}
# gs_clf_svm = GridSearchCV(model, parameters, n_jobs=-1)
# gs_clf_svm = gs_clf_svm.fit(X, Y)
# print(gs_clf_svm.best_score_)
# print(gs_clf_svm.best_params_)


#preparing the final pipeline using the selected parameters
model = Pipeline([('vectorizer', CountVectorizer(ngram_range=(1,2))),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', OneVsRestClassifier(LinearSVC(class_weight="balanced")))])

model.fit(X_train, y_train)
#evaluation on test data
pred = model.predict(X_test)
confusion_matrix(pred, y_test)
accuracy = accuracy_score(y_test, pred)
print("LinearSVC Accuracy: %.2f%%" % (accuracy * 100.0))


nb = Pipeline([('vect', CountVectorizer(stop_words='english')),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)
predictedNB = nb.predict(X_test)
nbaccuracy = accuracy_score(y_test, predictedNB)
print("NB Algo Accuracy: %.2f%%" % (nbaccuracy * 100.0))


sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=10, tol=None)),
               ])
sgd.fit(X_train, y_train)
predictedLinearSVM = sgd.predict(X_test)
lsvmaccuracy = accuracy_score(y_test, predictedLinearSVM)
print("Linear SVM Algo Accuracy: %.2f%%" % (lsvmaccuracy * 100.0))


logreg = Pipeline([('vect', CountVectorizer(stop_words={'english','spanish','french'})),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
logreg.fit(X_train, y_train)
predictedLog = logreg.predict(X_test)
logaccuracy = accuracy_score(y_test, predictedLog)
print("LogisticRegression Algo Accuracy: %.2f%%" % (logaccuracy * 100.0))

xgb = Pipeline([('vect', CountVectorizer(stop_words={'english','spanish','french'})),
                ('tfidf', TfidfTransformer()),
                ('clf', XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1)),
               ])
xgb.fit(X_train, y_train)
predictedxgb = xgb.predict(X_test)
xgbmaccuracy = accuracy_score(y_test, predictedxgb)
print("XGBoost Algo Accuracy: %.2f%%" % (xgbmaccuracy * 100.0))



testdf = pd.read_csv('test.csv',encoding='utf-8-sig')
#testdf = testdf[pd.notnull(testdf['Consumer-complaint-summary'])]
# testdf = testdf[pd.notnull(testdf['Consumer-disputes'])]
#testdf = testdf[pd.notnull(testdf['Company-response'])]
#testdf = testdf[pd.notnull(testdf['Complaint-reason'])]
#testdf = testdf[pd.notnull(testdf['Transaction-Type'])]
testdf['Complaint-Details'] = testdf['Consumer-complaint-summary']
#testdf['Complaint-Status-Category'] = testdf['Complaint-Status'].factorize()[0]
testdf['Transaction-Type-Category'] = testdf['Transaction-Type'].factorize()[0]

del testdf['Consumer-complaint-summary']
del testdf['Consumer-disputes']
del testdf['Company-response']
del testdf['Transaction-Type']
del testdf['Transaction-Type-Category']
del testdf['Complaint-reason']
del testdf['Date-received']
del testdf['Date-sent-to-company']

X1 = []
for i in range(testdf.shape[0]):
    X1.append(clean_str(testdf.iloc[i][1]))

predictedNB = nb.predict(X1)
testdf['Complaint-Status'] = predictedNB
del testdf['Complaint-Details'] 
testdf.to_csv('submission_nb.csv',index=False)




# prediction = pd.DataFrame(predictedNB, columns=['predictions']).to_csv('prediction.csv')

# predictedLinearSVM = sgd.predict(X1)
# print(predictedLinearSVM)
# pd.DataFrame(predictedLinearSVM, columns=['predictions']).to_csv('prediction.csv')

# predictedTestSVC = model.predict(X1)
# print(predictedTestSVC)


#Creating Vectors from text
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(X_train)

# #Creating tf-idf weight from documents
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# #Using Naive Bayes 
# clf_nb = MultinomialNB().fit(X_train_tfidf, y_train)
# predictedNB = clf_nb.predict(count_vect.transform(X_test))
# nbaccuracy = accuracy_score(y_test, predictedNB)
# print("NB Algo Accuracy: %.2f%%" % (nbaccuracy * 100.0))



# clf_xg = XGBClassifier().fit(X_train_tfidf, y_train)
# predictedXG = clf_xg.predict(count_vect.transform(X_test))
# xgaccuracy = accuracy_score(y_test, predictedXG)
# print("XG Algo Accuracy: %.2f%%" % (xgaccuracy * 100.0))


#Using SVM
# clf_svm = SVC(gamma='auto').fit(X_train_tfidf, y_train)
# predictedSVM = clf_svm.predict(count_vect.transform(X_test))
# svmaccuracy = accuracy_score(y_test, predictedSVM)
# print("SVM Algo Accuracy: %.2f%%" % (svmaccuracy * 100.0))

# fig = plt.figure(figsize=(10,6))
# df.groupby('Complaint-Status-Category').count().plot.bar(ylim=0)
# plt.show()

# category_id_df = df[['Product', 'category_id']].drop_duplicates().sort_values('category_id')
# category_to_id = dict(category_id_df.values)
# id_to_category = dict(category_id_df[['category_id', 'Product']].values)
# print(df.shape)
# # Removing rows , if your machine is not able to handle so much of rows
# df = df[:-350000]
# print(df.columns.values)
# print(df.iloc[0]['Consumer_complaint_narrative'])




# # to get features, can be used later
# # tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
# # features = tfidf.fit_transform(df.Consumer_complaint_narrative).toarray()
# # labels = df.category_id
# # print(features.shape)

# # to check coorelation can be used later
# # N = 2
# # for Product, category_id in sorted(category_to_id.items()):
# #   features_chi2 = chi2(features, labels == category_id)
# #   indices = np.argsort(features_chi2[0])
# #   feature_names = np.array(tfidf.get_feature_names())[indices]
# #   unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
# #   bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
# #   print("# '{}':".format(Product))
# #   print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
# #   print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))

# #Creating training and Test Data
# X_train, X_test, y_train, y_test = train_test_split(df['Consumer_complaint_narrative'], df['Product'], random_state = 0)

# #Creating Vectors from text
# count_vect = CountVectorizer(stop_words='english')
# X_train_counts = count_vect.fit_transform(X_train)

# #Creating tf-idf weight from documents
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# #Using Naive Bayes 
# clf_nb = MultinomialNB().fit(X_train_tfidf, y_train)
# predictedNB = clf_nb.predict(count_vect.transform(X_test))
# print('NB Algo Accuracy: ', np.mean(predictedNB == y_test))

# #Using Linear SVM 
# clf__linear_svm = LinearSVC().fit(X_train_tfidf, y_train)
# predictedLinearSVM = clf__linear_svm.predict(count_vect.transform(X_test))
# print('Linear SVM Algo Accuracy: ', np.mean(predictedLinearSVM == y_test))

# #Using SVM
# clf_svm = SVC(gamma='auto').fit(X_train_tfidf, y_train)
# predictedSVM = clf_svm.predict(count_vect.transform(X_test))
# print('SVM Algo Accuracy: ', np.mean(predictedSVM == y_test))