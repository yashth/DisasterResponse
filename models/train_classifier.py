import sys
import pandas as pd
import re
from sqlalchemy import create_engine
import sqlite3
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
nltk.download(['punkt', 'wordnet'])
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.metrics import classification_report


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql_table('CategorizedMessages', engine)
    categories = ['related','request','offer','aid_related','medical_help','medical_products','search_and_rescue','security','military','child_alone','water','food','shelter','clothing','money','missing_people','refugees','death','other_aid','infrastructure_related','transport','buildings','electricity','tools','hospitals','shops','aid_centers','other_infrastructure','weather_related','floods','storm','fire','earthquake','cold','other_weather','direct_report']
    X = df.message
    Y = df[categories]
    
    return X,Y,categories
    


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens
    


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize,)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'clf__estimator__bootstrap': [True, False],
        'clf__estimator__n_estimators': [5, 10]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    categories = ['related','request','offer','aid_related','medical_help','medical_products','search_and_rescue','security','military','child_alone','water','food','shelter','clothing','money','missing_people','refugees','death','other_aid','infrastructure_related','transport','buildings','electricity','tools','hospitals','shops','aid_centers','other_infrastructure','weather_related','floods','storm','fire','earthquake','cold','other_weather','direct_report']
    class_report = classification_report(Y_test, y_pred, target_names=categories)
    print(class_report)
    


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        print('Filepath for pickle -> {}'.format(database_filepath))

        X, Y, category_names = load_data(database_filepath)
        print('Y.head()=\n',Y.head())
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        print('model...',model)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()