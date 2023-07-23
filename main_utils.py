#### Importing libraries ---------------------------------------------------------------------
import numpy as np
import pandas as pd

import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('wordnet_ic')

### Functions and Objects ---------------------------------------------------------------------

punctuations = string.punctuation

lemmatizer = WordNetLemmatizer()
def lemmatize(word):
    return lemmatizer.lemmatize(word.lower())

stop_words=nltk.corpus.stopwords.words('english')

tfidf = TfidfVectorizer()

# Similarity finding function -> Fitting & Transforming over training data
def similarity_calculation_function(preprocessed_data,similarities):
    for i in range(len(preprocessed_data)):
        doc1 = preprocessed_data['text1'][i]
        doc2 = preprocessed_data['text2'][i]
        
        corpus = [doc1,doc2]
        matrix = tfidf.fit_transform(corpus)
        sim_score = ((matrix * matrix.T).A)[0,1]
        similarities.append(sim_score)
    return similarities

# Similarity finding function -> Transforming & Testing over test data
def similarity_calculation_function_for_input(prepped_ip_data,ip_similarities):
    for i in range(len(prepped_ip_data)):
        doc1 = prepped_ip_data['text1'][i]
        doc2 = prepped_ip_data['text2'][i]
        
        corpus = [doc1,doc2]
        matrix = tfidf.transform(corpus)
        sim_score = ((matrix * matrix.T).A)[0,1]
        ip_similarities.append(sim_score)
    return ip_similarities

# Function to convert input texts into a Dataframe
def to_df_input_data(text1,text2):
    text1 = [text1]
    text2 = [text2]
    input_df = pd.DataFrame(list(zip(text1, text2)), columns = ['text1', 'text2'])  
    return input_df


## Data Ingestion and Preprocessing ---------------------------------------------------------------------

df = pd.read_csv('Precily_Text_Similarity.csv')

# Preprocessor Function 
def preprocess_data(data_frame):
    
    # Creating a copy of the data
    data = data_frame.copy()
    
    # Replacing erronous punctuations with spaces
    for i in range(len(data['text1'])):
        t1 = data['text1'][i]
        t2 = data['text2'][i]
        data['text1'][i] = t1.replace('/',' ').replace('.', ' ').replace('-',' ').replace('!',' ').replace('%',' ').replace(',',' ')
        data['text2'][i] = t2.replace('/',' ').replace('.', ' ').replace('-',' ').replace('!',' ').replace('%',' ').replace(',',' ')
    
    # Tokenization
    data['text1']=data['text1'].apply(lambda x: word_tokenize(x))
    data['text2']=data['text2'].apply(lambda x: word_tokenize(x))
    
    # Removing stop words
    for i in range(len(data['text1'])):
        data['text1'][i] = [token for token in data['text1'][i] if token not in stop_words]
        data['text2'][i] = [token for token in data['text2'][i] if token not in stop_words]
        
    # Removing punctuations and Lemmatizing words
    for i in range(len(data['text1'])):
        data['text1'][i] = [lemmatize(token) for token in data['text1'][i] if token not in punctuations]
        data['text2'][i] = [lemmatize(token) for token in data['text2'][i] if token not in punctuations]
    
    # Removing Numbers
    data['text1']=data['text1'].apply(lambda x: [token for token in x if not token.isdigit() ])
    data['text2']=data['text2'].apply(lambda x: [token for token in x if not token.isdigit()])

    # Filtering out empty string tokens
    data['text1']=data['text1'].apply(lambda x: [token for token in x if token!=''])
    data['text2']=data['text2'].apply(lambda x: [token for token in x if token!=''])

    # Removing single letter words -> mostly meaningless
    data['text1']=data['text1'].apply(lambda x: [token for token in x if len(token)>1 ])
    data['text2']=data['text2'].apply(lambda x: [token for token in x if len(token)>1 ])
    
    data['text1']= data['text1'].apply(lambda x: TreebankWordDetokenizer().detokenize(x))
    data['text2']= data['text2'].apply(lambda x: TreebankWordDetokenizer().detokenize(x))
    
    return data


### Fitting and transforming on preprocessed dummy data -------------------------------------------------

dummy_data = df.copy()
preprocessed_data = preprocess_data(dummy_data)
# preprocessed_data.head()

similarities = []
similarities = similarity_calculation_function(preprocessed_data,similarities)
data_labeled = df.copy()
data_labeled['Similarity Score'] = similarities

### Fitting and transforming on Original data  -----------------------------------------------------------------

og_data = df.copy()

og_similarities = []
og_similarities = similarity_calculation_function(og_data,og_similarities)
og_data_labeled = df.copy()
og_data_labeled['Similarity Score'] = og_similarities

# print(sorted(og_data_labeled['Similarity Score'], key=None, reverse=True)[:5])

## Preprocessing input data and return similarity score  -----------------------------------------------------

def return_df_with_similarities_and_sim_score(text1,text2):
    
#     Converting text1,text2 into a df
    testing_df = to_df_input_data(text_1,text_2)
    
#     Preprocessing the testing_df
    prepped_input_df = preprocess_data(testing_df)
    
#     Creating empty list that will store similarity of our 2 texts
    ip_similarities = []
    
#     Calculating similarity of texts 
    ip_similarities = similarity_calculation_function_for_input(prepped_input_df,ip_similarities)
    
#     Creating a new df that will be returned -> text1, text2, similarity score of i/p data
    df_with_similarities = testing_df.copy()
    
#     Appending calculated similarity score to our df
    df_with_similarities['similarity score of i/p data'] = ip_similarities

#     Similarity score
    sim_score = df_with_similarities['similarity score of i/p data'][0]
    
    return df_with_similarities,sim_score


#### Validating using random data ---............----------------..................-.-.-.-.-.-.-.-.-..-.-.--.


text_1 = "The weather today is partly cloudy with a chance of scattered showers in the afternoon."
# input text 1

text_2 = "The forecast for today predicts some clouds and the possibility of isolated rain showers later in the day."
# input text 2

df_with_similarities,sim_score = return_df_with_similarities_and_sim_score(text_1,text_2)


print(sim_score)
