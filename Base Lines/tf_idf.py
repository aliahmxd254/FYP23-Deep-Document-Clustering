import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans



corpus = []
doc_list_sequence = []

def ReadDocuments(dir_name):
    for Path in os.listdir(dir_name + '\\'): 
        file_path = os.getcwd() + f"\{dir_name}\\" + Path
        with open(file_path, 'r') as file: 
            FileContents = file.read()
            corpus.append(FileContents.lower())
            doc_list_sequence.append(Path)     

def in_wordnet(word):
    synsets = wordnet.synsets(word)
    return len(synsets) > 0

def contains_number(word):
    for char in word:
        if char.isnumeric():
            return True
    return False

def min_length_word(word):
    if  len(word) in [1,2]:
        return True
    return False

def custom_preprocessor(text):
    lematizer = WordNetLemmatizer()
    used_terms = {} # keep track of which terms have already been considered
    tokens = word_tokenize(text)
    filtered_tokens = []
    for word in tokens:
        if (not contains_number(word)) and (not min_length_word(word)) and (word not in stopwords.words('english')) and (in_wordnet(word)):
            lema_word = lematizer.lemmatize(word)
            if lema_word in used_terms.keys():
                continue
            else:
                used_terms[lema_word] = 0
                filtered_tokens.append(lema_word)
    return ' '.join(filtered_tokens)

def print_terms(terms):
    for term in terms:
        print(term)

def Purity_Score(true_labels, pred_labels):
    # Calculate the confusion matrix to compare true labels and cluster assignments
    confusion = confusion_matrix(true_labels, pred_labels)
    # Calculate the purity
    purity = np.sum(np.max(confusion, axis=0)) / np.sum(confusion)
    return purity

def KMeans_Labels(X, n, rstate_limit, true_labels):

    # Specify the number of clusters (you can choose an appropriate value)
    num_clusters = n
    
    # find centoids which give maximum purity
    purity_collection = {}
    for i in range(rstate_limit):
        clusters = KMeans(n_init='auto', n_clusters=num_clusters, random_state=i, init='k-means++').fit(X).labels_
        purity_collection[i] = Purity_Score(true_labels, clusters)
    
    max_rand_state = max(purity_collection, key=purity_collection.get)
    print(f"Maximum purity of {purity_collection[max_rand_state]} found on random state {max_rand_state}")

    # Create a KMeans model
    kmeans = KMeans(n_init='auto', n_clusters=num_clusters, random_state=max_rand_state, init='k-means++')
    # Fit the KMeans model to the TF-IDF data
    kmeans.fit(X)
    # Get the cluster assignments for each document
    cluster_assignments = kmeans.labels_
    
    return cluster_assignments

def Actual_Labels():
    actual_labels = {} # dictionary to store true assignments for each document | read sequence not followed
    label_path = os.getcwd() + '\\Doc50 GT\\'
    for labels_directory in os.listdir(label_path): # for each assignment folder
        actual_cluster = int(labels_directory[1]) # extract cluster label from directory name
        doc_labels = os.listdir(label_path + f"\\{labels_directory}") # for all document ids assigned to this cluster
        for doc in doc_labels:
            actual_labels[doc] = actual_cluster-1 # save cluster label
    
    label_seq = [] # save labels in order of documents read
    for doc in doc_list_sequence:
        label_seq.append(actual_labels[doc])
    return label_seq

def print_results(true_labels, predicted_labels, X):
    print("RESULTS:")
    print(f"Purity: {Purity_Score(true_labels, predicted_labels)}")
    print(f"Silhouette Score: {silhouette_score(X, predicted_labels)}")



def wrapperFunction():
    ReadDocuments('Doc50')
    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', preprocessor=custom_preprocessor)
    X = vectorizer.fit_transform(corpus)
    true_labels = Actual_Labels()
    predicted_labels = KMeans_Labels(X, 5, 1500, true_labels)
    print_results(true_labels, predicted_labels, X)
    return predicted_labels, X
if __name__ == '__main__':
    predictedLabels = wrapperFunction()
    # print_results(trueLabels, predictedLabels)
