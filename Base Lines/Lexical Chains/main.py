import os
import pickle
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from collections import defaultdict
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

corpus = []
totalFeatures = []
doc_list_sequence = []


def in_wordnet(word):
    synsets = wordnet.synsets(word)
    return len(synsets) > 0


def contains_number(word):
    for char in word:
        if char.isnumeric():
            return True
    return False


def min_length_word(word):
    if len(word) <= 2:
        return True
    return False


def custom_preprocessor(text):
    lematizer = WordNetLemmatizer()
    used_terms = {}  # keep track of which terms have already been considered
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    filtered_tokens = []
    for word in tokens:
        if (
            (not contains_number(word))
            and (not min_length_word(word))
            and (word not in stopwords.words("english"))
            and (in_wordnet(word))
        ):
            lema_word = lematizer.lemmatize(word)
            if lema_word in used_terms.keys():
                continue
            else:
                used_terms[lema_word] = 0
                filtered_tokens.append(lema_word)
    return filtered_tokens


def preprocess_text(text: str, remove_stopwords: bool) -> str:
    updated_token = custom_preprocessor(text)
    return updated_token


def buildRelation(nouns):
    relation_list = defaultdict(list)

    for k in range(len(nouns)):
        relation = []
        for syn in wordnet.synsets(nouns[k], pos=wordnet.NOUN):
            for l in syn.lemmas():
                relation.append(l.name())
                if l.antonyms():
                    relation.append(l.antonyms()[0].name())
            for l in syn.hyponyms():
                if l.hyponyms():
                    relation.append(l.hyponyms()[0].name().split(".")[0])
            for l in syn.hypernyms():
                if l.hypernyms():
                    relation.append(l.hypernyms()[0].name().split(".")[0])
        relation_list[nouns[k]].append(relation)
    return relation_list


def buildLexicalChain(nouns, relation_list):
    lexical = []
    threshold = 0.5
    for noun in nouns:
        flag = 0
        for j in range(len(lexical)):
            if flag == 0:
                for key in list(lexical[j]):
                    if key == noun and flag == 0:
                        lexical[j][noun] += 1
                        flag = 1
                    elif key in relation_list[noun][0] and flag == 0:
                        syns1 = wordnet.synsets(key, pos=wordnet.NOUN)
                        syns2 = wordnet.synsets(noun, pos=wordnet.NOUN)
                        if syns1[0].wup_similarity(syns2[0]) >= threshold:
                            lexical[j][noun] = 1
                            flag = 1
                    elif noun in relation_list[key][0] and flag == 0:
                        syns1 = wordnet.synsets(key, pos=wordnet.NOUN)
                        syns2 = wordnet.synsets(noun, pos=wordnet.NOUN)
                        if syns1[0].wup_similarity(syns2[0]) >= threshold:
                            lexical[j][noun] = 1
                            flag = 1
        if flag == 0:
            dic_nuevo = {}
            dic_nuevo[noun] = 1
            lexical.append(dic_nuevo)
            flag = 1
    return lexical


def eliminateWords(lexical):
    final_chain = []
    while lexical:
        result = lexical.pop()
        if len(result.keys()) == 1:
            for value in result.values():
                if value != 1:
                    final_chain.append(result)
        else:
            final_chain.append(result)
    return final_chain


def getallFiles():
    print(os.getcwd() + "\Doc50")
    file_paths = os.listdir(os.getcwd() + "\Doc50")
    return file_paths


def extractDataFromFiles():
    files = getallFiles()
    dataset = ""

    for i in files:
        path = r"Doc50"
        path = path + "\\" + i
        f = open(path, "r")
        dataset = preprocess_text(f.read(), remove_stopwords=True)
        # use lexical chains as the feature selection method
        nouns = []
        l = nltk.pos_tag(dataset)
        for word, n in l:
            if n == "NN" or n == "NNS" or n == "NNP" or n == "NNPS":
                nouns.append(word)

        relation = buildRelation(nouns)
        lexical = buildLexicalChain(nouns, relation)
        final_chain = eliminateWords(lexical)
        storeDocFeatures(i, final_chain)


def storeDocFeatures(filename, doc_dict):
    file_path = os.getcwd() + "\docFeatures\\" + filename + ".pkl"
    file = open(file_path, "wb")
    pickle.dump(doc_dict, file)
    file.close()


def buildVsmForDocuments():
    files = getallFiles()

    for i in files:
        a_file = open(os.getcwd() + "\docFeatures\\" + i + ".pkl", "rb")
        docFeaturesdict = pickle.load(a_file)
        for features in docFeaturesdict:
            for docFeature in features.keys():
                if docFeature not in totalFeatures:
                    totalFeatures.append(docFeature)

    # print(totalFeatures)
    print(len(totalFeatures))

    final_training_Features = []

    for i in files:
        a_file = open(
            os.getcwd() + "\docFeatures\\" + i + ".pkl",
            "rb",
        )

        docFeaturesdict = pickle.load(a_file)

        print(a_file)
        print(docFeaturesdict)

        temp = []
        for j in totalFeatures:
            check = False
            for features in docFeaturesdict:
                if j in features.keys():
                    temp.append(features[j])
                    check = True
                    break
            if not check:
                temp.append(0)

        final_training_Features.append(temp)

    print("final training features")
    print(len(final_training_Features))
    return final_training_Features


def storeDocumentVectors(documentVectors):
    file_path = os.getcwd() + "\docFeatures\documentVectors.pkl"
    file = open(file_path, "wb")
    pickle.dump(documentVectors, file)
    file.close()


def readDocumentVectors():
    file_path = os.getcwd() + "\docFeatures\documentVectors.pkl"
    a_file = open(file_path, "rb")
    X = pickle.load(a_file)
    a_file.close()
    return X


def Purity_Score(label_seq, pred_labels):
    # Calculate the confusion matrix to compare true labels and cluster assignments
    confusion = confusion_matrix(label_seq, pred_labels)
    # Calculate the purity
    purity = np.sum(np.max(confusion, axis=0)) / np.sum(confusion)
    return purity


def kMeansClustering(X, maxClusters):
    # # initialize KMeans with 5 clusters
    K = 5
    print("Number of clusters = " + str(K))
    kmeans = KMeans(n_clusters=K, init="k-means++", random_state=42)
    y = kmeans.fit(X)
    clusters = kmeans.labels_

    # Getting the Centroids

    WCSS = kmeans.inertia_
    labels_pred = kmeans.labels_
    print()
    print("K-means labels:\n")
    print(kmeans.labels_)
    print("\nWithin-Cluster Sum-of-Squares: " + str(WCSS))
    silhouette = metrics.silhouette_score(X, labels_pred, metric="euclidean")
    print("Silhouette Coefficient: " + str(silhouette))

    # Purity Score
    for Path in os.listdir("Doc50" + "\\"):
        file_path = os.getcwd() + f"\{'Doc50'}\\" + Path
        with open(file_path, "r") as file:
            FileContents = file.read()
            corpus.append(FileContents.lower())
            doc_list_sequence.append(Path)

    actual_labels = (
        {}
    )  # dictionary to store true assignments for each document | read sequence not followed

    label_path = os.getcwd() + "/Doc50 GT/"

    for labels_directory in os.listdir(label_path):  # for each assignment folder
        actual_cluster = int(
            labels_directory[1]
        )  # extract cluster label from directory name
        doc_labels = os.listdir(
            label_path + f"\\{labels_directory}"
        )  # for all document ids assigned to this cluster
        for doc in doc_labels:
            actual_labels[doc] = actual_cluster - 1  # save cluster label

    label_seq = []  # save labels in order of documents read
    for doc in doc_list_sequence:
        label_seq.append(actual_labels[doc])

    print(label_seq)

    purity_collection = {}
    for i in range(1500):
        clusters = (
            KMeans(n_init="auto", n_clusters=K, random_state=i, init="k-means++")
            .fit(X)
            .labels_
        )
        purity_collection[i] = Purity_Score(label_seq, clusters)

    max_rand_state = max(purity_collection, key=purity_collection.get)
    print(
        f"Maximum purity of {purity_collection[max_rand_state]} found on random state {max_rand_state}"
    )


def kmeansBestClustering(data, max_clusters, scaling=True, visualization=True):
    n_clusters_list = []
    silhouette_list = []
    print()
    print()
    print(
        "----------------Selecting efficient K By Using Best SilhouetteScore Method-----------------"
    )
    print()
    if scaling:
        # Data Scaling
        scaler = MinMaxScaler()
        data_std = scaler.fit_transform(data)
    else:
        data_std = data

    for n_c in range(2, max_clusters + 1):
        kmeans_model = KMeans(n_clusters=n_c, random_state=58).fit(data_std)
        labels = kmeans_model.labels_
        n_clusters_list.append(n_c)
        silhouette_list.append(silhouette_score(data_std, labels, metric="euclidean"))

    # Best Parameters
    param1 = n_clusters_list[np.argmax(silhouette_list)]
    param2 = max(silhouette_list)
    best_params = param1, param2

    # Data labeling with the best model
    kmeans_best = KMeans(n_clusters=param1, random_state=58).fit(data_std)
    labels_best = kmeans_best.labels_
    labeled_data = np.concatenate((data, labels_best.reshape(-1, 1)), axis=1)

    if visualization:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.plot(
            n_clusters_list,
            silhouette_list,
            linewidth=3,
            label="Silhouette Score Against # of Clusters",
        )
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Silhouette score")
        ax.set_title("Silhouette score according to number of clusters")
        ax.grid(True)
        plt.plot(
            param1,
            param2,
            "tomato",
            marker="*",
            markersize=20,
            label="Best Silhouette Score",
        )

        plt.legend(loc="best", fontsize="large")
        plt.show()
        print("Number of clusters = %i \nSilhouette_score = %.2f." % best_params)


# store Document Features into particular document name file

extractDataFromFiles()

# make vectors for each document
X = buildVsmForDocuments()
# storeDocumentVectors(X)

print(X)

# read Vectors for every document
# X = readDocumentVectors()

# select K using elbow method used pca decomposition on the data
# kMeansClustering(X, 15)

# select K on basis of best silhouette_score used minmax scaling on the data
# kmeansBestClustering(X, 15, True, True)
