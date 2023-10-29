import numpy as np
import nltk
import os
import re
from collections import defaultdict
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from nltk.corpus import wordnet as wn, stopwords
from nltk.stem import WordNetLemmatizer
from tf_idf import wrapperFunction
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import adjusted_rand_score  # Import the Rand Index metric


# from cuml import PCA

# Sample document data
doc_content = []  # all the content in the document
doc_name = []  # name of the document
files_path = []  # path to the documents
lexical_chain = []  # list of lexical chains from each document
total_features = []  # total number of features. 1652
final_training_Features = []
corpus = []
doc_list_sequence = []


def ReadDocuments(dir_name):
    for Path in os.listdir(dir_name):
        file_p = os.path.join(dir_name, Path)
        with open(file_p, "r") as file:
            FileContents = file.read()
            doc_content.append(FileContents.lower())
            doc_name.append(Path)
            files_path.append(file_p)


def preprocess_text(text: str, remove_stopwords: bool) -> str:
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r"http\S+", "", text)
    text = re.sub("[^A-Za-z]+", " ", text)
    if remove_stopwords:
        tokens = nltk.word_tokenize(text)
        updated_tokens = []
        for i in range(len(tokens)):
            if tokens[i].lower() in stopwords.words("english"):
                continue
            else:
                updated_tokens.append(lemmatizer.lemmatize(tokens[i].lower()))

    return updated_tokens


def buildRelation(nouns):
    relation_list = defaultdict(list)

    for k in range(len(nouns)):
        relation = []
        for syn in wn.synsets(nouns[k], pos=wn.NOUN):
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
                        syns1 = wn.synsets(key, pos=wn.NOUN)
                        syns2 = wn.synsets(noun, pos=wn.NOUN)
                        if syns1[0].wup_similarity(syns2[0]) >= threshold:
                            lexical[j][noun] = 1
                            flag = 1
                    elif noun in relation_list[key][0] and flag == 0:
                        syns1 = wn.synsets(key, pos=wn.NOUN)
                        syns2 = wn.synsets(noun, pos=wn.NOUN)
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


def PreprocessDocuments():
    for i in files_path:
        f = open(i, "r")
        dataset = preprocess_text(f.read(), remove_stopwords=True)
        # use lexical chains as the feature selection method
        nouns = []
        l = nltk.pos_tag(dataset)
        for word, n in l:
            if n == "NN" or n == "NNS" or n == "NNP" or n == "NNPS":
                nouns.append(word)

        relation = buildRelation(nouns)
        lexical = buildLexicalChain(nouns, relation)
        chain = eliminateWords(lexical)
        lexical_chain.append(chain)

    global total_features
    for features in lexical_chain:
        for docfeature in features:
            total_features.extend(docfeature.keys())

    total_features = list(set(total_features))

    for feature in lexical_chain:
        temp = []
        # print(feature)
        for j in total_features:
            check = False
            for f in feature:
                if j in f:
                    temp.append(f[j])
                    check = True
                    break
            if not check:
                temp.append(0)

        final_training_Features.append(temp)


def build_lexical_chains(doc):
    tokens = nltk.word_tokenize(doc)
    pos_tags = nltk.pos_tag(tokens)
    chains = {}

    for token, pos in pos_tags:
        synsets = wn.synsets(token, pos=wn.NOUN)
        for synset in synsets:
            if synset not in chains:
                chains[synset] = [token]
            else:
                chains[synset].append(token)

    return chains


def Purity_Score(label_seq, pred_labels):
    # Calculate the confusion matrix to compare true labels and cluster assignments
    confusion = confusion_matrix(label_seq, pred_labels)
    # Calculate the purity
    purity = np.sum(np.max(confusion, axis=0)) / np.sum(confusion)
    return purity


def calculate_consensus_matrix(labels1, labels2):
    n = len(labels1)
    consensus_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            #Calculate the Jaccard similarity between the two label sets
            intersection = np.intersect1d(labels1[i], labels2[j])
            union = np.union1d(labels1[i], labels2[j])
            agreement = len(intersection) / len(union)
        

            consensus_matrix[i, j] = agreement
            consensus_matrix[j, i] = agreement

    return consensus_matrix


doc_50_path = os.getcwd() + "\Doc50"
ReadDocuments(doc_50_path)
PreprocessDocuments()

normalizer = Normalizer()
normalize_features = normalizer.fit_transform(final_training_Features)

print("\n----------------------------LEXICAL CHAINS--------------------------------")



SumSqDis = []
pca = PCA(n_components=30, random_state=42)
pca_vecs = pca.fit_transform(normalize_features)

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
label_path = os.getcwd() + "\Doc50 GT"
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

purity_collection = {}
for i in range(1500):
    clusters = KMeans(n_init="auto", n_clusters=5, random_state=i, init="k-means++").fit(pca_vecs).labels_
    purity_collection[i] = Purity_Score(label_seq, clusters)

max_rand_state = max(purity_collection, key=purity_collection.get)
print(
    f"Maximum purity of {purity_collection[max_rand_state]} found on random state {max_rand_state}"
)

lexicalChainsLabels = KMeans(n_init="auto", n_clusters=5, random_state=max_rand_state, init="k-means++").fit(pca_vecs).labels_

print(f"""K-means lables using lexical chains: {lexicalChainsLabels}
      \nPurity {Purity_Score(label_seq, lexicalChainsLabels)}
      \nSilhoutte Score: {metrics.silhouette_score(final_training_Features, lexicalChainsLabels, metric='euclidean')}""")




print("\n----------------------------TF-IDF Clustering--------------------------------")
tfidfLabels, tfidfMatrix = wrapperFunction()

# CONSENSUS CLUSTERING
print("\n----------------------------Consensus Clustering--------------------------------")


consensus_matrix = calculate_consensus_matrix(tfidfLabels, lexicalChainsLabels)

n_clusters = 5  # You can adjust this as needed
purity_collection = {}
for i in range(1500):
    clusters = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=i).fit(1 - consensus_matrix).labels_
    purity_collection[i] = Purity_Score(label_seq, clusters)

max_rand_state = max(purity_collection, key=purity_collection.get)
print(f"Maximum purity of {purity_collection[max_rand_state]} found on random state {max_rand_state}")
spectral_labels = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=max_rand_state).fit(1 - consensus_matrix).labels_
print("Purity Score: ", Purity_Score(label_seq, spectral_labels))
print("Sillhouette Coefficient: ",metrics.silhouette_score(pca_vecs, spectral_labels, metric="euclidean"),)


print("\n----------------------------Clustering With Topics--------------------------------")

num_topics = 5  # Adjust as needed
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
topic_proportions = lda.fit_transform(tfidfMatrix)

combined_features = np.hstack((spectral_labels.reshape(-1, 1), topic_proportions))

normalize_combined_features = Normalizer().fit_transform(combined_features)

topic_purity_collection = {}
for i in range(1500):
    topic_clusters = (
        KMeans(n_init="auto", n_clusters=5, random_state=i, init="k-means++")
        .fit(normalize_combined_features)
        .labels_
    )
    topic_purity_collection[i] = Purity_Score(label_seq, topic_clusters)

topic_max_rand_state = max(topic_purity_collection, key=topic_purity_collection.get)
print(f"Maximum purity of {topic_purity_collection[topic_max_rand_state]} found on random state {topic_max_rand_state}")
max_labels = (
    KMeans(
        n_init="auto", n_clusters=5, random_state=topic_max_rand_state, init="k-means++"
    )
    .fit(normalize_combined_features)
    .labels_
)
print("Purity: ", Purity_Score(label_seq, max_labels))
print("Sillhouette Coefficient: ",metrics.silhouette_score(normalize_combined_features, max_labels, metric="euclidean"))


