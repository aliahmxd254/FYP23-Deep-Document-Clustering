# K-Means-Clustering-Using-Lexical-Chains
This repository features K-means document clustering code. It preprocesses text, builds relations and lexical chains using WordNet, extracts features, constructs a VSM, performs clustering, and allows for selecting optimal K. It provides visualizations and evaluation metrics for K selection.


This repository contains code for a project focused on document clustering using the K-means algorithm.

The main functionalities of the code include:

Preprocessing: The code preprocesses text data by removing links, numbers, special characters, and optionally stopwords. It applies lemmatization to the remaining words.

Building Relations and Lexical Chains: The code builds relations between nouns in the text using WordNet, a lexical database. It creates lexical chains based on these relations, using a threshold for similarity comparison.

Document Feature Extraction: The code extracts features from documents by using lexical chains as the feature selection method. It stores the document features in separate files using pickle.

Vector Space Model (VSM) Construction: The code constructs a VSM representation for the documents by combining the document features extracted earlier. It identifies the total features present across all documents and creates a matrix representing the features and their occurrences in each document.

Document Vector Storage and Retrieval: The code stores and retrieves the document vectors obtained from the VSM representation, using pickle.

K-means Clustering: The code performs K-means clustering on the document vectors. It provides functionality for selecting the optimal value of K using the elbow method and visualizing the results. It also calculates the within-cluster sum of squares (WCSS) and the Silhouette Coefficient as evaluation metrics.

Best K-means Clustering: The code provides an alternative method for selecting the optimal value of K based on the best Silhouette score. It offers the option to scale the data using MinMaxScaler before clustering and visualizes the results.

**Note: The code includes specific file paths that need to be adjusted based on the user's file structure and requirements.**
Overall, this project aims to cluster documents using K-means and provides options for selecting the optimal number of clusters based on different evaluation metrics.
