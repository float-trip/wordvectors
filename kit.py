# pip install pymagnitude matplotlib networkx numpy scikit-learn

import re
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
from pymagnitude import *
from sklearn.manifold import TSNE
import re
import json
import numpy as np
from pymagnitude import *
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import networkx as nx

vectors = Magnitude('vectors.magnitude')

class Kit:
    def __init__(self, vectors):
        self.vectors = vectors
        with open("frequencies.json", "r") as f:
            self.frequencies = json.loads(f.read())

    def find_special_tokens(self):
        emoticon_regex = re.compile(r"^:[!#@\w\-]{1,72}:$")
        mention_regex = re.compile(r"^@.*")

        emoticons = []
        mentions = []

        for key, _ in self.vectors:
            if self.frequencies.get(key, 0) < 50:
                continue
            if emoticon_regex.fullmatch(key):
                emoticons.append(key)
            elif mention_regex.fullmatch(key):
                mentions.append(key)

        return emoticons, mentions

    def cosine_similarity(self, word1, word2):
        return self.vectors.similarity(word1, word2)

    def most_similar(self, word, topn=5):
        return self.vectors.most_similar(positive=[word], topn=topn)

    def analogy(self, positive, negative, topn=5):
        return self.vectors.most_similar(positive=positive, negative=negative, topn=topn)

    def tsne_plot_words(self, words, perplexity=40, n_components=2, n_iter=3000):
        word_vectors = self.vectors.query(words)
        tsne_model = TSNE(perplexity=perplexity, n_components=n_components, init='pca', n_iter=n_iter)
        new_values = tsne_model.fit_transform(word_vectors)

        plt.figure(figsize=(16, 9))
        for i, word in enumerate(words):
            plt.scatter(new_values[i][0], new_values[i][1])
            plt.annotate(word, xy=(new_values[i][0], new_values[i][1]), xytext=(5, 2), textcoords='offset points')
        plt.show()

    def create_similarity_graph(self, words, threshold=0.5):
        G = nx.Graph()
        for i, word1 in enumerate(words):
            for word2 in words[i + 1:]:
                if self.vectors.similarity(word1, word2) > threshold:
                    G.add_edge(word1, word2)

        pos = nx.spring_layout(G)
        plt.figure(figsize=(16, 9))
        nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='grey', node_size=2500, alpha=0.7)
        plt.title('Similarity Graph')
        plt.show()

    def cluster(self, words, n_clusters=5):
        word_vectors = self.vectors.query(words)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(word_vectors)
        clusters = defaultdict(list)
        for word, label in zip(words, kmeans.labels_):
            clusters[label].append(word)
        return dict(clusters)

kit = Kit(vectors)

emoticons, mentions = kit.find_special_tokens()
print("Emoticons found:", len(emoticons))
print("Mentions found:", len(mentions))

print("Most similar to 'dog':", kit.most_similar('dog'))

print("Analogy - 'man' is to 'king' as 'woman' is to:", kit.analogy(['woman', 'king'], ['man'], topn=1))

user = "@float-trip"
emoticon = max(((emoticon, kit.vectors.similarity(user, emoticon)) for emoticon in emoticons), key=lambda x: x[1])
print(f"Most similar emoticon to @float-trip: {emoticon}")

words_for_clustering = ['dog', 'cat', 'fish', 'car', 'bicycle']
clusters = kit.cluster(words_for_clustering, n_clusters=2)
print("Clusters:", clusters)

import code; code.interact(local=dict(globals(), **locals()))

