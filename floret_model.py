import logging
import subprocess

import floret
import numpy as np
from optuna import Trial
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity


class FloretModel:
    @staticmethod
    def create_from_trial(trial: Trial):
        model = trial.suggest_categorical("mode", ["fasttext", "floret"])

        if model == "fasttext":
            model_params = {
                "mode": "fasttext",
                "model": trial.suggest_categorical("model", ["cbow", "skipgram"]),
                "min_count": trial.suggest_int("min_count", 1, 20),
                "dim": trial.suggest_int("dim", 100, 300),
                "minn": trial.suggest_int("minn", 2, 6),
                "maxn": trial.suggest_int("maxn", 6, 10),
                "bucket": trial.suggest_categorical(
                    "bucket", [200000, 1000000, 2000000]
                ),
                "lr": trial.suggest_float("lr", 0.001, 0.1),
                "epoch": trial.suggest_int("epoch", 1, 10),
                "wordNgrams": trial.suggest_int("wordNgrams", 1, 5),
            }
        elif model == "floret":
            model_params = {
                "mode": "floret",
                "hashCount": trial.suggest_int("hashCount", 1, 4),
                "model": trial.suggest_categorical("model", ["cbow", "skipgram"]),
                "min_count": trial.suggest_int("min_count", 1, 20),
                "dim": trial.suggest_int("dim", 100, 300),
                "minn": trial.suggest_int("minn", 2, 6),
                "maxn": trial.suggest_int("maxn", 6, 10),
                "bucket": trial.suggest_categorical(
                    "bucket", [200000, 1000000, 2000000]
                ),
                "lr": trial.suggest_float("lr", 0.01, 0.3),
                "epoch": trial.suggest_int("epoch", 1, 10),
                "wordNgrams": trial.suggest_int("wordNgrams", 1, 5),
            }

        model = FloretModel(model_params)
        return model

    def __init__(self, model_params):
        self.model_params = model_params
        self.model = None

    def train(self, file_path: str) -> bool:
        try:
            self.model = floret.train_unsupervised(input=file_path, **self.model_params)
            return True
        except Exception as e:
            logging.exception("Failed to train Floret model.")
            return False

    def vector(self, word: str):
        return self.model.get_word_vector(word)

    def prune_and_save_vectors(self, output_stem, frequencies, frequency_limit):
        pruned_vec_path = f"{output_stem}_pruned.vec"
        with open(pruned_vec_path, 'w', encoding='utf-8') as pruned_vec_file:
            words = self.model.get_words()
            pruned_words = [word for word in words if frequencies.get(word, 0) > frequency_limit]

            pruned_vec_file.write(f"{len(pruned_words)} {self.model.get_dimension()}\n")
            for word in pruned_words:
                vector = self.model.get_word_vector(word)
                vector_str = " ".join(map(str, vector))
                pruned_vec_file.write(f"{word} {vector_str}\n")

        return pruned_vec_path

    def convert_vec_to_magnitude(self, vec_path, magnitude_path):
        try:
            subprocess.run(
                ["python", "-m", "pymagnitude.converter", "-i", vec_path, "-o", magnitude_path, "-s", "-a"],
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            logging.exception("Magnitude conversion failed.")
            return False

    def convert_vec_to_tsv(self, vec_path, vectors_tsv_path, metadata_tsv_path):
        with open(vec_path, 'r', encoding='utf-8') as vec_file:
            header = next(vec_file)
            _, dimensions = header.split()
            dimensions = int(dimensions)
            words, vectors = zip(*(line.strip().split(' ', 1) for line in vec_file if line.strip()))

        vectors = [vector for vector in vectors if len(vector.split()) == dimensions]
        words = words[:len(vectors)]

        vectors_tsv = '\n'.join(['\t'.join(vector.split()) for vector in vectors])
        metadata_tsv = '\n'.join(words)

        with open(vectors_tsv_path, 'w', encoding='utf-8') as f_vectors:
            f_vectors.write(vectors_tsv)

        with open(metadata_tsv_path, 'w', encoding='utf-8') as f_metadata:
            f_metadata.write(metadata_tsv)

        logging.info(f"Vectors TSV file saved to {vectors_tsv_path}")
        logging.info(f"Metadata TSV file saved to {metadata_tsv_path}")

    def save(self, filename: str) -> bool:
        try:
            self.model.save_model(filename)
            if "floret" in self.model_params.get("mode", ""):
                floret_vectors_filename = filename.replace(".bin", ".floret")
                self.model.save_floret_vectors(floret_vectors_filename)
            return True
        except Exception as e:
            logging.exception("Failed to save model.")
            return False

    def load(self, filename: str) -> bool:
        try:
            self.model = floret.load_model(filename)
            return True
        except Exception as e:
            logging.exception("Failed to load model.")
            return False

    def evaluate(self, evaluation_data: dict) -> dict:
        return {
            "analogy": self.evaluate_analogies(evaluation_data["analogies"]),
            "clusters": self.evaluate_clusters(evaluation_data["clusters"]),
            "similarity": self.evaluate_similarity(evaluation_data["similarities"]),
        }

    def evaluate_analogies(self, analogies, topn=1):
        correct = 0
        total = 0

        for a, b, c, expected in analogies:
            try:
                predictions = self.model.get_analogies(a, b, c, k=topn)
                predicted_words = [predicted[0] for predicted in predictions]
                if expected in predicted_words:
                    correct += 1
                total += 1
            except Exception as e:
                logging.exception("Failed to evaluate analogy.")
                continue

        return correct / total if total > 0 else 0

    def evaluate_clusters(self, cluster_sets: list[list[list[str]]]):
        ari_scores = []
        nmi_scores = []

        for cluster_set in cluster_sets:
            token_to_cluster_index = {}
            for cluster_index, cluster in enumerate(cluster_set):
                for token in cluster:
                    token_to_cluster_index[token] = cluster_index

            unique_tokens = list(token_to_cluster_index.keys())
            vectors = [self.model.get_word_vector(token) for token in unique_tokens]

            kmeans = KMeans(
                n_clusters=len(cluster_set), random_state=42, n_init=10
            ).fit(vectors)

            predicted_labels = kmeans.labels_
            true_labels = [token_to_cluster_index[token] for token in unique_tokens]
            ari = adjusted_rand_score(true_labels, predicted_labels)
            nmi = normalized_mutual_info_score(true_labels, predicted_labels)
            ari_scores.append(ari)
            nmi_scores.append(nmi)

        average_ari = np.mean(ari_scores) if ari_scores else 0
        average_nmi = np.mean(nmi_scores) if nmi_scores else 0

        return average_ari, average_nmi

    def evaluate_similarity(self, word_pairs):
        total_similarity = 0
        num_pairs = len(word_pairs)

        for term1, term2 in word_pairs:
            vector1 = self.model.get_word_vector(term1)
            vector2 = self.model.get_word_vector(term2)
            total_similarity += cosine_similarity([vector1], [vector2])[0][0]

        average_similarity = total_similarity / num_pairs if num_pairs > 0 else 0
        return average_similarity

