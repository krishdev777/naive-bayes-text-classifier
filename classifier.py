import math
import time
from collections import defaultdict

training_data = [
    ("Win cash prizes and free rewards today only", "spam"),
    ("Exclusive lottery winner claim your free cash now", "spam"),
    ("Urgent account alert claim free coupon right away", "spam"),
    ("Congratulations you won a free mobile gift card", "spam"),
    ("Hey are we meeting at the computer science library today", "ham"),
    ("Please review the pull request for algorithmic optimization", "ham"),
    ("Can you share notes for binary tree and heap lecture", "ham"),
    ("Let us discuss the Google Student Researcher application", "ham")
]

class NaiveBayesClassifier:
    def __init__(self):
        self.class_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.vocab = set()
        self.total_docs = 0

    def tokenize(self, text):
        return text.lower().split()

    def train(self, data):
        self.total_docs = len(data)
        for text, label in data:
            self.class_counts[label] += 1
            words = self.tokenize(text)
            for word in words:
                self.word_counts[label][word] += 1
                self.vocab.add(word)

    def predict(self, text):
        words = self.tokenize(text)
        best_label = None
        max_log_prob = -float('inf')

        for label, count in self.class_counts.items():
            log_prob = math.log(count / self.total_docs)
            total_words_in_class = sum(self.word_counts[label].values())
            vocab_size = len(self.vocab)

            for word in words:
                word_freq = self.word_counts[label][word]
                prob_word = (word_freq + 1) / (total_words_in_class + vocab_size)
                log_prob += math.log(prob_word)

            if log_prob > max_log_prob:
                max_log_prob = log_prob
                best_label = label

        return best_label

if __name__ == "__main__":
    model = NaiveBayesClassifier()

    t0 = time.perf_counter()
    model.train(training_data)
    train_latency = (time.perf_counter() - t0) * 1e6

    test_samples = [
        "claim your free cash prize today",
        "let us study binary tree optimization in the library",
        "free rewards waiting for your account"
    ]

    print("Predictions:")
    t1 = time.perf_counter()
    for text in test_samples:
        pred = model.predict(text)
        print(f"[{pred.upper()}] \"{text}\"")
    eval_latency = ((time.perf_counter() - t1) / len(test_samples)) * 1e6

    print("\nMetrics:")
    print(f"- Unique Vocabulary: {len(model.vocab)} terms")
    print(f"- Training Latency: {train_latency:.2f} us")
    print(f"- Mean Inference Latency: {eval_latency:.2f} us / sample")