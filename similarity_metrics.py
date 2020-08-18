import nltk

import rouge as rouge_score
from nltk.translate.bleu_score import sentence_bleu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Metric(object):
    def __init__(self):
        pass
    def compute_score(self):
        pass

class Rouge(Metric):
    def __init__(self):
        Metric.__init__(Rouge)

    def compute_score(self, ngrams, answer1, answer2):
        rouge = rouge_score.Rouge()

        n_grams = [" ".join(ngram) for ngram in ngrams]

        scores_a1 = rouge.get_scores(n_grams, [answer1]*len(n_grams))
        scores_a2 = rouge.get_scores(n_grams, [answer2]*len(n_grams))

        scores = []
        for a1, a2 in zip(scores_a1, scores_a2):
            scores.append(max(a1['rouge-l']['f'], a2['rouge-l']['f']))
        return scores


class Bleu(Metric):
    def __init__(self):
        Metric.__init__(Bleu)

    def compute_score(self, ngrams, answer1, answer2):
        a1_tokenized = nltk.word_tokenize(answer1)
        a2_tokenized = nltk.word_tokenize(answer2)

        scores = []
        for ngram in ngrams:
            scores.append(
                sentence_bleu(
                    [a1_tokenized, a2_tokenized],
                    list(ngram),
                    weights=(0.8, 0.2)
                ))
        return scores

class Cosine(Metric):
    def __init__(self, max_n):
        Metric.__init__(Cosine)
        self.vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)

    def compute_score(self, ngrams, answer1, answer2):
        ngrams_join = [" ".join(ngram) for ngram in ngrams]
        vectorized_ngrams = self.vectorizer.fit_transform(ngrams_join)
        query_tfidf = self.vectorizer.transform([answer1, answer2])
        cosine_similarities = cosine_similarity(query_tfidf, vectorized_ngrams)
 
        scores = []
        for i in range(len(ngrams_join)):
            scores.append(max(cosine_similarities[:, i]))
        return scores
