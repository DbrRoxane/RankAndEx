import rouge as rouge_score
from nltk.translate.bleu_score import sentence_bleu

class Metric:
    def __init__(self):
        pass
    def get_score(self):
        pass

class Rouge(Metric):
    def __init__(self):
        pass

    def compute_score(self, ngrams, answer1, answer2):
        rouge = rouge_score.Rouge()

        n_grams = [" ".join(ngram) for ngram in ngrams]

        scores_a1 = rouge.get_scores(n_grams, [answer1]*len(n_grams))
        scores_a2 = rouge.get_scores(n_grams, [answer2]*len(n_grams))

        rougeL_scores = []
        for a1, a2 in zip(scores_a1, scores_a2):
            rougeL_scores.append(max(a1['rouge-l']['f'], a2['rouge-l']['f']))
        return rougeL_scores


def Bleu(Metric):
    def __init__(self):
        pass

def compute_score(self, ngrams, answer1, answer2):
        bleu_scores = []
        for ngram in ngrams:
            bleu_scores.append(sentence_bleu([answer1, answer2], ngram))
        return bleu_scores


