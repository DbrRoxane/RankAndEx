import argparse
import csv
import time
import re
import nltk
import numpy as np
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from convertor import convert_docs_in_dic


TFIDF_FILE = "./data/output/tfidf_predictions_optimized.tsv"
BOOK_EVAL_FILE = "./data/processed/narrativeqa_all.eval"

def compute_tfidf(query, story_id, paragraphs, vect_paragraphs, vectorizer, n):
    query_tfidf = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_tfidf, vect_paragraphs)\
            .flatten()
    best_index = np.flip(np.argsort(cosine_similarities), axis=0)[:n]
    best_p = [paragraphs[i] for i in best_index]
    best_i = [p.split(" ")[0] for p in best_p]
    return best_i

def gather_tfidf(dataset, vectorizer, n, attach_answer):
    predictions = dict()
    for story_id, story_details in dataset.items():
        paragraphs = ["{} {}".format(k,v)
                      for k, v in dataset[story_id]['paragraphs'].items()]
        vectorized_paragraphs = vectorizer.fit_transform(paragraphs)
        for query_id, query_details in story_details['queries'].items():
            query = "{} {} {}".format(query_details['query'],
                                        query_details['answer1'],
                                        query_details['answer2']) if \
                    attach_answer else query_details['query']
            query = query.lower()
            predictions[query_id] = compute_tfidf(query,
                                                  story_id,
                                                  paragraphs,
                                                  vectorized_paragraphs,
                                                  vectorizer,
                                                  n)
    return predictions

def write_tfidf_pred(dataset, vectorizer, n, attach_answer, output_file):
    with open(output_file, "w") as f:
        predictions = gather_tfidf(dataset, vectorizer, n, attach_answer)
        for query_id,paragraphs in predictions.items():
            for i,p in enumerate(paragraphs):
                f.write("{}\t{}\t{}\n".format(query_id, p, i+1))

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunked_stories", default="./data/processed/narrativeqa_all.eval", \
        type=str, help="Path for the chunked stories")

    parser.add_argument(
        "--output_file", default="./data/ranking/bm25.tsv", \
         type=str, help="Path to store the bm25 ranking")

    args = parser.parse_args()

    dataset = convert_docs_in_dic(args.chunked_stories)
    vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, ngram_range=(1,2))
    write_tfidf_pred(dataset, vectorizer, n=3, attach_answer=True,
                     output_file=args.output_file)

if __name__=="__main__":
    main()
