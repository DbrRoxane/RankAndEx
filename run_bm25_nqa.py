import argparse
import csv
import time
import re
import nltk
import numpy as np
from rank_bm25 import BM25Okapi

import sys

from utils import convert_docs_in_dic



def compute_bm25(tokenized_query, story_id, paragraphs, n):
    tokenized_paragraphs = [paragraph.split(" ") \
                            for paragraph in paragraphs]
    bm25 = BM25Okapi(tokenized_paragraphs)
    best_p = bm25.get_top_n(tokenized_query, paragraphs, n=n)
    best_i = [p.split(" ")[0] for p in best_p]
    return best_i

def gather_bm25(dataset, n, attach_answer):
    predictions = dict()
    for story_id, story_details in dataset.items():
        paragraphs = ["{} {}".format(k,v) for k,v in dataset[story_id]['paragraphs'].items()]
        for query_id, query_details in story_details['queries'].items():
            query = "{} {} {}".format(query_details['query'],
                                        query_details['answer1'],
                                        query_details['answer2']) if \
                    attach_answer else query_details['query']
            tokenized_query = query.split(" ")
            predictions[query_id] = compute_bm25(
                tokenized_query, story_id,
                paragraphs, n)
    return predictions

def write_bm25_pred(dataset, n, attach_answer, output_file):
    with open(output_file, "w") as f:
        predictions = gather_bm25(dataset, n, attach_answer)
        for query_id, paragraphs in predictions.items():
            for i, p in enumerate(paragraphs):
                f.write("{}\t{}\t{}\n".format(query_id, p, i+1))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunked_stories", default="./data/processed/narrativeqa_all.eval", \
        type=str, help="Path for the chunked stories")

    parser.add_argument(
        "--output_file", default="./data/ranking/bm25.tsv", \
        type=str, help="Path to store the bm25 ranking")

    parser.add_argument(
        "--max_rank", default=3,
        type=int, help="Number of best rank to store")

    parser.add_argument(
        "--use_answer", default=True,
        type=bool, help="Create oracle ranking or not")

    args = parser.parse_args()

    dataset = convert_docs_in_dic(args.chunked_stories)
    print("Dataset loaded")
    write_bm25_pred(dataset, n=args.max_rank,
                    attach_answer=args.use_answer,
                    output_file=args.output_file)

if __name__=="__main__":
    main()
