import csv
import jsonlines
import nltk
import itertools
import linecache
import rouge as rouge_score
import numpy as np
import time
import sys
import tokenization

csv.field_size_limit(sys.maxsize)

DOCUMENTS_FILE = "./data/narrativeqa/documents.csv"
SUMMARIES_FILE = "./data/narrativeqa/third_party/wikipedia/summaries.csv"

tokenizer = tokenization.BasicTokenizer()

def process_str(text):
    return " ".join(tokenizer.tokenize(text))


def retrieve_doc_info(story_id):
    with open(DOCUMENTS_FILE, "r") as f:
        csv_reader = csv.DictReader(f, delimiter=",")
        for row in csv_reader:
            if row['document_id'] == story_id:
                return row['set'], row['kind']
        print("did not find story", story_id)

def retrieve_summary(story_id):
    with open(SUMMARIES_FILE, "r") as f:
        csv_reader = csv.DictReader(f, delimiter=",")
        for row in csv_reader:
            if row['document_id'] == story_id:
                return process_str(row['summary'])

def convert_docs_in_dic(chunked_stories):
    with open(chunked_stories, "r", encoding="ascii", errors="ignore") as f:
        dataset = {}
        csv_reader = csv.reader(f, delimiter="\t")
        for row in csv_reader:
            query_id = row[0]
            story_id = query_id.split("_")[0]
            if story_id not in dataset.keys():
                summary = retrieve_summary(story_id)
                train_dev_test, book_movie = retrieve_doc_info(story_id)
                dataset[story_id] = {'paragraphs': {}, 'queries':{},
                                     'summary' : summary,
                                     'set':train_dev_test, 'kind':book_movie}
            paragraph_id = row[1]
            if paragraph_id not in dataset[story_id]['paragraphs'].keys():
                pargraph = process_str(row[3])
                dataset[story_id]['paragraphs'][paragraph_id] = pargraph
            if query_id not in dataset[story_id]['queries'].keys():
                query = process_str(row[2])
                answer1, answer2 = process_str(row[4]), process_str(row[5])
                dataset[story_id]['queries'][query_id] = {
                    'query': query,
                    'answer1' : answer1,
                    'answer2' : answer2}
    return dataset

def convert_rank_in_dic(ranking_files, max_rank=3):
    ranking_dic = dict()
    for ranking_filename in ranking_files:
        with open(ranking_filename, 'r') as ranking_file:
            ranking_reader = csv.reader(ranking_file, delimiter="\t")
            for row in ranking_reader:
                if row[0] not in ranking_dic.keys():
                    ranking_dic[row[0]] = {}
                if ranking_filename not in ranking_dic[row[0]].keys():
                    ranking_dic[row[0]][ranking_filename] = []
                if eval(row[2]) < max_rank:
                    ranking_dic[row[0]][ranking_filename].append(row[1])
    return ranking_dic

def merge_ranks(ranking_dic):
    rank_merged = dict()
    for query_id, ranks in ranking_dic.items():
        merged_list = [par for groupped_rank in zip(*list(ranks.values()))
                       for par in groupped_rank]
        merged_list = list(set(merged_list))
        rank_merged[query_id] = merged_list
    return rank_merged

