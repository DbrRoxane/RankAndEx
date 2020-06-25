import argparse
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

from utils import *

class Convertor(object):
    """
    Convert best ranked paragraph to narrativeQA over summaries format
    Convertor is an abstract class of specific convertor for given format
    """
    def __init__(self, converted_filename, dataset, ranking_dic=None):
        self.converted_filename = converted_filename
        self.dataset = dataset
        self.ranking_dic = ranking_dic
        self.tokenizer = tokenization.BasicTokenizer()

    def find_and_convert(self, just_book, train_dev_test):
        """
        Retrieve the n best paragraphs in a story based on a question
        """

        converted_file = self.open_file()
        for query_id, paragraphs_ids in self.ranking_dic.items():
            story_id, _ = query_id.split("_")
            select_book = self.dataset[story_id]['kind'] == 'gutenberg' if \
                    just_book else True
            select_set = self.dataset[story_id]['set'] == train_dev_test
            if select_book and select_set:
                context, query, answer1, answer2 = self.extract_query_details(
                    story_id, query_id, paragraphs_ids)
                entry = {'query_id':query_id,
                         'story_id':story_id,
                         'paragraphs_id':paragraphs_ids,
                         'query':query,
                         'context':context,
                         'answer1':answer1,
                         'answer2':answer2}
                self.write_to_converted_file(converted_file, entry)
        self.close_file(converted_file)

    def find_and_convert_from_summaries(self, train_dev_test):
        converted_file = self.open_file()
        for story_id, story_details in self.dataset.items():
            if story_details['set'] == train_dev_test:
                for query_id, query_details in story_details['queries'].items():
                    entry = {'story_id':story_id,
                             'query_id':query_id,
                             'query':query_details['query'],
                             'context':story_details['summary'],
                             'answer1':query_details['answer1'],
                             'answer2':query_details['answer2']
                            }
                    self.write_to_converted_file(converted_file, entry)
        self.close_file(converted_file)

    def open_file(self):
        pass

    def close_file(self, converted_file):
        pass

    def write_to_converted_file(self, converted_file, entry):
        pass

    def extract_query_details(self, story_id, query_id, paragraphs_id):
        context = ""
        for p_id in paragraphs_id:
            p_str = self.dataset[story_id]['paragraphs'].get(p_id, "")
            context += p_str + "\n" if p_str != "" else p_str
            if p_str == "":
                print("Cannot retrieve paragraph {}".format(p_id))
                print("All the p_id are {} \n \n".
                      format(self.dataset[story_id]['paragraphs'].keys()))
        query, answer1, answer2 = self.dataset[story_id]['queries'][query_id].values()
        return context, query, answer1, answer2


class MinConvertor(Convertor):
    def __init__(self, converted_filename, dataset, ranking_dic=None, rouge_threshold=0.5):

        super().__init__(converted_filename, dataset, ranking_dic)
        self.rouge_threshold = rouge_threshold

    def open_file(self):
        return jsonlines.open(self.converted_filename, mode="w")

    def write_to_converted_file(self, converted_file, entry):
        paragraphs = entry['context'].split("\n")
        paragraphs_tokenized = [self.tokenizer.tokenize(paragraph.replace('.', ''))
                                for paragraph in paragraphs]
        answers = [self.find_likely_answer(p_tokenized,
                                           entry['answer1'],
                                           entry['answer2']) \
                   for p_tokenized in paragraphs_tokenized]
        final_answers = [answer['text'] for paragraph in answers
                         for answer in paragraph if answer != []]
        final_answers =  [entry['answer1'], entry['answer2']] + final_answers
        converted_file.write({'id'       : entry['query_id'],
                              'question' : entry['query'],
                              'context'  : paragraphs_tokenized,
                              'answers'  : answers,
                              'final_answers' : final_answers})

    def close_file(self, converted_file):
        converted_file.close()

    def match_first_span(self, paragraph, subtext):
        size_ngram = len(subtext)
        subtext = [sub.replace('`', '\'') for sub in subtext]
        paragraph = [par.replace('`', '\'') for par in paragraph]
        start_index = [i for i, x in enumerate(paragraph) if x in subtext[0]]
        for i in start_index:
            if paragraph[i:i+size_ngram] == subtext:
                return i, i+size_ngram-1
        return None, None

    def find_likely_answer(self, paragraph, answer1, answer2, max_n=20):
        """
        Knowing an answer, find spans in the paragraphs with high rouge score
        max_n is the biggest n-gram analyzed
        """

        previous_max_score, max_score = 0, 0
        masked_paragraph = paragraph.copy()
        subtext = paragraph.copy()
        rouge = rouge_score.Rouge()
        max_n = min(max_n, len(paragraph))
        answers = []
        i = max_n
        while i > 0:
            n_grams = [" ".join(n_gram) for n_gram in nltk.ngrams(subtext, i)]
            scores = [score['rouge-l']['f']
                      for score in rouge.get_scores(n_grams, [answer1]*len(n_grams))]
            scores += [score['rouge-l']['f']
                       for score in rouge.get_scores(n_grams, [answer2]*len(n_grams))]
            max_index_score = np.argmax(np.array(scores))
            max_score = scores[max_index_score]

            #if the previous score was better than the actual, 
            #it means that we have a better span and no need to fo further
            #or if max_score=0, we know there is nothing interesting here
            if previous_max_score > max_score or max_score == 0:
                if previous_max_score < self.rouge_threshold or max_score == 0:
                    break
                index_start, index_end = self.match_first_span(masked_paragraph, subtext)
                if not index_start and not index_end:
                    break
                answers.append({'text':" ".join(subtext), 'word_start':index_start, 'word_end':index_end})

                #once we find a good answer, we remove it from the initial paragraph
                #and rerun the exploration
                previous_max_score = 0
                i = max_n
                for j in range(index_start, index_end+1):
                    masked_paragraph[j] = "MASK"
                subtext = masked_paragraph.copy()
            else:
                subtext = self.tokenizer.tokenize(n_grams[max_index_score % len(n_grams)])
                previous_max_score = max_score
                i -= 1

        if max_score >= self.rouge_threshold:
            index_start, index_end = self.match_first_span(masked_paragraph, subtext)
            if index_start:
                answers.append({'text':" ".join(subtext), 'word_start':index_start, 'word_end':index_end})

        return answers


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunked_stories", default="./data/processed/narrativeqa_all.eval", \
        type=str, help="Path for the chunked stories")

    parser.add_argument(
        "--output_file", default="./data/output/nqa_squadformat.json", \
        type=str, help="Path for the generated dataset")

    parser.add_argument(
        "--ranking_file", \
        type=list, help="Paths of the ranking predictions files")

    parser.add_argument(
        "--rouge_threshold", default=0.6, \
        type=float, help="Rouge score threshold")

    parser.add_argument(
        "--summary", default=False, \
        type=bool, help="True if NarrativeQA on summaries, False on stories")

    parser.add_argument(
        "--sets", default="test", \
        type=str, help="which set to use between {train, dev, test}")

    parser.add_argument(
        "--max_rank", default=20, \
        type=str, help="how many paragraph to use")

    args = parser.parse_args()

    ranking = merge_ranks(
        convert_rank_in_dic(args.ranking_file, args.max_rank))
    print("Created rankind dic")

    dataset = convert_docs_in_dic(args.chunked_stories)
    print("Created dataset")

    convertor = MinConvertor(args.output_file,
                             dataset,
                             ranking,
                             args.rouge_threshold)
    if not args.summary:
        convertor.find_and_convert(just_book=False, train_dev_test=args.sets)
    else:
        convertor.find_and_convert_from_summaries(train_dev_test=args.sets)

 if __name__=="__main__":
     main()