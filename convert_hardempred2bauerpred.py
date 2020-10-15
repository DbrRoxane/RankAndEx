import argparse
import jsonlines
import json
import tokenization
import nltk
import Levenshtein

from utils import convert_docs_in_dic

def convert(dataset, input_file, output_file, bauer, n=1, levenshtein_threshold=5):
    tokenizer = tokenization.BasicTokenizer()
    with open(input_file, "r") as pred_file:
        pred = json.load(pred_file)
    with open(output_file, "w") as writer:
        with jsonlines.open(bauer, "r") as bauer_file:
            for example in bauer_file:
                if example['doc_num'] in dataset.keys():
                    writen = False
                    for query_key, query_value in dataset[example['doc_num']]['queries'].items():
                        levenshtein = Levenshtein.distance(
                            "".join(tokenizer.tokenize(" ".join(example['ques']))),
                            "".join(tokenizer.tokenize(query_value['query']))
                        )
                        if levenshtein < levenshtein_threshold:
                            query_id = query_key
                            generated_answer = pred.get(query_id, ["NO PREDICTION"]*(n))[n-1]
                            writer.write(generated_answer+"\n")
                            writen = True
                            break
                    if not writen:
                        writer.write("NO PREDICTION\n")
if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunked_stories", default="./data/narrativeqa_all.eval", \
        type=str, help="Path for the chunked stories")

    parser.add_argument(
        "--input_prediction", default="./data/predictions/prediction.json", \
        type=str, help="Path for the predicted answers")

    parser.add_argument(
        "--output_prediction", default="./data/predictions/prediction_converted", \
        type=str, help="Path for the converted predictions")

    parser.add_argument(
        "--bauer_file", default="./data/bauer/narrative_qa_test.jsonl", \
        type=str, help="Path for the data downloaded from Commonsense Multi-Hop repository")

    parser.add_argument(
        "--index_paragraphs", default=1, \
        type=str, help="If several passage size were indicated for Hard-EM run, enter the index of the best nb of parahraphs")

    parser.add_argument(
        "--levenshtein_maxdistance", default=5,
        type=int, help="Maximum levenshtein distance between two queries since not tokenized exactly the same")

    args = parser.parse_args()

    dataset = convert_docs_in_dic(args.chunked_stories)
    convert(dataset, args.input_prediction, args.output_prediction,
           args.bauer_file, args.index_paragraphs,
           args.levenshtein_maxdistance)
