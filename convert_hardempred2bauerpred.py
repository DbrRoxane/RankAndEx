import argparse
import jsonlines
import json
import tokenization
import nltk
import Levenshtein

from utils import convert_docs_in_dic

def get_weak_label(squad_file):
    pred = {}

def get_predictions(prediction_file, nb_pred):
    pred = {}
    with open(prediction_file) as pred_reader:
        for line in pred_reader:
            if "_q" in line:
                qid = line[line.index("\"")+1:line.index(":")-1]
                pred[qid] = []
                for i in range(nb_pred):
                    pred[qid].append(pred_reader.readline().strip())
    return pred

def convert_pred2bauer(input_file, output_file, for_training, predictions=None, index_num_answers=None):
    tokenizer = tokenization.BasicTokenizer()
    sum_empty = 0
    with jsonlines.open(output_file, mode="w") as writer:
        with jsonlines.open(input_file, mode="r") as reader:
            for example in reader:
                qid = example['id']
                #context = [subelmt for elmt in example["context"] for subelmt in elmt]
                question = tokenizer.tokenize(example['question'])
                answer1, answer2 = example['final_answers'][:2]
                answer1, answer2 = tokenizer.tokenize(answer1), tokenizer.tokenize(answer2)
                if for_training:
                    context = example['final_answers'][2:]
                    if len(context)==0:
                        sum_empty+=1
                        context = example['context'][0] 
                else:
                    context = predictions[qid][index_num_answers]
                writer.write({"commonsense":[], "summary": context,
                              "ques": question,
                              "answer1": answer1, "answer2":answer2,
                              "doc_num": qid})
    print(sum_empty)


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

    parser.add_argument(
        "--for_training", default=False,
        type=bool, help="If True, take just in account the weakly labelled spans as prediction, otherwise, predictions")

    args = parser.parse_args()
    #squad_file = "data/squad_format/nrnae_bertbm25_20sorted_noanswer_parasplit.json"
    squad_file = "data/squad_format/rbc_sum_train_point_split3.json"#data/squad_format/min_train.jsonl"
    output_file = "data/models/P1_MIN/sum_train_bauerformat.json"
    #output_file = "data/models/P3_rbc_split2_minP1/NRNAE_sorted_test_predictions_bauerformat.json"
    if not args.for_training:
        pred_file = "data/models/P3_rbc_split2_minP1/NRNAE_sorted_test_predictions.json"
        nb_para = 7
        interesting_index = 5
        pred = get_predictions(pred_file, nb_para)
        convert_pred2bauer(squad_file, output_file, args.for_training, pred, interesting_index)
    else:
       convert_pred2bauer(squad_file, output_file, args.for_training) 
    print("FINITO", output_file)
    #dataset = convert_docs_in_dic(args.chunked_stories)
    #convert(dataset, args.input_prediction, args.output_prediction,
    #       args.bauer_file, args.index_paragraphs,
    #       args.levenshtein_maxdistance)
