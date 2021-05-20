import json
import sys
sys.path.append("../")

from similarity_metrics import Rouge, Bleu, Cosine

import tokenization
from convert_squad_format import MinConvertor

def coqa2min(inp, out, tokenizer):
    with open(inp) as f:
        list_input_data = json.load(f)["data"]
    output_data = []
    with open(out, "w") as f:
        for story in list_input_data:
            context = [tokenizer.tokenize(story["story"])]
            for question, answer in zip(story["questions"], story["answers"]):
                q = question["input_text"]
                index = str(story["id"]) + "_" + str(question["turn_id"])
                if not answer["span_text"].strip() : 
                    continue
                span_start = len(tokenizer.tokenize(story["story"][:answer["span_start"]].strip()))
                span_end = span_start + len(tokenizer.tokenize(answer["span_text"].strip()))
                a = [[{"text":answer["span_text"], "word_start":span_start, "word_end":span_end}]]
                final = [answer["input_text"]]
                json.dump({"id":index,
                           "question":q,
                           "context":context,
                           "answers": a,
                           "final_answers":final},
                         f)
                f.write("\n")

def squad2min(inp, out, tokenizer):
    with open(inp) as f:
        list_input_data = json.load(f)["data"]
    output_data = []
    with open(out, "w") as f:
        for story in list_input_data:
            context = [tokenizer.tokenize(paragraph["context"]) for paragraph in story["paragraphs"]]
            for i, paragraph in enumerate(story["paragraphs"]):
                a = [[] for j in story["paragraphs"]]
                final = []
                for qa in paragraph["qas"]:
                    index = qa["id"]
                    q = qa["question"]
                    answers = qa["answers"]
                    if len(answers)<1 : 
                        continue
                    for answer in answers:
                        char_start = answer.pop("answer_start")
                        answer["word_start"] = len(tokenizer.tokenize(paragraph["context"][:char_start].strip()))
                        answer["word_end"] = answer["word_start"] + len(tokenizer.tokenize(answer["text"].strip()))
                        a[i].append(answer)
                        final.append(answer["text"])
                    json.dump({"id":index,
                               "question":q,
                               "context":context,
                               "answers":a,
                               "final_answers":final},
                             f)
                    f.write("\n")

def race2min(data_path, out, tokenizer):
    import glob
    input_files = glob.glob(data_path+"/high/*") + glob.glob(data_path+"/middle/*")
    dic_opt = {"A":0, "B":1, "C":2, "D":3}

    convertor = MinConvertor(converted_filename=None, 
                             dataset=None,
                             ranking_dic=None,
                             metrics=(Rouge, Bleu),
                             metrics_thresholds=(0.6,0.5))
    with open(out, "w") as write_file:
        for filename in input_files:
            with open(filename) as f:
                story = json.load(f)
            context = [tokenizer.tokenize(story["article"])]
            index = story["id"]
            for question, options, answer in zip(story["questions"], story["options"], story["answers"]):
                q = question
                final = [options[dic_opt[answer]]]
                a = [convertor.find_likely_answer(context[0], final[0], max_n=10)]
                # answer to add with span retrieve ? 
                json.dump({"id":index,
                           "question":q,
                           "context":context,
                           "answers":a,
                           "final_answers":final},
                         write_file)
                write_file.write("\n")


def wikihop2min(inp, out, tokenizer):
    convertor = MinConvertor(converted_filename=None, 
                             dataset=None,
                             ranking_dic=None,
                             metrics=(Rouge, Bleu),
                             metrics_thresholds=(0.6,0.5))
    with open(inp) as f:
        list_input_data = json.load(f)
    output_data = []
    with open(out, "w") as f:
        for story in list_input_data:
            context = [tokenizer.tokenize(paragraph) for paragraph in story["supports"]]
            index = story["id"]
            q= story["query"]
            final = [story["answer"]]
            a = [convertor.find_likely_answer(c, final[0], max_n=10) for c in context]
            json.dump({"id":index,
                       "question":q,
                       "context":context,
                       "answers":a,
                       "final_answers":final},
                     f)
            f.write("\n")

def msmarco2min(inp, out, tokenizer):
    convertor = MinConvertor(converted_filename=None, 
                             dataset=None,
                             ranking_dic=None,
                             metrics=(Rouge, Bleu),
                             metrics_thresholds=(0.6,0.5))
    with open(inp) as f:
        list_input_data = json.load(f)
    output_data = []
    with open(out, "w") as f:
        for idx in list_input_data["answers"].keys():
            context = [tokenizer.tokenize(paragraph["passage_text"]) for paragraph in list_input_data["passages"][idx]]
            index = list_input_data["query_id"][idx]
            q= list_input_data["query"][idx]
            final = list_input_data["wellFormedAnswers"][idx] \
                    if list_input_data["wellFormedAnswers"][idx] \
                    else list_input_data["answers"][idx]
            answer1 = list_input_data["answers"][idx][0]
            if len(list_input_data["answers"][idx])<1:
                continue
            answer2 = None if len(list_input_data["answers"][idx])<2 else list_input_data["answers"][idx][1]
            a = [convertor.find_likely_answer(c, answer1, answer2, max_n=10) for c in context]
            json.dump({"id":index,
                       "question":q,
                       "context":context,
                       "answers":a,
                       "final_answers":final},
                     f)
            f.write("\n")


if __name__=="__main__":
    tokenizer = tokenization.BasicTokenizer()

    print("COQA starts")
    train_coqa_loc = "../../data/coqa/coqa-train-v1.0.json"
    dev_coqa_loc = "../../data/coqa/coqa-train-v1.0.json"
    out_train = "../../data/hardem_format/coqa_train.json"
    out_dev = "../../data/hardem_format/coqa_dev.json"
    #coqa2min(train_coqa_loc, out_train, tokenizer)
    #coqa2min(dev_coqa_loc, out_dev, tokenizer)

    print("SQUAD starts")
    train_squad_loc = "../../data/Squad/train-v2.0.json"
    dev_squad_loc = "../../data/Squad/dev-v2.0.json"
    out_train = "../../data/hardem_format/squad_train.json"
    out_dev = "../../data/hardem_format/squad_dev.json"
    #squad2min(train_squad_loc, out_train, tokenizer)
    #squad2min(dev_squad_loc, out_dev, tokenizer)

    print("RACE starts")
    train_race_loc = "../../data/RACE/train"
    dev_race_loc = "../../data/RACE/dev"
    out_train = "../../data/hardem_format/race_train.json"
    out_dev = "../../data/hardem_format/race_dev.json"
    #race2min(train_race_loc, out_train, tokenizer)
    #race2min(dev_race_loc, out_dev, tokenizer)

    print("WikiHop starts")
    train_wh_loc = "../../data/qangaroo_v1.1/wikihop/train.json"
    dev_wh_loc = "../../data/qangaroo_v1.1/wikihop/dev.json"
    out_train = "../../data/hardem_format/wikihop_train.json"
    out_dev = "../../data/hardem_format/wikihop_dev.json"
    #wikihop2min(train_wh_loc, out_train, tokenizer)
    #wikihop2min(dev_wh_loc, out_dev, tokenizer)
    
    print("MSMARCO starts")
    train_msmarco_loc = "../../data/msmarco/train_v2.1.json_indent"
    dev_msmarco_loc = "../../data/msmarco/dev_v2.1.json"
    out_train = "../../data/hardem_format/msmarco_train_indent.json"
    out_dev = "../../data/hardem_format/msmarco_dev.json"
    msmarco2min(train_msmarco_loc, out_train, tokenizer)
    msmarco2min(dev_msmarco_loc, out_dev, tokenizer)
