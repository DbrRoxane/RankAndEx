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
                a = [[{"text":answer["span_text"], "word_start":answer["span_start"], "word_end":answer["span_end"]}]]
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
                    for answer in answers:
                        answer["span_start"] = answer.pop("answer_start")
                        answer["span_end"] = answer["span_start"] + len(tokenizer.tokenize(answer["text"].strip()))
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
                print(q,a,final, " ".join(context[0]))


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
    out_dev = "../../data/hardem_format/squada_dev.json"
    #squad2min(train_squad_loc, out_train, tokenizer)
    #squad2min(dev_squad_loc, out_dev, tokenizer)

    print("RACE starts")
    train_race_loc = "../../data/RACE/train"
    dev_race_loc = "../../data/RACE/dev"
    out_train = "../../data/hardem_format/race_train.json"
    out_dev = "../../data/hardem_format/race_dev.json"
    race2min(train_race_loc, out_train, tokenizer)
    race2min(dev_race_loc, out_dev, tokenizer)
