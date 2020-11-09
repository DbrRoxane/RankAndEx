#dabord sur SUM les weak label - qui sont dans fichier min_test.json
# il faut ici avoir meilleur score que 67.96/67.84 en rouge l et 27.05/28.24 en b4

import jsonlines
import json

def convert2pgnet(input_file, output_file, train=True):
    output_file_src = output_file + "-src.txt"
    src_writer = open(output_file_src, mode="w")
    output_file_tgt1 = output_file  + "-tgt1.txt"
    output_file_tgt2 = output_file  + "-tgt2.txt"
    tgt1_writer = open(output_file_tgt1, mode="w")
    tgt2_writer = open(output_file_tgt2, mode="w") 
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            question = obj["question"]
            answer_gt1 = obj["final_answers"][0] 
            answer_gt2 = obj["final_answers"][1]
            answer_spans = " ".join(obj["final_answers"][2:]) if obj["final_answers"][2:] else "No span!" 
            src_writer.write("{} || {}\n".format(question, answer_spans))
            if train : 
                tgt1_writer.write("{} \n".format(answer_gt1))
                tgt2_writer.write("{} \n".format(answer_gt2))
    tgt1_writer.close()
    tgt2_writer.close()

def convert2coqa(input_file, output_file, id_answer=0):
    output_data = {"version": "NARRATIVEQA_SUM_COQAFORMAT", "data": []}
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            one_story = {"source": "narrative_qa",
                         "id" : obj["id"],
                         "name": "narrative_qa",
                         "filename" : obj["id"].split("_")[0],
                         "story" : " ".join([word for cont in obj["context"] for word in cont ]),
                         "questions":
                         [{
                             "input_text": obj["question"],
                             "turn_id" : eval(obj["id"].split("_")[1][1:])
                         }]
                        }
            size_context = [len(para) for para in obj["context"]]
            answers = []
            final_answer = obj["final_answers"]
            for i, para in enumerate(obj["answers"]):
                if len(para) > 0 : 
                    answer = para[0]
                    span_start = size_context[i] + answer["word_start"]
                    span_end = size_context[i] + answer["word_end"]
                    answers.append({"span_text":answer["text"],
                                    "input_text":final_answer[id_answer],
                                    "span_start":span_start,
                                    "span_end":span_end,
                                    "turn_id" : eval(obj["id"].split("_")[1][1:])
                                   })
                    break
            if len(answers) == 0:
                answers.append({"span_text": "unknown",
                                "input_text":"unknown",
                                "span_start":-1,
                                "span_end":-1,
                                "turn_id" : eval(obj["id"].split("_")[1][1:])
                               })
            assert len(answers)==1
            one_story["answers"] = answers
            output_data["data"].append(one_story)
    with open(output_file, "w") as writer:
        json.dump(output_data, writer, indent=4) 


#convert2pgnet("./data/squad_format/rbc_sum_train_point_split3.json", "./data/pgnet_format/min_train_weaklabel", train=True)
#convert2pgnet("./data/squad_format/rbc_sum_dev_point_split3.json", "./data/pgnet_format/min_dev_weaklabel", train=True)

convert2coqa("./data/squad_format/rbc_sum_train_point_split3.json", "./data/coqa_format/rbc_sum_train_coqa_format.json", 0)
convert2coqa("./data/squad_format/rbc_sum_dev_point_split3.json", "./data/coqa_format/rbc_sum_dev_coqa_format.json", 0)
#ensuite sur sum avec prediction

# ensuite sur story weak label

# ensuite sur story avec predictions

