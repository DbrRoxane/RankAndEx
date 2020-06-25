from utils import convert_docs_in_dic
import tokenization

BOOK_EVAL_FILE = "./data/narrativeqa/narrativeqa_all.eval"
dataset = convert_docs_in_dic(BOOK_EVAL_FILE)

tokenizer = tokenization.BasicTokenizer()

def process_answer(answer):
    answer1 = " ".join(tokenizer.tokenize(answer))
    answer1 = answer1[:-2] if answer1[-1]=="." else answer1
    return answer1


val_ref1 = open("val_ref1.txt", "w")
val_ref2 = open("val_ref2.txt", "w")
test_ref1 = open("test_ref1.txt", "w")
test_ref2 = open("test_ref2.txt", "w")

for story in dataset.values():
    if story['set'] == "valid":
        for query in story['queries'].values():
            val_ref1.write(process_answer(query['answer1'])+"\n")
            val_ref2.write(process_answer(query['answer2'])+"\n")
    elif story['set'] == "test":
        for query in story['queries'].values():
            test_ref1.write(process_answer(query['answer1'])+"\n")
            test_ref2.write(process_answer(query['answer2'])+"\n")

val_ref1.close()
val_ref2.close()
test_ref1.close()
test_ref2.close()


