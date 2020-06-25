# RankAndEx
End-to-End System to perform on NarrativeQA over stories. 

## Guide to reproduce the experiment
All the commands are launched from the main directory of this repository.

## Get the initial dataset
`mkdir data`

`cd data

git clone https://github.com/deepmind/narrativeqa.git
 
cd narrativeqa
 
./download_stories.sh `
 
## Passage Ranking

The passage ranking can be made with BERT, BM25 or TF-IDF (cosine similarity).
The one using BERT takes more time to develop but is supposed to perform really better (see [this paper](https://arxiv.org/abs/1901.04085)).
In any case, the first step is to chunk the stories into smaller passages.

`mkdir data/processed

python chunk_narrativeqa_stories.py`

This script will store the chunked stories into paragraphs into `data/processed/narratativeqa_all.eval` 
Here is a graphic to have an idea of the amount of chunks per story and their size ![alt text](https://github.com/DbrRoxane/RankAndEx/blob/master/stat_chunks.png)

The results of the prediction will be in any case a .tsv file with 3 colums : the query id, the passage id and its rank
This .tsv should be stored in `data/ranking` (`mkdir data/ranking` )



### Passage Ranking with BERT. 
The passage ranking is made thanks to [this repository](https://github.com/nyu-dl/dl4marco-bert). 
We will reuse their architecture and training on MSMarco and run the model on NarrativeQA. 
This section describes how to convert the NarrativeQA stories in order to run the Nogueria's model on it. 
You can either generate the data by yourself or download mine (soon available). 
To do it by yourself, follow the instructions bellow.

#### Download BERT pretrained on MSMARCO
You can chose between BERT Large and BERT base. In our case, we use BERT Large since we do not fine-tune. 
Base model

`sh dl_bert_base_msm.sh

 unzip BERT_BASE_MSMARCO.zip -d BERT_BASE_MSMARCO`
 
Large model
 `sh dl_bert_large_msm.sh
 
 unzip BERT_LARGE_MSMARCO.zip -d BERT_LARGE_MSMARCO`


#### Create the input files of the Passage Ranker 

`python convert_nqa_to_tfrecord.py`

The script will store tensorflow files and the query_doc_ids files into ./data/processed folder.
The files are split in 8 subfiles to allow the processing. 
You can choose to create an "oracle" model, i.e using the answer to rank the passage thanks to the parameter "use_answer". 

#### Run your experiment with Google Colab

Create a project on GCS (or any other cloud) and store the output of the tf convertor in addition to the bert folder. 
Then you can run the experiment with Google TPUs using [this adapted version of Nogueria's CoLab](https://colab.research.google.com/drive/1tR30uAIOQeniEv-noppFdtbWN_t5IJZr?usp=sharing) (the original is available [here](https://drive.google.com/open?id=1vaON2QlidC0rwZ8JFrdciWW68PYKb9Iu)). 
The only params to chages are on the paths in [this cell](https://drive.google.com/open?id=1vaON2QlidC0rwZ8JFrdciWW68PYKb9Iu)
Do not forget to copy the output from GCS to `data/output`

### Passage Ranking with BM25 and TF-IDF
The data can be regenerate or downlodad here (available soon).
To generate them: 
 - run on BM25, `python run_bm25_nqa.py`. 
 - run on TF-IDF `python run_tfidf_nqa.py`

You can choose to create an "oracle" model, i.e using the answer to rank the passage thanks to the parameter "use_answer". 
The input and output files might also be updated. 

## Answer extraction
For this part, we have used the code from [this repository](https://github.com/shmsw25/qa-hard-em) associated with [this paper](https://arxiv.org/abs/1909.04849)

For our experiment, we have fine-tuned BERT for Question Answering with, in a first place, NarrativeQA over summaries and then on NarrativeQA over stories with an oracle ranking.
We call this second fine-tuning Cheated Ranking on Naive Answer Extraction (CRNAE) and compute it by using the 3 best predictions of BERT, BM25 and TF-IDF using the answer).
We test our model on the stories with a Naive Ranking on Naive Answer Extraction (NRNAE) which is computed using the 20 best predictions of BERT and BM25. 
To convert our data in a processable format for the Hard-EM algorithm (which is the same format as in SQUAD).

### Data Conversion
The algorithm works for summaries as well as with stories by turning the task from generative to extractive by extracting the possible answers thanks to ROUGE-score (see details in paper soon available). 
`python convert_squad_format.py` with your own arguments
(`python convert_squad_format.py -h` to see the list of arguments).
The data should be stored in `data/processed`

### Running the Hard-EM

Clone the initial repository [here](https://github.com/shmsw25/qa-hard-em.git) and change your python environment to make it fit with this repository.

To make it run on the summaries, which have only one passage per context, change the following.

Change [the line](https://github.com/shmsw25/qa-hard-em/blob/5620861901f097c69264e64dc77c7d653f193812/DataLoader.py#L11) by
`self.input_ids, self.input_mask, self.segment_ids = [torch.cat([i for i in input], 0) \
                                            for input in [input_ids, input_mask, segment_ids]]`

and [the line](https://github.com/shmsw25/qa-hard-em/blob/5620861901f097c69264e64dc77c7d653f193812/DataLoader.py#L16) by 
`self.start_positions, self.end_positions, self.switches, self.answer_mask = [torch.cat([i \
                for i in input], 0) for input in [start_positions, end_positions, switches, answer_mask]]`

Also, [this line](https://github.com/shmsw25/qa-hard-em/blob/5620861901f097c69264e64dc77c7d653f193812/prepro.py#L315) has to be changed by `switches.append(3)`

The expermiment can now be launched with adapting the parameters into the `run.sh` file. 
We have trained with hard-em loss and, as suggested in their paper in Appendix B., with a tau value of 2k.
Our batch size is 16 for training and 256 for prediction. 
Tthe random seed is set to 42.  
We first train with the initial checkpoint being the same as in their repository, i.e Bert Base unsaced for Question Answering. 
Our train/dev file is computed with the previous data conversion section on summaries with 0.6 as a threshold (ranking files and max_rank can be empty params). 
The best model is then the initial checkpoint of our second fine-tuning, with this time the data converted with oracle ranking files (unse_answer=True) of Bert Passage Reranking, BM25 and Tf-IDF; using their 3 highest-ranked paragraphs. 
The final test in processed on data converted with naive oracle and the 20th best passage from Bert Passage Reranking in addition to BM25.
In both case the rouge threshold is still 0.6 .

This will create some prediction files with the argument in run `--do_predict`.
These predictions can be stored into `data/predictions`

Our results are presented bellow (TODO)

## Evaluation

To evaluate our predictions, we have chosen to use the same evaluation metrics and code as in [this paper](https://arxiv.org/abs/1905.10847) to be able to compare fairly our score. 
That is why we use [this repository](https://github.com/yicheng-w/CommonSenseMultiHopQA/), which use the same [base](https://github.com/tylin/coco-caption). 

To do so, we generate the ground truth evaluation files with this script `python generate_evalfile_groundtruth.py `.
The predictions are converted into a text file in the same format with `convert_hardempred2bauerpred.py`
We can then pass as arguments the ground truths files (`<ref0> and <ref1>`) generates as well as the prediction file (`<output>`) to the script in the Commonsense 
repo `python src/eval_generation.py <ref0> <ref1> <output>`
