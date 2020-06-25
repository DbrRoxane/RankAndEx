#!/bin/bash
fileid="1cyUrhs7JaCJTTu-DjFUqP6Bs4f8a6JTX"
filename="BERT_BASE_MSMARCO.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
