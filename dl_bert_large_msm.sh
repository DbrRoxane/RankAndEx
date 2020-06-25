#!/bin/bash
fileid="1crlASTMlsihALlkabAQP6JTYIZwC1Wm8"
filename="BERT_LARGE_MSMARCO.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
