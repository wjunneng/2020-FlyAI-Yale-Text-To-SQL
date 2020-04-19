# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash

data=/home/wjunneng/Ubuntu/NLP/2020_3/IRNet/data/dev.json
table_data=/home/wjunneng/Ubuntu/NLP/2020_3/IRNet/data/tables.json
output=/home/wjunneng/Ubuntu/NLP/2020_3/IRNet/data/output

echo "Start download NLTK data"
python download_nltk.py

echo "Start process the origin Spider dataset"
python data_process.py --data_path ${data} --table_path ${table_data} --output "process_data.json"

echo "Start generate SemQL from SQL"
python sql2SemQL.py --data_path process_data.json --table_path ${table_data} --output ${data}

rm process_data.json
