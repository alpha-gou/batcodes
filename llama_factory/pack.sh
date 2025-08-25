#!/bin/bash
# run:
# > sh pach.sh project_name

cp -r code_files $1
cp ../vllm_server/run_vllm_server.sh $1/
zip -r $1.zip $1
rm -rf $1
