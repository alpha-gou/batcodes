#!/bin/bash
# run:
# > sh pach.sh [project_name]

cp -r project_template $1
cp ../vllm_server/run_vllm_server.sh $1/
zip -r $1.zip $1
rm -rf $1
