#!/bin/bash
# run:
# > sh pach.sh [project_name]

cp -r project_template $1
cp ../../frameworks/vllm_server/run.sh $1/run_vllm_server.sh
zip -r $1.zip $1
rm -rf $1
