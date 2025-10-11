# parallel_ssh iplist "docker exec llama_factory bash -c \"ps -ef | grep llamafactory | awk '{print \$2}' | xargs kill -9\""
# parallel_ssh iplist "docker exec llama_factory bash -c \"kill -9 `ps -ef | grep train_pt.sh | awk '{print $2}'`\""
# parallel_ssh iplist "docker exec llama_factory bash -c \"kill -9 `ps -ef | grep torchrun | awk '{print $2}'`\""


docker exec llama_factory bash -c "ps -ef | grep llamafactory | awk '{print \$2}' | xargs kill -kill"
docker exec llama_factory bash -c "ps -ef | grep train_pt.sh | awk '{print \$2}' | xargs kill -kill"
docker exec llama_factory bash -c "ps -ef | grep torchrun | awk '{print \$2}' | xargs kill -kill"
