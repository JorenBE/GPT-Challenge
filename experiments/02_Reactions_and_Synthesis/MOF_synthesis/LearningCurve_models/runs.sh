for run in 1 2 3
do
for representation in prompt_raw
do
for epoch in 50 
do
for train_size in 5 10 15 20
do
for target in completion_raw
do
for model in meta-llama/Meta-Llama-3.1-8B-Instruct EleutherAI/gpt-j-6b mistralai/Mistral-7B-Instruct-v0.3   
do
echo cmc run $run $model Epoch $epoch Train size $train_size $representation $target
python run_experiments_sh.py $train_size $run $model $epoch $representation $target train_taddei.csv 50

done
done
done
done
done
done