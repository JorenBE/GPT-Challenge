for run in 1 2 3 4 5 6 7 8 9 10
do
for representation in reaction
do
for epoch in 50 
do
for train_size in 5 10 13
do
for target in yield_bin
do
for model in meta-llama/Meta-Llama-3.1-8B-Instruct EleutherAI/gpt-j-6b mistralai/Mistral-7B-Instruct-v0.3   
do
echo cmc run $run $model Epoch $epoch Train size $train_size $representation $target
python run_experiments_sh.py $train_size $run $model $epoch $representation $target train_leander.csv 6

done
done
done
done
done
done