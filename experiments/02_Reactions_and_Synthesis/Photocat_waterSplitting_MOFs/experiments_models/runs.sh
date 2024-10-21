for run in 1 2 3
do
for representation in mofid elemetns_all
do
for epoch in 25 
do
for train_size in 50
do
for target in HER_bool OER_bool VIS_bool
do
for model in meta-llama/Meta-Llama-3.1-8B-Instruct EleutherAI/gpt-j-6b mistralai/Mistral-7B-Instruct-v0.3   
do
echo cmc run $run $model Epoch $epoch Train size $train_size $representation $target
python run_experiments_sh.py $train_size $run $model $epoch $representation $target final_noDuplicates.csv 50

done
done
done
done
done
done