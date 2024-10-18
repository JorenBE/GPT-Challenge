for run in 1 2 3
do
for target in rate_bin
do
for representation in SMILES+Molar
do
for epoch in 25 50 100
do
for train_size in 15
do
for dataset in 0 1 2 3 4 5 6 7
do

for model in meta-llama/Meta-Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3   
do

echo rate run $run $target $representation Epoch $epoch Train size $train_size 
python run_experiments_sh.py $train_size $run $model $epoch $target $representation $dataset

done
done
done
done
done
done
done