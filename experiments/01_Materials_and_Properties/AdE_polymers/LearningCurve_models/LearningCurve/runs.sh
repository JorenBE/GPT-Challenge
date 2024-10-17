for run in 1 2 3
do
for representation in SMILES
do
for epoch in 4
do
for train_size in 400 500 1000 5000
do
for target in y_bin
do
echo cmc run $run Epoch $epoch Train size $train_size $representation $target
python run_experiments_sh.py $train_size $run meta-llama/Meta-Llama-3.1-8B $epoch $representation $target train_polymers.csv 50

done
done
done
done
done