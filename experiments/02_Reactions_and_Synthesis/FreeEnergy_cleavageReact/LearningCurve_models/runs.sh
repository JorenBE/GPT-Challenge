for run in 1 2 3
do
for representation in Smiles
do
for epoch in 20
do
for train_size in 10 50 100 500 1000
do
for target in deltaG4_bin
do
echo LLPS run $run Epoch $epoch Train size $train_size $representation $target
python run_experiments_sh.py $train_size $run mistralai/Mistral-7B-Instruct-v0.3 $epoch $representation $target

done
done
done
done
done