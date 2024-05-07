for run in 1 2 3
do
for representation in SMILES
do
for epoch in 4
do
for train_size in 1000
do
for target in y_bin
do
echo mit run $run Epoch $epoch Train size $train_size $representation $target 
python run_experiments_sh.py $train_size $run EleutherAI/gpt-j-6b $epoch $representation $target

done
done
done
done
done