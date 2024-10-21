for run in 1 2 3
do
for representation in SMILES InChI iupac omega
do
for epoch in 25
do
for train_size in 10 25 50 100 150
do
for target in mp_bin
do
echo denis run $run Epoch $epoch Train size $train_size $representation $target 
python run_experiments_sh.py $train_size $run EleutherAI/gpt-j-6b $epoch  $target $representation

done
done
done
done
done