for run in 1 2 3
do
for representation in SMILES
do
for epoch in 15 20
do
for train_size in 10 50 100 300
do
for target in E_coh_bin T_g_bin R_gyr_bin Densities_bin
do
echo denis run $run Epoch $epoch Train size $train_size $representation $target 
python run_experiments_sh.py $train_size $run EleutherAI/gpt-j-6b $epoch $representation $target

done
done
done
done
done