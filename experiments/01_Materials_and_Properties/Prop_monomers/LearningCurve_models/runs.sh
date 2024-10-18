for run in 1 2 3
do
for representation in SMILES
do
for epoch in 20 
do
for train_size in 10 50 100 300
do
for target in E_coh_bin T_g_bin R_gyr_bin Densities_bin
do
for model in meta-llama/Meta-Llama-3.1-8B-Instruct EleutherAI/gpt-j-6b mistralai/Mistral-7B-Instruct-v0.3   
do
echo cmc run $run $model Epoch $epoch Train size $train_size $representation $target
python run_experiments_sh.py $train_size $run $model $epoch $representation $target train_ben.csv 50

done
done
done
done
done
done