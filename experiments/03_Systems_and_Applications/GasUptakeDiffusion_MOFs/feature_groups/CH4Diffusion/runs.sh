for run in 1 2 3
do
for target in Diffusion_bool
do
for representation in mofkey mofid A AB ABC ABCD B
do
for epoch in 25
do
for train_size in 500
do

script_location="$(cd "$(dirname "$0")" && pwd)"

echo MOF run $run $target $representation Epoch $epoch Train size $train_size 
python $script_location/run_experiments_sh.py $train_size $run EleutherAI/gpt-j-6b $epoch $target $representation

done
done
done
done
done