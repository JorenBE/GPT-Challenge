for run in 1 2 3
do
for target in HER_bool OER_bool VIS_bool
do
for representation in elements_linkers elements_nodes elemetns_all
do
for epoch in 25
do
for train_size in 50
do
echo Hydrides run $run $target $representation Epoch $epoch Train size $train_size 
python run_experiments_sh.py $train_size $run EleutherAI/gpt-j-6b $epoch $target $representation

done
done
done
done
done