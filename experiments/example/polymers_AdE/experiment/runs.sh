for run in 1 2 3 # loop over three seeds
do
for representation in sequence_prompt
do
for epoch in 4
do
for train_size in 100 500 1000 5000 # loop over various training sizes
do
for target in E_adh_2classes
do
echo Polymers run $run Epoch $epoch Train size $train_size $representation $target 
python run_experiments_sh.py $train_size $run EleutherAI/gpt-j-6b $epoch $representation $target

done
done
done
done
done