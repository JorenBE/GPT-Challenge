for run in 1 2 3
do
for representation in formula_eqPressure_25C
do
for epoch in 50
do
for train_size in 100 200 300 350
do
for target in binary_Heat
do
echo cmc run $run Epoch $epoch Train size $train_size $representation $target
python run_experiments_sh.py $train_size $run mistralai/Mistral-7B-Instruct-v0.3 $epoch $representation $target Hydrides.csv 50

done
done
done
done
done
