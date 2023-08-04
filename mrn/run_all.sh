mkdir -p results/trainlogs

# 'monolithic', 'bilinear', 'l2', 'asym', 'dn', 'wn'

gpu=0 # gpu number

#for method in monolithic bilinear dn wn sym max asym-max
for method in asym-max
do
    for seed in 100 200 300 400 500
    do
        #./run.sh monolithic $seed $gpu > results/trainlogs/$method_$seed.log 2>&1 &
        #./run.sh $method $seed $gpu > results/trainlogs/$method_$seed.log 2>&1 &
        ./run.sh $method $seed $gpu
        #./run.sh $method $seed $gpu
    done
done
