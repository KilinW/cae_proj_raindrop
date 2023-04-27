for j in $(seq 0.000011 0.000001 0.000022); do
    for i in $(seq 0.96 0.01 2.88); do
        echo current state $i $j
        python3 cantilever_piezo_voltage.py --force $i --mass $j --output /home/aicenter/cae_proj_raindrop/new-data
    done
done