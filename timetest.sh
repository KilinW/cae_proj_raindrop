for i in $(seq 0.96 0.01 2.88); do
    echo current factor $i
    python3 test.py --force $i
done
