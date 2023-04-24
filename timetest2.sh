for i in $(seq 0.01 0.01 0.95); do
    echo current factor $i
    python3 test.py --force $i
done
