start=$(date +%s)

python3 test.py --force 1.80

end=$(date +%s)
take=$(( end - start))
echo use ${take} seconds.
