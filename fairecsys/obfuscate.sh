DIR="./fairrecsys/"
if [ ! -d "$DIR" ]; then
  git clone https://github.com/marinimau/fairrecsys.git
fi

source ../venv/bin/activate

classLabels=("gender" "age")
cutOffs=(10 20 50)
models=("user-knn" "bprmf" "wrmf" "multidae")

for m in "${models[@]}"
do
  for c in "${cutOffs[@]}"
  do
    for g in "${classLabels[@]}";
    do
      python3 fairrecsys/src/alg_distMet_mu1.py output/altered_"$m"_"$g"_cutoff_"$c".csv input/users_"$g".csv input/recs.csv "$c" 0.2
    done
  done
done