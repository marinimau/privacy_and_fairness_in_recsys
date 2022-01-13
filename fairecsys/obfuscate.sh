DIR="./fairrecsys/"
if [ ! -d "$DIR" ]; then
  git clone https://github.com/edizel/fairrecsys.git
fi

source ../venv/bin/activate

# python3 fairrecsys/src/alg_distMet_mu1.py ../../altered_m1 ../../input/users.csv ../../input/recs.csv 20 0.2

python3 alg_distMet_mu1.py altered_fair_matrix ../data/ml-1m-users.csv ../data/ml-1m-pred-wrmf.csv 20 0.2