DIR="./fairrecsys/"
if [ ! -d "$DIR" ]; then
  git clone https://github.com/marinimau/fairrecsys.git
fi

source ../venv/bin/activate

python3 fairrecsys/src/alg_distMet_mu1.py output/altered.csv input/users.csv input/recs.csv 20 0.2