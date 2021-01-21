set -x
source .venv/bin/activate

git reset --hard origin/master
git pull
git reset --hard origin/master

python3 main.py