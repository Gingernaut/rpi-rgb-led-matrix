set -x

git reset --hard origin/master
git pull
git reset --hard origin/master

python3 main.py