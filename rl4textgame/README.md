## Installation
```bash
git clone git@github.com:pearls-lab/llm-search-textgame.git
cd llm-search-textgame
pip install -r requirements.txt

cd verl
git submodule init
git submodule update
pip install -r requirements.txt
pip install awscli
```

## Single-turn RL
### Data process
Prepare train, validation, and test data in `.jsonl` format. 
Make sure to configure aws s3 first. 
For textworld data, run the following to transform data into parquet form. 
```bash
sh rl4textgame/data_preprocess.sh
```
