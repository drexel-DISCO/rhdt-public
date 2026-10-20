bash training/train.sh -c tuned -i 2 -d cuda:0,cuda:1 -j 12
bash eval/eval.sh -c tuned -i 2 -d cuda:0,cuda:1 -j 12
