cd training
bash train.sh -t halfcheetah-medium-v2,hopper-medium-replay-v2 -c tuned-b1.5 -d cuda:0 -j 6 -i 2
cd ../eval
bash eval.sh -t halfcheetah-medium-v2,hopper-medium-replay-v2 -c tuned-b1.5 -d cuda:0 -j 8 -i 2
