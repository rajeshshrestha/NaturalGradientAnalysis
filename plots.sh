# figure 1
python experiments/exp1.py --dataset weather --lrs 3e-3 --layers 4 --activations relu None --task regression --epochs 10 --running-loss-window 7 --lambda-reg 1e-4 --use-tengrad --use-exact-ngd --use-sgd

# figure 2
python experiments/exp1.py --dataset weather --lrs 3e-4 3e-3 3e-2 3e-1 --layers 4 --activations relu None --task regression --epochs 10 --running-loss-window 7 --lambda-reg 1e-4 --use-tengrad --use-exact-ngd --use-sgd

# figure 3
python experiments/exp1.py --dataset weather --lrs 3e-3 3e-2 3e-1 --layers 32 16 --activations relu relu None --task regression --epochs 3 --running-loss-window 7 --lambda-reg 1e-4 --use-tengrad --use-exact-ngd --use-sgd

# figure 4
python experiments/exp1.py --dataset weather --lrs 3e-3 3e-2 3e-1 --layers 32 32 32 16 --activations relu relu relu relu None --task regression --epochs 3 --running-loss-window 7 --lambda-reg 1e-4 --use-tengrad --use-exact-ngd --use-sgd

# figure 5
python experiments/exp1.py --dataset weather --lrs 3e-3 3e-2 3e-1 --layers 32 16 --activations relu relu None --batch-size 8 --task regression --epochs 3 --running-loss-window 7 --lambda-reg 1e-4 --use-tengrad --use-exact-ngd --use-sgd

# figure 6
python experiments/exp1.py --dataset weather --lrs 3e-3 3e-2 3e-1 --layers 32 16 --activations relu relu None --batch-size 32 --task regression --epochs 3 --running-loss-window 7 --lambda-reg 1e-4 --use-tengrad --use-exact-ngd --use-sgd

# figure 7
python experiments/exp1.py --dataset weather --lrs 3e-3 3e-2 3e-1 --layers 32 16 --activations relu relu None --batch-size 1024 --task regression --epochs 3 --running-loss-window 7 --lambda-reg 1e-4 --use-tengrad --use-exact-ngd --use-sgd

# Figure 8
python experiments/exp3.py --dataset weather  --task regression --epochs 1 --hidden-layer-size 20 --hidden-layer-nums  10 20 30 40 50 --use-tengrad --use-block-ngd --use-exact-ngd --batch-size=32 --analyze-time --analyze-memory

# Figure 9
python experiments/exp3.py --dataset weather  --task regression --epochs 1 --hidden-layer-size 20 --hidden-layer-nums  10 20 30 40 50 100 --use-tengrad --use-sgd --use-block-ngd --batch-size=32 --analyze-time
