# figure 1: Comparison of method for small model with the same lr
py/thon experiments/exp1.py --dataset weather --lrs 3e-3 --layers 4 --activations relu None --task regression --epochs 10 --running-loss-window 7 --lambda-reg 1e-4 --use-tengrad --use-exact-ngd --use-sgd


# figure 2: With grid search lrs. Best for each one
python experiments/exp2.py --dataset weather --tengrad-lr 3e-3 --sgd-lr 3e-1 --exact-ngd-lr 3e-3 --layers 4 --activations relu None --task regression --epochs 10 --running-loss-window 7 --lambda-reg 1e-4 

# figure 3: With grid search lrs. Best for each one and lr decay
python experiments/exp2.py --dataset weather --tengrad-lr 3e-3 --sgd-lr 3e-1 --exact-ngd-lr 3e-3 --layers 4 --activations relu None --task regression --epochs 10 --running-loss-window 7 --lambda-reg 1e-4 --lr-decay 

# Figure 1,2 and 3 for bigger model
# figure 4: Comparison of method with the same lr
python experiments/exp1.py --dataset weather --lrs 3e-3 --layers 32 16 --activations relu relu None --task regression --epochs 10 --running-loss-window 7 --lambda-reg 1e-4 --use-tengrad --use-exact-ngd --use-sgd


# figure 5: With grid search lrs. Best for each one
python experiments/exp2.py --dataset weather --tengrad-lr 3e-3 --sgd-lr 3e-1 --exact-ngd-lr 3e-3 --layers 32 16 --activations relu relu None --task regression --epochs 10 --running-loss-window 7 --lambda-reg 1e-4 

# figure 6: With grid search lrs. Best for each one and lr decay
python experiments/exp2.py --dataset weather --tengrad-lr 3e-3 --sgd-lr 3e-1 --exact-ngd-lr 3e-3 --layers 32 16 --activations relu relu None --task regression --epochs 10 --running-loss-window 7 --lambda-reg 1e-4 --lr-decay 

#Effect of lr decay due to noisy update in tengrad
# figure 7: Comparison of method with the same lr but without lr decay
python experiments/exp1.py --dataset weather --lrs 1e-3 1e-2 1e-1 --layers 128 64 --activations relu relu None --task regression --epochs 10 --running-loss-window 7 --lambda-reg 1e-12 --use-tengrad --use-sgd

# figure 8: Comparison of method with the same lr but with smooth lr decay
python experiments/exp1.py --dataset weather --lrs 1e-3 1e-2 1e-1 --layers 128 64 --activations relu relu None --task regression --epochs 10 --running-loss-window 7 --lambda-reg 1e-12 --lr-decay --lr-decay-type smooth --use-tengrad --use-sgd

# figure 8: Comparison of method with the same lr but with step lr decay
python experiments/exp1.py --dataset weather --lrs 1e-3 1e-2 1e-1 --layers 128 64 --activations relu relu None --task regression --epochs 10 --running-loss-window 7 --lambda-reg 1e-12 --lr-decay --lr-decay-type step --use-tengrad --use-sgd


# For deep network with high number of layers, time and space analysis
# figure : Comparison of small model with the same lr
python experiments/exp1.py --dataset weather --lrs 3e-3 --layers 4 4 4 4 4 4 --activations sigmoid sigmoid sigmoid sigmoid sigmoid sigmoid None --task regression --epochs 10 --running-loss-window 7 --lambda-reg 1e-12 --alpha 1e-6

# figure : With grid search lrs. Best for each one
python experiments/exp1.py --dataset weather --lrs 1e-3 1e-2 1e-1 --layers 4 4 4 4 4 4 --activations sigmoid sigmoid sigmoid sigmoid sigmoid sigmoid None --task regression --epochs 10 --running-loss-window 7 --lambda-reg 1e-12 --alpha 1e-6

# figure : With grid search lrs. Best for each one and smooth lr decay
python experiments/exp1.py --dataset weather --lrs 1e-3 1e-2 1e-1 --layers 4 4 4 4 4 4 --activations sigmoid sigmoid sigmoid sigmoid sigmoid sigmoid None --task regression --epochs 10 --running-loss-window 7 --lambda-reg 1e-12 --lr-decay --lr-decay-type smooth --alpha 1e-6

# figure : With grid search lrs. Best for each one and step lr decay
python experiments/exp2.py --dataset weather --lrs 1e-3 1e-2 1e-1 --layers 4 4 4 4 4 4 --activations sigmoid sigmoid sigmoid sigmoid sigmoid sigmoid None --task regression --epochs 10 --running-loss-window 7 --lambda-reg 1e-12 --lr-decay --lr-decay-type step --alpha 1e-6


# For deep network time and space analysis based on the layer num
# Figure 9
python experiments/exp3.py --dataset weather  --task regression --epochs 1 --hidden-layer-size 20 --hidden-layer-nums  10 20 30 40 50 --use-tengrad --use-block-ngd --use-exact-ngd --batch-size=32 --analyze-time --analyze-memory
# Figure 10
python experiments/exp3.py --dataset weather  --task regression --epochs 1 --hidden-layer-size 20 --hidden-layer-nums  10 20 30 40 50 100 --use-tengrad --use-sgd --use-block-ngd --batch-size=32 --analyze-time

# For deep network time and space analysis based on the batch size
# Figure
python experiments/exp4.py --dataset weather  --task regression --epochs 1 --hidden-layer-size 20 --hidden-layer-num  20 --use-tengrad --use-block-ngd --use-exact-ngd --batch-sizes 32 64 128 256 1024 --analyze-time --analyze-memory
# Figure
python experiments/exp4.py --dataset weather  --task regression --epochs 1 --hidden-layer-size 20 --hidden-layer-num  20 --use-tengrad --use-sgd --use-block-ngd --batch-sizes 32 64 128 256 1024 --analyze-time


# For house price prediction task
# Same lr
python experiments/exp1.py --dataset houseprice --lrs 3e-3 --layers 4 --activations relu None --task regression --epochs 20 --running-loss-window 7 --lambda-reg 0 --use-tengrad --use-exact-ngd --use-sgd  

# Best lr
python experiments/exp2.py --dataset houseprice --tengrad-lr 3e-3 --sgd-lr 3e-1 --exact-ngd-lr 3e-3 --layers 4 --activations relu None --task regression --epochs 20 --running-loss-window 7 --lambda-reg 0

# lr grid search
python experiments/exp1.py --dataset houseprice --lrs 3e-3 1e-3 3e-2 1e-2 3e-1 1e-1 --layers 128 64 --activations relu relu None --task regression --epochs 20 --running-loss-window 7 --lambda-reg 1e-6 --use-tengrad --use-sgd

# lr grid search with smooth decay
python experiments/exp1.py --dataset houseprice --lrs 3e-3 1e-3 3e-2 1e-2 3e-1 1e-1--layers 128 64 --activations relu relu None --task regression --epochs 20 --running-loss-window 7 --lambda-reg 1e-6 --use-tengrad --use-sgd --lr-decay --lr-decay-type smooth

# lr grid search with step decay
python experiments/exp1.py --dataset houseprice --lrs 3e-3 1e-3 3e-2 1e-2 3e-1 1e-1 --layers 128 64 --activations relu relu None --task regression --epochs 20 --running-loss-window 7 --lambda-reg 1e-6 --use-tengrad --use-sgd --lr-decay --lr-decay-type step


#figure 11
python experiments/exp1.py --dataset separable_svm --lrs 1e-4 1e-3 1e-2 1e-1 --layers 4 --activations relu sigmoid --task classification --epochs 100 --running-loss-window 7 --lambda-reg 1e-6 --use-tengrad  --use-sgd   --batch-size=32 # --lr-decay --lr-decay-type='smooth'

#figure 12
python experiments/exp1.py --dataset overlapped_svm --lrs 1e-6 1e-4 1e-3 1e-2 --layers 4 --activations relu sigmoid --task classification --epochs 100 --running-loss-window 7 --lambda-reg 1e-6 --use-tengrad  --use-sgd   --batch-size=32  # --lr-decay --lr-decay-type='step'

#figure 13
python experiments/exp1.py --dataset houseprice --lrs 3e-4 3e-3 3e-2 3e-1 --layers 4 --activations relu None --task regression --epochs 100 --running-loss-window 7 --lambda-reg 1e-6 --use-tengrad  --use-sgd   --batch-size=32 --alpha=1e-6