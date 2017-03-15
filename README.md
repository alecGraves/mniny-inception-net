# mniny-inception-net
This is my version of inception net for mnist. It was created as part of a competition between friends (I won with the lowest categorical crossentropy). I created this architecture using inspiration gained by [this](https://arxiv.org/abs/1409.4842) paper.

# Notable Results:

Results gained after approx. 24 hours of training on a NVIDIA GTX 1060 GPU

--------------------------------------

ensemble

Test loss (categorical crossentropy): 0.0100604627563

error: 0.330001115799%

--------------------------------------
individual

Test loss: 0.0148106931992

error: 0.34%

--------------------------------------
individual

Test loss: 0.0129321537635

error: 0.34%

--------------------------------------

individual

Test loss: 0.0130156670398

error: 0.34%

--------------------------------------

# Usage
## Training
```
from mniny_inception_module import train
run = 0
while True:
    train(run)
    run += 1
```
## Evaluations
### Ensembles
```
from mniny_inception_module import evaluate_ensemble

# To evaluate all models in weights folder:
evaluate_ensemble(Best=False)

# To evaluate ensemble of best models per training session:
evaluate_ensemble()
```
### Individuals
```
from mniny_inception_module import evaluate

#Evaluate all models in weights directory:
evaluate(eval_all=True)

# Evaluate 'Best' models in weights directory:
evaluate()
```
