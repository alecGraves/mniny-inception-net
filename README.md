# mniny-inception-net
This is my version of inception net for mnist. I use [keras](https://github.com/fchollet/keras) with [tensorflow](https://github.com/tensorflow/tensorflow) for this project. It was created as part of a competition between friends (I won with the lowest categorical crossentropy). I created this architecture after the one detailed in [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842). Major changes made include downscaling, simplifying kernel sizes, and adding Batch Normalization. I also use [this](https://github.com/titu1994/Snapshot-Ensembles) snapshot ensembles object (thanks, Somshubra Majumdar!).

# Notable Results:

Results gained after approx. 24 hours of training on a NVIDIA GTX 1060 6GB GPU. 
Some of these results are without ensembling. No data augmentation is used.

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
```python
from mniny_inception_net import train

run = 0
while True:
    train(run)
    run += 1
```

## Evaluation

### ensemble
```python
from mniny_inception_module import evaluate_ensemble

# To evaluate ensemble of all models in weights folder:
evaluate_ensemble(Best=False)

# To evaluate ensemble of best models per training session:
evaluate_ensemble()
```

### individual
```python
from mniny_inception_module import evaluate

#Evaluate all models in weights directory:
evaluate(eval_all=True)

# Evaluate 'Best' models in weights directory:
evaluate()
```

## Requirements
* Tensorflow
* Keras

### Setting up an Environment:
1. Install [Anaconda](https://www.continuum.io/downloads)
2. ```pip install tensorflow-gpu keras```
