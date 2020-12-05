# IBIT: Intervention-based Invariant Transfer learning

This is a PyTorch implementation of **IBIT** from

**Intervention Design for Effective Sim2Real Transfer** by

## Instructions
To train the 
```
python train.py 
```
This will produce the `runs` folder, where all the outputs are going to be stored including train/eval logs, tensorboard blobs, and evaluation episode videos. To launch tensorboard run
```
tensorboard --logdir runs
```

The console output is also available in a form:
```
| train | E: 5 | S: 5000 | R: 11.4359 | D: 66.8 s | BR: 0.0581 | ALOSS: -1.0640 | CLOSS: 0.0996 | TLOSS: -23.1683 | TVAL: 0.0945 | AENT: 3.8132
```
a training entry decodes as
```
train - training episode
E - total number of episodes 
S - total number of environment steps
R - episode return
D - duration in seconds
BR - average reward of a sampled batch
ALOSS - average loss of the actor
CLOSS - average loss of the critic
TLOSS - average loss of the temperature parameter
TVAL - the value of temperature
AENT - the actor's entropy
```
while an evaluation entry
```
| eval  | E: 20 | S: 20000 | R: 10.9356
```
contains 
```
E - evaluation was performed after E episodes
S - evaluation was performed after S environment steps
R - average episode return computed over `num_eval_episodes` (usually 10)
```

## Acknowledgements 
We used [kornia](https://github.com/kornia/kornia) for data augmentation.

ROS_Interface

Modified DMSuite

Modified Wrapper
