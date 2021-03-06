# EmoNet
## Introduction
EmoNet is a neural network tool for emotion recognition, described in this [paper](https://www.aclweb.org/anthology/P17-1067/). The current packaged version predicts 9 emotion categories (p2 in the paper). This is a work in progress, and so the rest of categories will be added. ... [TBD]

## Installation
This repo is tested on Python 3.6+, PyTorch 1.2.0+ and TensorFlow 2.0.0

First you need to install both, [TensorFlow](https://www.tensorflow.org/install/pip#tensorflow-2.0-rc-is-available) 2.0 and [PyTorch](https://pytorch.org/get-started/locally/#start-locally) .

### With pip
[TBD]

### From Source
Once TensorFlow 2.0 and PyTorch has been installed, you can install from source by running:
```
pip install git+https://github.com/UBC-NLP/EmoNet.git
```
It will automatically install other dependencies including '
happiestfuntokenizing', 'transofrmers', 'numpy' and 'pandas'.

## Usage
### As a Python Library
You can use emonet as a Python library.

*EmoNet.predict*


| **Parameters** | **Description** | 
|---| --- | 
| *text* | Specifies the string for prediction |
| *path* | Specifies the file path to the tsv file for prediction. (It must have a 'content' column) |
| *with_dist* | Specifies whether diplay the distrubution over all class labels |


**Example**:
```python
from emonet import EmoNet

em = EmoNet()

# predict a text language and returning the distribution over all languages
prediction = em.predict(text='Spectacular day in Brisbane today. Perfect for sitting in the sun and thinking up big ideas and resetting plans.', with_dist=True)
print("Predict a text and display the distribution:")
print(prediction)

# Predict text in a tsv file line by line
predictions = em.predict(path=path_to_tsv_file)
print("Predict a text file line by line:")
print(predictions)


```

Here is the output:
```
Predict a text and display the distribution
[('joy', 0.8978073, {'anger': 0.0008576517, 'anticipation': 0.06090205, 'disgust': 0.00068270933, 'fear': 0.007252514, 'joy': 0.8978073, 'sadness': 0.004249889, 'surprise': 0.025819499, 'trust': 0.0024283403})]

Predict a text file line by line:
[('joy', 0.9871133), ('anger', 0.94085765), ('fear', 0.99755955), ('anticipation', 0.98000574), ('joy', 0.5602796), ('joy', 0.35310036)]
```


### As A Command-Line Tool
> emonet.py [options]

***Options***: 

| -b, --batch | specify a file path on the command line | 
|---| --- | 
| -d, --dist | show full distribution over languages |

**Example**:

Assuming `path/to/file/text.tsv` containts the following: 
```
content
I am happy about this 
That movie was scary
```

then running: 

```
$ emonet -b path/to/file/text.tsv --dist
```
gets you the output: 
```
[('joy', 0.85855854, {'anger': 0.009707213, 'anticipation': 0.011866086, 'disgust': 0.00912305, 'fear': 0.018755471, 'joy': 0.85855854, 'sadness': 0.039343264, 'surprise': 0.037687544, 'trust': 0.014958782}), 
('fear', 0.6364692, {'anger': 0.02605511, 'anticipation': 0.020012422, 'disgust': 0.013645536, 'fear': 0.6364692, 'joy': 0.09334463, 'sadness': 0.14103319, 'surprise': 0.06486984, 'trust': 0.00457005})]
```


### Interactive Mode
It is very simple to use the interactive mode. Invoke using ` python
emonet.py ` or just ` emonet.py ` if you have already installed the
package.


```
~> python emonet.py
>>> Spectacular day in Brisbane today. Perfect for sitting in the sun and thinking up big ideas and resetting plans.
[('joy', 0.8978073)]
>>>
```

You can also specify a file path by using `--batch` option. The script
also detect when the input is redirected.
```
python emonet.py < test.tsv
[('joy', 0.8978073)]
```
