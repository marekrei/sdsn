
SDSN: Supervised Directional Similarity Network
===============================================

Project for scoring lexical relations between two words, for example detecting hyponym relations.

This is code for the following paper:

[Scoring Lexical Entailment with a Supervised Directional Similarity Network](http://aclweb.org/anthology/P18-2101)  
Marek Rei, Daniela Gerz and Ivan Vulić  
In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL 2018)  
Melbourne, Australia, 2018  


Paper: [http://aclweb.org/anthology/P18-2101](http://aclweb.org/anthology/P18-2101)  
ArXiv: [https://arxiv.org/abs/1805.09355](https://arxiv.org/abs/1805.09355)  
ACL 2018 presentation: [https://vimeo.com/285805844](https://vimeo.com/285805844)  
ACL 2018 slides: [http://www.marekrei.com/pub/presentation-2018-acl-ssnhyperlex.pdf](http://www.marekrei.com/pub/presentation-2018-acl-ssnhyperlex.pdf)  


Requirements
-----------------------------------------------

Tested with:

    python 3.6
    theano 1.0.3
    lasagne 0.2.dev1

This conda installation works for me:

    conda create --name sdsn python=3.6 theano=1.0.3 cython
    conda activate sdsn
    pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip



Running
-----------------------------------------------

Specify settings, including file paths, in a config file. See the conf/ directory for example config files.

Then run an experiment with:

    python experiment.py path/to/config.conf

To run on the GPU you might use: 

    THEANO_FLAGS='device=cuda0,floatX=float32' python -u experiment.py path/to/config.conf


Data format
-----------------------------------------------

The input file format looks like this, with the (potential) hyponym in the second column, hypernym in the third column and the human judgement score in the first column:

    7.18    trail   follow
    7.5     mason   worker
    0.15    radish  carrot

For the experiments with sparse distributional features, you can preprocess the dataset and add numerical features as additional columns to the end of each line:

    7.18    trail   follow  0.0447 0.0873 0.0964 0.4513  0.0495 0.0983 0.1752  0.1935  0.7331  0.1153
    7.5     mason   worker  0.0393 0.0739 0.0733  0.4007  0.0556 0.0707 0.1325  0.1368   0.6092  0.1300
    0.15    radish  carrot  0.0255  0.0661 0.0617 0.3695  0.0342 0.1927  0.3901  0.3792  0.6737  0.1025




Example configs
-----------------------------------------------
Example configuration files for replicating the experiments in [Table 1 of the paper](http://aclweb.org/anthology/P18-2101) can be found in conf/conf_acl18/.
Due to using newer versions of theano, lasagne and CUDA, the results with the latest code are slightly different but the conclusions hold. These are the replicated results using the current repository, averaged over 10 random seeds, measuring Spearman's correlation.


Lexical splits:

||**DEV mean** |**DEV std** |**DEV median** |**TEST mean** |**TEST std** |**TEST median** |
|---|---|---|---|---|---|---|
|SDSN |0.5640 |0.0380 |0.5718 |0.4632 |0.0395 |0.4636 |
|SDSN+SDF |0.5535 |0.0189 |0.5540 |0.4832 |0.0218 |0.4926 |
|SDSN+SDF+AS |**0.5902** |0.0104 |**0.5890** |**0.5459** |0.0322 |**0.5454** |



Random splits:

||**DEV mean** |**DEV std** |**DEV median** |**TEST mean** |**TEST std** |**TEST median** |
|---|---|---|---|---|---|---|
|SDSN |0.6994 |0.0142 |0.7054 |0.6598 |0.0120 |0.6575 |
|SDSN+SDF |0.7220 |0.0144 |0.7212 |0.6797 |0.0065 |0.6803 |
|SDSN+SDF+AS |**0.7679** |0.0143 |**0.7714** |**0.6890** |0.0092 |**0.6900** |



Pretrained model
-----------------------------------------------
We also make a pretrained model available, so it could be used for downstream experiments.
However, because the sparse distributional features need to be calculated for each word pair beforehand, we do not include the SDF component in this model. Instead, this is the regular SDSN, with additional supervision (pre-training), trained on HyperLex with random splits.

[SDSN+AS model trained on random splits](https://s3-eu-west-1.amazonaws.com/sdsn-models/model_randomsplits.sdsn_as.v0.model)

The configuration file used for training them can be found in conf/conf_saved/.

Because this is trained with a specific random seed, as opposed to an average over 10 seeds, we also report the performance of this model separately:


||DEV |TEST |
|---|---|---|
|SDSN+AS (v0) |0.7455 |0.6842 |


To try out the pretrained model, run:

    python example.py path/to/model.model

Enter two words, for example "highway road" in order to check if highway is a type of road. The script will return a confidence score from the model, with values between 0 and 10.

