# W266-Final-Project

This repository contains all the code that was ultimately used to develop this Final Project.  The files are as follows:

* **lstm_baseline.ipynb:** Used to create the baseline LSTM model for reference
* **pytorch-practice.ipynb:** Used to experiment with pytorch 
* **pretraining.py:** Script used in combination with **L1 Pretraining.ipynb** to pretrain the L1 Data
* **L1 Pretraining.ipynb:** Notebook used in combination with **pretraining.py** to pretrain the L1 Data
* **L3 Bert.ipynb:** Notebook used to experiment with fine-tuning BERT on the L3 Data
* **L3_from_pretrain.py:** Script used to train L3 Data on fine-tuned model and evaluate F1 scores
* **L3_rebuild.py:** Idk looks basically same as above Neha will know
* **BERT_Preprocess.ipynb:** Notebook used to fine-tune BERT on L1 and L2 Data and train on L3 Data

The following code references were used:

* https://aws.amazon.com/blogs/machine-learning/maximizing-nlp-model-performance-with-automatic-model-tuning-in-amazon-sagemaker/
* https://github.com/danwild/sagemaker-sentiment-analysis/blob/163913a21837683e7605f6122ad2c10718347f65/train/train.py#L45
* https://mccormickml.com/2019/07/22/BERT-fine-tuning/#3-tokenization--input-formatting
* https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py?fbclid=IwAR37EHe9W3DW08xrcBurwUiAKLCSyrn6L1aSSpRyVdoINMMotGG8I7TsFDw

### Authors

* Sayan Das
* Neha Kumar

