# Knowledge Distilling
Implementation of Distilling the Knowledge in a Neural Network https://arxiv.org/pdf/1503.02531.pdf

#### Reference
    * https://arxiv.org/pdf/1503.02531.pdf
    * https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    * https://github.com/wonbeomjang/Knowledge-Distilling-PyTorch/blob/master/loss.py
    * https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py
    * https://github.com/peterliht/knowledge-distillation-pytorch/issues/2
    * https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    * https://keras.io/examples/vision/knowledge_distillation/
    * https://github.com/SforAiDl/KD_Lib/blob/master/README.md
    * https://intellabs.github.io/distiller/knowledge_distillation.html#hinton-et-al-2015
    * https://viblo.asia/p/paper-explain-thu-thuat-cho-cac-mang-cnn-trong-classification-LzD5dRMoZjY
    * [KD: A good teacher is patient and consistent](https://github.com/google-research/big_transfer)

#### Compare
    * optimize speed and memory
    * dont need train teacher with Pseudo Teacher


#### TODO
- [x] write custom dataset for outp_Teacher, see [data.py](https://github.com/watson21/Knowledge-Distillation/blob/main/data.py#L74)
- [ ] write metric to caculator in training, accuracy, precision, recall, F1-score, see [link](https://machinelearningcoban.com/2017/08/31/evaluation/)
- [ ] write custom dataset for ImageFolder
- [x] write Pseudo Teacher, input index class, return vector of Probability distribution of class [0.1, 0.4, 0.3, 0.2, 0.0, 0.0, 0.0, ...], maybe generate random array and put it thought softmax and get output.
- [ ] add feature for Pseudo Teacher
    -> input accuracy for each class; For example: 5 classes [0.8, 0.89, 1.0, 0.5, 0.99]   
    -> return return vector of Probability distribution of class
- [ ] write unittest for PseudoTeacher, check vector output, random wrong index, num_data wrong/total data(1.-acc)
- [ ] write Distiller like keras template
- [x] [Augmentation](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0) using [albumentation](https://github.com/albumentations-team/albumentations) 
- [ ] using [augmentor](https://github.com/mdbloice/Augmentor)
- [ ] implement mixup: dataset and function aug, ref timm
- [ ] implement [Label smoothing](https://github.com/seominseok0429/label-smoothing-visualization-pytorch.git) template to using target transform in dataset
