### NLP models

A suite of NLP TensorFlow (and Keras) workhorse models targeting the [Toxic Comments classification problem](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) and easily adapted to other NLP tasks. 
The modules are largely self-contained, performing multi-fold training and evaluation against the ~150K sample Toxic Comments data-set. 

| Model |Module| Comment|Paper|
|-------|------|--------|-----|
|Keras BiGRU FastText|[Module](keras_bigru_fasttext_base.py)|Keras implementation of 2-layer Bidirectional-GRU using pre-trained FastText embeddings||
|TF BiGRU FastText|[Module](tf_bigru_fasttext_base.py)|TensorFlow implementation of 2-layer Bidirectional-GRU using pre-trained FastText embeddings||
|Keras BERT|[Module](keras_bert_base.py)| [Modified version](https://github.com/kpot/keras-transformer) of Keras implementation of BERT|[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)|
|Keras BiGRU ELMo| [Module](keras_bigru_elmo_base.py)|Keras 2-layer Bidirectional-GRU using pre-trained ELMo embeddings from tensorflow hub|[Deep contextualized word representations](https://arxiv.org/abs/1802.05365)|
|Keras BiGRU ELMo (generator)|[Module](keras_bigru_elmo_generator.py)|Abortive attempt to use a multiprocessing-based generator to load the ELMo embeddings faster|[Deep contextualized word representations](https://arxiv.org/abs/1802.05365)|
|Keras BiGRU FastText w/ MLM as an auxiliary loss|[Module](keras_bigru_fasttext_mlm_auxiliary.py)|Added secondary task of Masked Language Model|
|Keras BiGRU FastText w/ MLM and pseudo-labelling|[Module](keras_bigru_fasttext_mlm_auxiliary_test_fakelabels.py)|MLM + pseudo-labelling|
|Keras BiGRU FastText w/ MLM and Tied out weights|[Module](keras_bigru_fasttext_mlm_auxiliary_tiedoutweights.py)|MLM + tied out weights|
|TF BERT|[Module](tf_bert.py)|[Modified version](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) of Nvidia + Google's TensorFlow implementation of BERT with support for AMP (i.e., FP16)|
|TF GPT|[Module](tf_gpt.py)|[Modified version](https://github.com/openai/gpt-2) of OpenAI's TensorFlow implementation of GPT|[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)|
|Distilled BERT|[Module](tf_bert_soft_targets.py)|BERT modified to train against combined loss function of soft & hard targets (i.e., knowledge distillation)|[Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)

### Setup
1. A [DockerFile](Dockerfile) is provided which builds against an Nvidia-provided Docker image, and installs the necessary Nvidia, system, and Python prerequisites - IMPORTANT: the Dockerfile installs an SSH server with a default password
2. The Toxic comments [training and test CSV files](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) are to be stored in the data/ sub-folder.
3. Pre-trained models can be downloaded for BERT ([here](https://github.com/google-research/bert#pre-trained-models)) and GPT ([here](https://storage.googleapis.com/gpt-2/))
4. In theory, users with the right Nvidia GPU can enable FP16 training by simply setting an OS environment setting within the docker container (export TF_ENABLE_AUTO_MIXED_PRECISION=1); in practice, the Nvidia BERT implementation required [manual TF variable casting to FP16](tf_transformer/gpu_environment.py) and [optimizer loss scaling](custom_optimization.py). Another quick optimization for Nvidia GPU users is to just XLA JIT compilation by setting the right ConfigProto flag (config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1). On a Titan RTX, training speed closed to tripled with both optimizations activated.

### Useful readings
1. [Generalized Language Models](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html) - Great survey of modern (post-Word2Vec) NLP techniques ranging from CoVe all-the-way to BERT
2. [Enriching Word Vectors with Subword Information](https://aclweb.org/anthology/Q17-1010) - FastText word embeddings
3. [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Paper that lays out the Transformer architecture that is the basis for recent architectures such as BERT and GPT
4. [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) - Lays out the BPE compression/tokenization technique
5. [Universal Transformers](https://arxiv.org/abs/1807.03819) - Improvement over the base Transformer architecture
6. [Nvidia Automatic Mixed Precision](https://devblogs.nvidia.com/nvidia-automatic-mixed-precision-tensorflow/) - Lays out Nvidia's FP16 training technique
7. [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)

