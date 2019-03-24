# Word embedding model Skip Gram with Negative Sampling

The project includes 2 parts:

**Notebooks**:
- word2vec.ipynb is implementation for skip-gram negative sampling
- topic_modeling.ipynb is implementation for LDA
- keyword_mapping.ipynb is implementation for topic-keywords network

In html folder is the static version of notebook

**Word2vec code**:
Word2vec run able program is in src folder. To try this,
- Install requirements.txt dependence
- From project directory, run "python src/main.py"
- The model will be saved in "models/w2v.txt" 

**Features**:
- Word embedding by skip-gram negative sampling
- Training in 37 languages from https://github.com/ufal/rh_nntagging
- Training from text file
- Training from Gensim dataset
- Save model in word vectors format
- Support GPU
 
 Adjustable params for Word2Vec class:
 
- lang = "english"
- n_epoch = 20
- batch_size = 500
- embed_dim = 300
- window_size = 5
- neg_sample = 10
- min_count = 5
- lr = 0.01
- report_every = 1
