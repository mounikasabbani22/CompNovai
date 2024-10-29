Original Data Based Model Files:
	This folder contains total 6 python notebooks, that are accessing "Real" data. So, although we cannot share our real confidential dataset, these notebooks will give you an idea about our various modeling approaches.
	_helper_function: Contains generic functions needed by other notebooks
  1. Exploratory_Data_Analysis Of Features: Notebook to perform exploratory data analysis and feature selection
  2. Data Preparation_Chunking_Embeddings: Notebook to chunk selected feature/document and create embeddings
  3. Create Self-managed Vector Search Index & Zero Shot Learning: Notebook to create vector search index on the embeddings table for faster retrival and perform zero-shot learning to get prediction
  4. Fine-Tuning a Pre-trained LLM Model: Notebook to fine tune pre-trained LLM model (Bert)
  5. Embeddings Based MultiClass Prediction: Notebook to train and inference multi-class classifiction model, trained on real data embeddings

Synthetic Data Based Model Files:
This folder contains total 5 python notebooks, that are accessing "Synthetic" data which is an alternate way to showcase our data and explore various modeling approaches.
  1. Preparing_Synthetic_Data_for_RAG: As the name suggests, this notebook focusses on generating synthetic data to be fed to the RAG pipeline to perform further tasks.
  2. Synth_Data_Vector_Search_Index: After performing embeddings on the existing synthetic data, a vector search index table would be created to fetch the docs quickly.
  3. Evaluation_Metrics_MLFlow_SynthData: This notebook evaluates the quality of synthetic data vs the original data which showcases the similarity of the datasets.
  4. Synth_data_BERT_Based_Uncased: This notebook performs model development by running it through google's BERT Based model and measures accuracy accordingly.
  5. Synth_Data_ZeroShot_Learning: The zero shot learning is another model that has been used for predicting the failure mode bucket and measures accuracy.

CompNovAi UI:
  1. requirements.txt - Covers the dependencies.
  2. chatbot.py - Main python notebook.
  3. compnovai_logo.PNG - Logo image file that is used across all material.
  4. synth_bert_model_test_result.csv - This is the data file that the UI fetches data.
  5. config.toml (/.streamlit) - This file is the intial CSS file.
  6. devcontainer.json (/.devcontainer) - This is the container file used for publishing the streamlit app on public cloud.
 
Commands used to run in terminal:
  1. pip install -r requirements.txt
  2. pip install streamlit
  3. streamlit run chatbot.py

Clone the githb repository into any of the IDEs such as Databricks, Google Colab or Jupyter notebooks in order to run the code base files and render the same on the UI using the streamlit app.
