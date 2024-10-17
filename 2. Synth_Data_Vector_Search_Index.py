# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Fine-Tuning an Embedding Model
# MAGIC
# MAGIC It's often desirable to fine-tune your out-of-the-box embedding model to see better retrieval performance on your domain-specific data. 
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Prepare necessary training and evaluation data for fine-tuning embedding model 
# MAGIC  - Fine-tune a popular embedding model 
# MAGIC  - Embed the dataset to Vector Search index using the newly fine-tuned embedding model 
# MAGIC  - Serve the model using Provisioned Throughput 
# MAGIC  - Evaluate the model using recall metrics and manual inspection
# MAGIC
# MAGIC In this notebook, we will fine-tune our out-of-the-box embedding model based on PySpark documentation. We will then compare retrieval performances when using a generic embedding model versus a fine-tuned embedding model.

# COMMAND ----------

# MAGIC %pip install mlflow==2.14.1 sentence-transformers==3.0.1 torch==2.3.0 transformers==4.40.1 databricks-sdk==0.31.1 tf-keras==2.16.0 accelerate==0.27.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# %run ./Includes/Classroom-Setup

# COMMAND ----------

# # %sql

# -- select * from cmidev.cbu_ccims_poc.synth_data_claims_feature;
import pandas as pd
import matplotlib.pyplot as plt
import pyspark.sql.functions as F
from random import randint
from typing import List, Callable

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceDoesNotExist, ResourceAlreadyExists
from databricks.sdk.service.vectorsearch import VectorIndexType, DeltaSyncVectorIndexSpecResponse, EmbeddingSourceColumn, PipelineType
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput

from langchain_core.embeddings import Embeddings

from mlflow.tracking.client import MlflowClient

from datasets import Dataset 

import sentence_transformers, requests, time, json, mlflow, yaml, os, torch
from sentence_transformers import InputExample, losses
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from transformers import AutoTokenizer

from torch.utils.data import DataLoader
import torch

import tempfile

from databricks.sdk.service.serving import AutoCaptureConfigInput

# COMMAND ----------

# MAGIC %sh PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # Prevent PyTorch from hogging too much GPU memory in reserve.

# COMMAND ----------

# VS_ENDPOINT = "adv_genai_course_endpoint_1" 
# VS_INDEX = "spark_docs_bge_finetuned"

# DBRX_ENDPOINT = "databricks-dbrx-instruct"
# LLAMA3_ENDPOINT = "databricks-meta-llama-3-1-70b-instruct"
# GENERATED_QUESTIONS = "generated_questions"
# GOLD_CHUNKS_FLAT = "spark_docs_gold_flat"
VS_ENDPOINT = "vs_endpoint_synthetic_claims_data"

gpu_is_available = torch.cuda.is_available()

# COMMAND ----------

# VS_INDEX_FULL_NAME = f"{SHARED_CATALOG}.{SHARED_SCHEMA}.{VS_INDEX}"
VS_INDEX_FULL_NAME= f"cmidev.cbu_ccims_poc.synthetic_claim_text_embeddings_index"


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We load an open-sourced embedding model. Then, we will fine-tune this embedding model based on the training data.

# COMMAND ----------

import pandas as pd
import numpy as np
def load_embeddings_chunks(claim_id):
  embeddings_chunks =[]
  label=[]
  df=pd.DataFrame()

  embeddings = spark.sql("select CLAIM_ID_SEQ, embedding from cmidev.cbu_ccims_poc.synthetic_data_claim_text_embeddings_gte where CLAIM_ID_SEQ =  "+str(claim_id)+" ").toPandas()
  for row in range(len(embeddings)):   
    if(row !=7):
      for i in embeddings['embedding'].iloc[row]: 
        embeddings_chunks.append(i)
     
  # print((embeddings_chunks))
  return embeddings_chunks

# COMMAND ----------

def ConcatenateFields(test_claims):
    from pyspark.sql.functions import concat_ws, struct
    
    # df_final_updated = test_claims.withColumn("FC", concat_ws(",", "FC_1","FC_2","FC_3","FC_4","FC_5"))
    final_test_df = test_claims.withColumn("DOCUMENT", struct("OEM_CODE", "DEALER", "DISTR", "FC", "ENGINE_NAME_DESC", "FAILCODE", "SHOPORDERNUM",  "CLEANED_CORRECTION"))
    # display("-----",type(final_test_df))
    return final_test_df

# COMMAND ----------

def get_label(claim_id):
  label = spark.sql("select FAILURE_MODE_BUCKET from cmidev.cbu_ccims_poc.synth_data_claims_feature where CLAIM_ID_SEQ =  "+str(claim_id)+" ").toPandas()
  return (label['FAILURE_MODE_BUCKET'].iloc[0])


# COMMAND ----------

# get_label(claim_id=claim_id)

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

# Define a window spec partitioned by FAILURE_MODE_BUCKET and ordered by CLAIM_ID_SEQ
windowSpec = Window.partitionBy("FAILURE_MODE_BUCKET").orderBy("CLAIM_ID_SEQ")

# Add a row number for each record within each partition
df_with_row_number = spark.sql("""
    SELECT *, ROW_NUMBER() OVER (PARTITION BY FAILURE_MODE_BUCKET ORDER BY CLAIM_ID_SEQ) AS rn
    FROM cmidev.cbu_ccims_poc.synth_data_claims_feature where FAILURE_MODE_BUCKET != 'Not Inspected'
""")

# Filter to keep only records with row number <= 20
df_filtered = df_with_row_number.filter("rn <= 5")

# Group by FAILURE_MODE_BUCKET and count the records in each group
df_counts = df_filtered.groupBy("FAILURE_MODE_BUCKET").count()

# Filter to keep only FAILURE_MODE_BUCKET groups with exactly 20 records
df_exact_5 = df_counts.filter("count = 5")

# Join back to the filtered DataFrame to get the original records, but only for FAILURE_MODE_BUCKET with exactly 20 records
df_final = df_filtered.join(df_exact_5, "FAILURE_MODE_BUCKET").drop("count", "rn")

display(df_final)

# COMMAND ----------

df_final.groupBy('FAILURE_MODE_BUCKET').count().display()

# COMMAND ----------

def CreateDatasetWithoutFailModeBucket(test_claims):
    # query = "select * from cmidev.cbu_ccims_poc.ccims_claim_inspection_feature where FAILURE_MODE_BUCKET != 'Not Inspected' and FAILURE_MODE_BUCKET IN ('ACT(E) BROKEN SOLDER JOINT','ACT(E) ELECTROLYTIC C26 FAILURE', 'TSS - INTERMITTENT SIGNAL') limit "+str(limit)+""
    # test_claims = spark.sql(query)

    final_test_df=ConcatenateFields(test_claims)
    final_test_df = final_test_df.toPandas()
    

    return final_test_df
    
df = CreateDatasetWithoutFailModeBucket(df_final)
df = df[["CLAIM_ID_SEQ","DOCUMENT","FAILURE_MODE_BUCKET"]]
df.rename(columns={"CLAIM_ID_SEQ":"claim_id",
              "DOCUMENT":"document",
              "FAILURE_MODE_BUCKET":"label"},inplace=True)

df.head(4)

# COMMAND ----------

# import pandas as pd
# from pyspark.sql import functions as F
# df = pd.DataFrame()
# def CreateDataset(limit):
#   query = "select CLAIM_ID_SEQ,DOCUMENT, FAILURE_MODE_BUCKET  from cmidev.cbu_ccims_poc.ccims_claim_inspection_feature where FAILURE_MODE_BUCKET !='Not Inspected' limit  "+str(limit)+""
#   print(query)


#   claims = spark.sql(query).toPandas()
#   df = pd.DataFrame()
#   chunks=[]
#   for claim_id in claims['CLAIM_ID_SEQ'].tolist():
#       embeddings_chunks = []
#       embeddings_chunks = (load_embeddings_chunks(claim_id=claim_id))
#       chunks.append(embeddings_chunks)
#   df['claim_id'] = claims['CLAIM_ID_SEQ'].tolist()
#   df['embeddings'] = chunks#str(embeddings_chunks)
#   df['document'] = claims['DOCUMENT'].tolist()
#   df['label'] = claims['FAILURE_MODE_BUCKET'].tolist()
#   return df

# # Convert the Pandas DataFrame to a PySpark DataFrame
# # sdf = spark.createDataFrame(df)

# # Display the DataFrame
# df = CreateDataset(limit=50)
# display(df)

# COMMAND ----------

len(np.unique(df['label']))

# COMMAND ----------

df.groupby('label').count()
# 'ACT(E) BROKEN SOLDER JOINT'	; 'ACT(E) ELECTROLYTIC C26 FAILURE'	; 'TSS - INTERMITTENT SIGNAL'	; 

# COMMAND ----------

from sklearn.model_selection import train_test_split  



import numpy as np  
  
# Load the chunked embeddings  
# embeddings_chunks = load_embeddings_chunks()  
  
# # Concatenate the embeddings into a single matrix  
# embeddings = np.concatenate(embeddings_chunks, axis=0)  
  
# Load the corresponding labels for each chunk  
# labels_chunks = load_labels_chunks()  

# labels = df['label'].tolist()
# embeddings = df['embeddings'].tolist()
# document = df['document'].tolist()

# labels = df['label'].tolist()
# embeddings = df['embeddings'].tolist()
# document = df['document'].tolist()

labels = df['label'].tolist()
document = df['document'].tolist()

# Concatenate the labels into a single array  
# labels = np.concatenate(labels_chunks, axis=0)  
  
# Split the data into training and test sets  
# X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.1, random_state=42,shuffle=True)  
X_train, X_test, y_train, y_test = train_test_split(document, labels, test_size=0.2, random_state=44, shuffle=True, stratify=labels)  




# COMMAND ----------

len(X_train)

# COMMAND ----------

len(X_test)

# COMMAND ----------

print((np.unique(y_train)))

# COMMAND ----------

print((y_test))

# COMMAND ----------

# MAGIC %md
# MAGIC To run below code successfully, make sure labels in y_test are present in y_train, else this code fails

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch

# Assuming y_train and y_test are pandas Series or lists of string labels

# Initialize the label encoder
label_encoder = LabelEncoder()

# Fit the label encoder and transform y_train and y_test to integer labels
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Convert the encoded labels to PyTorch tensors
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)


# COMMAND ----------

y_train_encoded

# COMMAND ----------

y_test_encoded

# COMMAND ----------

# MAGIC %md
# MAGIC Use below code if working with embeddings in X_train and X_test

# COMMAND ----------

# import torch
# from torch.nn.utils.rnn import pad_sequence

# # Assuming X_train and X_test are lists of tensors
# X_train_tensor = pad_sequence([torch.tensor(x, dtype=torch.float64) for x in X_train], batch_first=True)
# X_test_tensor = pad_sequence([torch.tensor(x, dtype=torch.float64) for x in X_test], batch_first=True)

# # Find the maximum sequence length from both train and test sets
# max_length = max(X_train_tensor.shape[1], X_test_tensor.shape[1])

# # Function to manually pad tensors to the same max_length
# def manual_pad(tensor, max_length):
#     return torch.nn.functional.pad(tensor, (0, max_length - tensor.shape[1]), 'constant', 0)

# # Pad the shorter sequences to have the same length as the longest sequence
# X_train_tensor = manual_pad(X_train_tensor, max_length)
# X_test_tensor = manual_pad(X_test_tensor, max_length)

# COMMAND ----------

# MAGIC %md
# MAGIC Use below code if working wiht text data in X_train and X_test

# COMMAND ----------

# import torch
# from torch.nn.utils.rnn import pad_sequence

# # Assuming X_train and X_test are lists of tensors
# X_train_tensor = pad_sequence([torch.tensor(str(x)) for x in X_train], batch_first=True)
# X_test_tensor = pad_sequence([torch.tensor(str(x)) for x in X_test], batch_first=True)

# # Find the maximum sequence length from both train and test sets
# max_length = max(X_train_tensor.shape[1], X_test_tensor.shape[1])

# # Function to manually pad tensors to the same max_length
# def manual_pad(tensor, max_length):
#     return torch.nn.functional.pad(tensor, (0, max_length - tensor.shape[1]), 'constant', 0)

# # Pad the shorter sequences to have the same length as the longest sequence
# X_train_tensor = manual_pad(X_train_tensor, max_length)
# X_test_tensor = manual_pad(X_test_tensor, max_length)

# COMMAND ----------

# (X_train[0]).keys()

# COMMAND ----------

from transformers import AutoTokenizer
bert_model_path = "/Volumes/cmidev/default/preventech/DBX_hack/google-bertbert-base-uncased"
# Initialize the tokenizer (example with BERT)
tokenizer = AutoTokenizer.from_pretrained(bert_model_path)

# Function to tokenize and encode the text data in the dictionaries
def encode_texts(text_dicts, tokenizer):
    texts = [text_dict['CLEANED_CORRECTION'] for text_dict in text_dicts]  # Extract text data
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Tokenize and encode X_train and X_test
X_train_encoded = encode_texts(X_train, tokenizer)
X_test_encoded = encode_texts(X_test, tokenizer)

# COMMAND ----------

def decode_texts(encoded_texts, tokenizer):
    decoded_texts = [tokenizer.decode(encoded_text, skip_special_tokens=True) for encoded_text in encoded_texts]
    return decoded_texts

# COMMAND ----------

X_test_encoded

# COMMAND ----------

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments,TFBertForSequenceClassification
import torch
import numpy as np
# Assuming you have a binary classification task
num_labels = len(np.unique(y_train))  # Adjust based on your task

# Load the pre-trained model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(bert_model_path, num_labels=num_labels)

# Custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Assuming y_train_encoded is your list/array of labels
train_dataset = CustomDataset(X_train_encoded, y_train_encoded)
val_dataset = CustomDataset(X_test_encoded, y_test_encoded)
# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    num_train_epochs=10,              # Total number of training epochs
    per_device_train_batch_size=8,   # Batch size per device during training
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Directory for storing logs
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset         

)

# Train the model
trainer.train()

# COMMAND ----------

len(X_train_encoded[0])

# COMMAND ----------

# !pip install torchsummary
# from torchsummary import summary


# summary(model, input_size=(1, 512))

# COMMAND ----------

model

# COMMAND ----------


# eval_results = trainer.evaluate(eval_dataset=X_test_encoded)  
# print(eval_results)

# COMMAND ----------

# Save the model
model.save_pretrained('/Volumes/cmidev/default/preventech/DBX_hack/qv942_ccims_bert_base_uncased_trained_model_v0')

# COMMAND ----------

from transformers import BertTokenizer

# Assuming 'model' is your BertForSequenceClassification model
# and 'tokenizer' is an instance of BertTokenizer or similar compatible with your model

input_text = "Replaced faulty turbocharger speed sensor due to broken solder joint. Performed functional test post-repair"

# Tokenize the input
inputs = tokenizer.encode_plus(
    input_text,
    add_special_tokens=True,
    return_tensors="pt"
)

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)

# Check if outputs is a tuple (for older versions of transformers)
if isinstance(outputs, tuple):
    logits = outputs[0]
else:
    logits = outputs.logits

# Assuming you want the class with the highest score
predictions = torch.argmax(logits, dim=-1)

# Display predictions
display(predictions)

# COMMAND ----------

label_encoder.inverse_transform(predictions)

# COMMAND ----------

# import torch
# from torch.nn.utils.rnn import pad_sequence

# # Assuming X_train and X_test are lists of dictionaries with numerical data under the key 'data'
# X_train_tensor = pad_sequence([torch.tensor(x['CLEANED_CORRECTION']) for x in X_train], batch_first=True)
# X_test_tensor = pad_sequence([torch.tensor(x['CLEANED_CORRECTION']) for x in X_test], batch_first=True)

# # Find the maximum sequence length from both train and test sets
# max_length = max(X_train_tensor.shape[1], X_test_tensor.shape[1])

# # Function to manually pad tensors to the same max_length
# def manual_pad(tensor, max_length):
#     return torch.nn.functional.pad(tensor, (0, max_length - tensor.shape[1]), 'constant', 0)

# # Pad the shorter sequences to have the same length as the longest sequence
# X_train_tensor = manual_pad(X_train_tensor, max_length)
# X_test_tensor = manual_pad(X_test_tensor, max_length)

# COMMAND ----------

print(X_train_tensor.shape,X_test_tensor.shape, len(y_train_tensor),len(y_test_tensor) )

# COMMAND ----------

def evaluate_model(model, dataloader):  
    model.eval()  
    total_loss = 0  
    total_correct = 0  
    total_samples = 0  
  
    with torch.no_grad():  
        for batch in dataloader:  
            # Get the inputs and labels for this batch  
            inputs, labels = batch  
  
            # Forward pass through the model  
            outputs = model(inputs)  
  
            # Compute the loss  
            loss = loss_function(outputs, labels)  
  
            # Compute the number of correct predictions  
            predictions = torch.argmax(outputs, dim=1)  
            correct = (predictions == labels).sum().item()  
  
            # Update the running totals  
            total_loss += loss.item() * inputs.size(0)  
            total_correct += correct  
            total_samples += inputs.size(0)  
  
    # Calculate the average loss and accuracy  
    average_loss = total_loss / total_samples  
    average_accuracy = total_correct / total_samples  
  
    return average_loss, average_accuracy  


# COMMAND ----------

# import torch
# import torch.nn
# from torch.utils.data import DataLoader, TensorDataset
# from transformers import AutoTokenizer, AutoModelForCausalLM, Pipeline, AutoModelForSeq2SeqLM,AutoModel


# train_dataset = TensorDataset(X_train_tensor, (y_train_tensor))
# test_dataset = TensorDataset(X_test_tensor, (y_test_tensor))

# # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Attempt to train model on embeddings

# COMMAND ----------



# Load your model
# Ensure 'model' is the correct model for your task, e.g., classification, question answering, etc.
bert_model_path = "/Volumes/cmidev/default/preventech/DBX_hack/google-bertbert-base-uncased"
# Define your loss function  
loss_function = torch.nn.CrossEntropyLoss()  
model = AutoModel.from_pretrained(bert_model_path, local_files_only=True, revision="no_timm")
# tokenizer = AutoTokenizer.from_pretrained(bert_model_path, local_files_only=True, revision="no_timm")
# Define your optimizer  
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) 
# inputs = tokenizer(prompt, question, truncation=True, padding="max_length", max_length=512, return_tensors="pt")  

batch_size = 16  
sequence_length = 17408  
embedding_size = 17408#512  

import torch.nn.utils.rnn as rnn_utils  
  
# Pad the input sequences to a fixed length of 17408  
padded_sequences = rnn_utils.pad_sequence(X_train_tensor, batch_first=True, padding_value=0, max_len=17408)  
  
# Reshape the padded sequences to match the expected input size of your model  
reshaped_sequences = padded_sequences.reshape(batch_size, sequence_length, -1)  
  
# Pass the reshaped sequences through your model  
outputs = model(reshaped_sequences)  


# Assuming that 'input_tensor' is your input tensor of size [batch_size, embedding_size]  
  
# Reshape the input tensor to [batch_size, sequence_length, embedding_size]  


# Transpose the input tensor to [sequence_length, batch_size, embedding_size]  

# for epoch in range(5):  
#     model.train()  
  
#     for batch in train_loader:  
#         print(batch)
#         # Get the inputs and labels for this batch  
#         inputs, labels = batch  
#         input_tensor = inputs.view(batch_size, sequence_length, -1)  
#         input_tensor = input_tensor.transpose(0, 1)  
#         # Clear the gradients from the previous iteration  
#         optimizer.zero_grad()  
  
#         # Forward pass through the model  
#         outputs = model(input_tensor)  
  
#         # Compute the loss  
#         loss = loss_function(outputs, labels)  
  
#         # Backward pass through the model  
#         loss.backward()  
  
#         # Update the model parameters  
#         optimizer.step()  
for epoch in range(5):  
    model.train()  

    for batch_idx, batch in enumerate(train_loader):  
        optimizer.zero_grad()  

        # Unpack the batch into inputs and labels  
        inputs, labels = batch[0], batch[1]  

        # Create the attention mask to mask out the padding tokens  
        attention_mask = (inputs != 0).float()  

        # Pass the inputs and attention mask through the model to get the predictions  
        outputs = model(inputs, attention_mask=attention_mask)  

        # Compute the loss between the predictions and the labels  
        loss = loss_function(outputs, labels)  

        # Backpropagate the loss and update the model parameters  
        loss.backward()  
        optimizer.step()  

        # Print some information about the training progress  
        if batch_idx % 10 == 0:  
            print('Epoch {} Batch {}: Loss = {:.4f}'.format(epoch, batch_idx, loss.item())) 
  
    # Evaluate the model on the test dataset  
    test_loss, test_accuracy = evaluate_model(model, test_loader)  
  
    # Print the loss and accuracy for this epoch  
    print(f"Epoch {epoch+1} - Train Loss: {loss.item():.3f} - Test Loss: {test_loss:.3f} - Test Accuracy: {test_accuracy:.3f}")  


# # Define your loss function and optimizer
# loss_function = torch.nn.CrossEntropyLoss()  # Adjust based on your task
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# # Training loop
# model.train()
# # for epoch in range(5):  # 'num_epochs' should be defined by you
# #     for embeddings, labels in train_loader:
# #         optimizer.zero_grad()
# #         outputs = model(embeddings)
# #         loss = loss_function(outputs, labels)
# #         loss.backward()
# #         optimizer.step()

# # Testing loop
# model.eval()
# model.predict()
# # total_correct = 0
# # with torch.no_grad():
# #     for embeddings, labels in  test_loader:
# #         outputs = model(embeddings)
# #         _, predicted = torch.max(outputs.data, 1)
# #         total_correct += (predicted == labels).sum().item()

# # print(f'Accuracy: {total_correct / len(test_dataset)}')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Before we can fine-tune the embedding model, we need to make sure that our training data conforms to the format that the `sentence transformer` library expects. From the `sentence transformer` library, we can represent our training data using the `InputExample` class. Refer to [documentation here](https://www.sbert.net/docs/training/overview.html#training-data).
# MAGIC
# MAGIC >As parameters, it accepts texts, which is a list of strings representing our pairs (or triplets). Further, we can also pass a label (either float or int).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Below code is from original Databricks training example, do not run unless updated

# COMMAND ----------

training_set = spark.table(f"{CATALOG}.{USER_SCHEMA}.{GENERATED_QUESTIONS}_train").toPandas()
eval_set = spark.table(f"{CATALOG}.{USER_SCHEMA}.{GENERATED_QUESTIONS}_eval").toPandas()
display(training_set)

# COMMAND ----------

for index, row in training_set.head(1).iterrows():
    print(f"index = {index}, question = {row.generated_question}, context = {row.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC Recall, the example from the slides:
# MAGIC
# MAGIC >Triplets consist of an anchor, positive example, and negative example.
# MAGIC
# MAGIC - Anchor: When was the first Land Rover designed?
# MAGIC - Positive: “The design for the original vehicle was started in 1947 by Maurice Wilks. Wilks, chief designer at the Rover Company, on his farm [...]”
# MAGIC - Negative: “Land Rover is a British brand of predominantly four-wheel drive, off-road capable vehicles, owned by multinational car manufacturer Jaguar Land Rover (JLR)”
# MAGIC
# MAGIC The `MultipleNegativesRankingLoss` and `CachedMultipleNegativesRankingLoss` only need to be supplied with anchor-positive pairs, and then it will create anchor-positive-negative triplets by using all other in-batch positive examples as negatives for each supplied anchor-positive.

# COMMAND ----------

column_remap = {"content" : "anchor",  "generated_question" : "positive"}

ft_train_dataset = Dataset.from_pandas(
    training_set.rename(columns=column_remap)
).select_columns(["anchor", "positive"])

ft_eval_dataset = Dataset.from_pandas(
    eval_set.rename(columns=column_remap)
).select_columns(["anchor", "positive"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 3: Define loss function

# COMMAND ----------

train_loss = CachedMultipleNegativesRankingLoss(
    model=model, 
    mini_batch_size=8
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 4: Specify Training Arguments (Optional)

# COMMAND ----------

num_epochs = 1

temp_checkpoint_dir = tempfile.TemporaryDirectory().name
dbutils.fs.mkdirs(temp_checkpoint_dir)

args = SentenceTransformerTrainingArguments(
    # Required parameters:
    output_dir=temp_checkpoint_dir, # Specify where outputs go
    # Optional training parameters:
    num_train_epochs=num_epochs, # How many full passes over the data should be done during training (epochs)?
    learning_rate=2e-5, # This takes trial and error, but 2e-5 is a good starting point.
    auto_find_batch_size=True, # Allow automatic determination of batch size
    warmup_ratio=1, # This takes trial and error
    seed=42, # Seed for reproducibility
    data_seed=42, # Seed for reproducibility
    # Optional tracking/debugging parameters:
    evaluation_strategy="steps",
    eval_steps=10, # How often evaluation loss should be logged
    logging_steps=10 # How often training loss should be calculated
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 5: Fine-tune embedding model
# MAGIC
# MAGIC Here, we define training loss to be Multiple Negatives Ranking Loss (MNRL). MNRL is useful for constrastive learning, where we identify similar vs. dissimilar pairs of examples. Refer to [docs here](https://www.sbert.net/docs/package_reference/losses.html#sentence_transformers.losses.CachedMultipleNegativesRankingLoss). 
# MAGIC

# COMMAND ----------

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,#ft_train_dataset,
    eval_dataset=val_dataset,#ft_eval_dataset,
    loss=train_loss,
    tokenizer=tokenizer

)

with mlflow.start_run() as run:
    trainer.train()

# COMMAND ----------

# MAGIC %md
# MAGIC # Serve the embedding model
# MAGIC
# MAGIC ## Log model 
# MAGIC Our fine-tuned `sentence-transformer` is ready. To use it as an embedding endpoint, make sure to include the `task` metadata. `metadata={"task": "llm/v1/embeddings", "model_type": "bge-small"}`

# COMMAND ----------

registered_ft_embedding_model_name = "bge_finetuned"
data = "Look at my finetuned model"

# Log the model to unity catalog
mlflow.set_registry_uri("databricks-uc")

signature = mlflow.models.infer_signature(
    model_input=data,
    model_output=trainer.model.encode(data).tolist(),
)

with mlflow.start_run() as run:
  # log the model to mlflow as a transformer with PT metadata
  _logged = mlflow.sentence_transformers.log_model(
      model=trainer.model,
      artifact_path="model",
      task="llm/v1/embeddings",
      registered_model_name=f"{CATALOG}.{USER_SCHEMA}.{registered_ft_embedding_model_name}",
      metadata={
        "model_type": "bge-large" if gpu_is_available else "bge-small" # Can be bge-small or bge-large
        },
      input_example=data
  )

# COMMAND ----------

def get_latest_model_version(model_name):
  client = MlflowClient()
  model_version_infos = client.search_model_versions(f"name = '{model_name}'")
  return max([int(model_version_info.version) for model_version_info in model_version_infos])

# If instructor needs to update the model, the schema needs to change to SHARED_SCHEMA
latest_model_version = get_latest_model_version(f"{CATALOG}.{USER_SCHEMA}.{registered_ft_embedding_model_name}")
latest_model_version

# COMMAND ----------

# MAGIC %md
# MAGIC The endpoint spin-up time could take ~15 mins

# COMMAND ----------

w = WorkspaceClient()

endpoint_name = f"adv_genai_{registered_ft_embedding_model_name}"

try:
    endpoint = w.serving_endpoints.get(endpoint_name)
    endpoint_exists = True
except:
    endpoint_exists = False

print(f"Endpoint exists: {endpoint_exists}")

if endpoint_exists and do_not_update_endpoint:
    print("Reusing existing endpoint...")
elif endpoint_exists and not do_not_update_endpoint:
    try:
        print(f"Updating endpoint to model version {latest_model_version}")
        print(w.serving_endpoints.update_config_and_wait(
            name=endpoint_name, 
            served_entities=[
                ServedEntityInput(
                    entity_name=f"{SHARED_CATALOG}.{SHARED_SCHEMA}.{registered_ft_embedding_model_name}",
                    entity_version=str(latest_model_version),
                    workload_type = "GPU_SMALL",
                    workload_size = "Small",
                    scale_to_zero_enabled = False
            )]
        ))
        print("Updating...")
    except:
        print("Update failed. Check your permissions and other ongoing tasks.")
else:
    try:
        print("Creating new serving endpoint...")
        print(w.serving_endpoints.create_and_wait(
            name=endpoint_name,
            config=EndpointCoreConfigInput(
                name=endpoint_name,
                served_entities=[
                    ServedEntityInput(
                        entity_name=f"{SHARED_CATALOG}.{SHARED_SCHEMA}.{registered_ft_embedding_model_name}",
                        entity_version=str(latest_model_version),
                        workload_type = "GPU_SMALL",
                        workload_size = "Small",
                        scale_to_zero_enabled = False
                    ),
                ]
            )
        ))
    except:
        print("Creation failed. Check your permissions and other ongoing tasks")

# COMMAND ----------

# MAGIC %md
# MAGIC # Set up Vector Search index

# COMMAND ----------

try:
    index = w.vector_search_indexes.get_index(f"{SHARED_CATALOG}.{SHARED_SCHEMA}.{VS_INDEX}")
    index_exists = True
    print("VS index already exists")
except:
    print(f"VS Index, {VS_INDEX}, did not already exist.")
    index_exists = False

if index_exists:
    print("Syncing existing index")
    w.vector_search_indexes.sync_index(f"{SHARED_CATALOG}.{SHARED_SCHEMA}.{VS_INDEX}") # If it exists, sync
else:
    print(f"Creating vector index: {CATALOG}.{USER_SCHEMA}.{VS_INDEX}")
    source_table = f"{CATALOG}.{USER_SCHEMA}.{GOLD_CHUNKS_FLAT}"
    _ = spark.sql(f"ALTER TABLE {source_table}  SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    w.vector_search_indexes.create_index(
        name=f"{CATALOG}.{USER_SCHEMA}.{VS_INDEX}",
        endpoint_name=VS_ENDPOINT,
        primary_key="uuid",
        index_type=VectorIndexType("DELTA_SYNC"),
        delta_sync_index_spec=DeltaSyncVectorIndexSpecResponse(
            embedding_source_columns=[
                EmbeddingSourceColumn(
                    name="content",
                    embedding_model_endpoint_name=endpoint_name
                )],
            pipeline_type=PipelineType("TRIGGERED"),
            source_table=source_table
                   )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC The following cell can take ~7 mins.

# COMMAND ----------

while w.vector_search_indexes.get_index(f"{SHARED_CATALOG}.{SHARED_SCHEMA}.{VS_INDEX}").status.ready == False:
    print(f"Waiting for vector search creation or sync to complete for {SHARED_CATALOG}.{SHARED_SCHEMA}.{VS_INDEX}")
    time.sleep(30) # Give it some time to finish

print(w.vector_search_indexes.get_index(f"{SHARED_CATALOG}.{SHARED_SCHEMA}.{VS_INDEX}").status)

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluate fine-tuned embedding model 
# MAGIC Wait for VS index to be created and then perform eval

# COMMAND ----------

def get_relevant_documents(question : str, index_name : str, k : int = 3, filters : str = None, max_retries : int = 3) -> List[dict]:
    response_received = False
    retries = 0
    while ((response_received == False) and (retries < max_retries)):
        try:
            docs = w.vector_search_indexes.query_index(
                index_name=index_name,
                columns=["uuid","content","category","filepath"],
                filters_json=filters,
                num_results=k,
                query_text=question
            )
            response_received = True
            docs_pd = pd.DataFrame(docs.result.data_array)
            docs_pd.columns = [_c.name for _c in docs.manifest.columns]
        except Exception as e:
            retries += 1
            time.sleep(1 * retries)
            print(e)
    return json.loads(docs_pd.to_json(orient="records"))

# COMMAND ----------

index_full_new = f"{SHARED_CATALOG}.{SHARED_SCHEMA}.{VS_INDEX}"
index_full_old = f"{SHARED_CATALOG}.{SHARED_SCHEMA}.spark_docs_gold_index"

def get_relevant_doc_ids(question : str, index_name : str) -> list[str]:
    docs = get_relevant_documents(question, index_name=index_name, k=10)
    return [_x["uuid"] for _x in docs]

# COMMAND ----------

eval_df = spark.table(f"{CATALOG}.{USER_SCHEMA}.{GENERATED_QUESTIONS}_eval")

eval_pd_new = eval_df.toPandas()
eval_pd_new["uuid"] = eval_pd_new["uuid"].transform(lambda x: [x])
eval_pd_new["retrieved_docs"] = eval_pd_new["generated_question"].transform(lambda x: get_relevant_doc_ids(x, index_full_new))

eval_pd_old = eval_df.toPandas()
eval_pd_old["uuid"] = eval_pd_old["uuid"].transform(lambda x: [x])
eval_pd_old["retrieved_docs"] = eval_pd_old["generated_question"].transform(lambda x: get_relevant_doc_ids(x, index_full_old))

display(eval_pd_new)

# COMMAND ----------

with mlflow.start_run() as run:
    eval_results_ft = mlflow.evaluate(
        data=eval_pd_new,
        model_type="retriever",
        targets="uuid",
        predictions="retrieved_docs",
        evaluators="default",
        extra_metrics=[mlflow.metrics.recall_at_k(i) for i in range(1,11,1)]
    )
    eval_results = mlflow.evaluate(
        data=eval_pd_old,
        model_type="retriever",
        targets="uuid",
        predictions="retrieved_docs",
        evaluators="default",
        extra_metrics=[mlflow.metrics.recall_at_k(i) for i in range(1,11,1)]
    )

# COMMAND ----------

plt.plot([eval_results_ft.metrics[f"recall_at_{i}/mean"] for i in range(1,11,1)], label="finetuned")
plt.plot([eval_results.metrics[f"recall_at_{i}/mean"] for i in range(1,11,1)], label="bge")
plt.title("Recall at k")
plt.xlabel("k")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Based on the plot above, we see that the fine-tuned model performs better overall!

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Compare answers retrieving from documents indexed using non-fine-tuned vs fine-tuned embedding model

# COMMAND ----------

example_question = "What function should if I have an array column and I want to make each item in the array a separate row?"

# COMMAND ----------

get_relevant_documents(example_question, index_full_old, k=2)

# COMMAND ----------

get_relevant_documents(example_question, index_full_new, k=2)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>