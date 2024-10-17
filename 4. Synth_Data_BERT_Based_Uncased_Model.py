# Databricks notebook source
# MAGIC %pip install mlflow==2.14.1 sentence-transformers==3.0.1 torch==2.3.0 transformers==4.40.1 databricks-sdk==0.31.1 tf-keras==2.16.0 accelerate==0.27.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------


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

VS_ENDPOINT = "vs_endpoint_synthetic_claims_data"

gpu_is_available = torch.cuda.is_available()

# COMMAND ----------

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
     
  
  return embeddings_chunks

# COMMAND ----------

def ConcatenateFields(test_claims):
    from pyspark.sql.functions import concat_ws, struct
    
    final_test_df = test_claims.withColumn("DOCUMENT", struct("OEM_CODE", "DEALER", "DISTR", "FC", "ENGINE_NAME_DESC", "FAILCODE", "SHOPORDERNUM",  "CLEANED_CORRECTION"))
    return final_test_df

# COMMAND ----------

def get_label(claim_id):
  label = spark.sql("select FAILURE_MODE_BUCKET from cmidev.cbu_ccims_poc.synth_data_claims_feature where CLAIM_ID_SEQ =  "+str(claim_id)+" ").toPandas()
  return (label['FAILURE_MODE_BUCKET'].iloc[0])


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
df_filtered = df_with_row_number.filter("rn <= 4")

# Group by FAILURE_MODE_BUCKET and count the records in each group
df_counts = df_filtered.groupBy("FAILURE_MODE_BUCKET").count()

# Filter to keep only FAILURE_MODE_BUCKET groups with exactly 20 records
df_exact_5 = df_counts.filter("count = 4")

# Join back to the filtered DataFrame to get the original records, but only for FAILURE_MODE_BUCKET with exactly 20 records
df_final = df_filtered.join(df_exact_5, "FAILURE_MODE_BUCKET").drop("count", "rn")


# COMMAND ----------

def CreateDatasetWithoutFailModeBucket(test_claims):
    final_test_df=ConcatenateFields(test_claims)
    final_test_df = final_test_df.toPandas()
    

    return final_test_df
df = CreateDatasetWithoutFailModeBucket(df_final)
df = df[["CLAIM_ID_SEQ","DOCUMENT","FAILURE_MODE_BUCKET"]]
df.rename(columns={"CLAIM_ID_SEQ":"claim_id",
              "DOCUMENT":"document",
              "FAILURE_MODE_BUCKET":"label"},inplace=True)



# COMMAND ----------

from sklearn.model_selection import train_test_split  

import numpy as np  
  
labels = df['label'].tolist()
document = df['document'].tolist()

X_train, X_test, y_train, y_test = train_test_split(document, labels, test_size=0.4, random_state=44, shuffle=True, stratify=labels)  

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

from transformers import AutoTokenizer
bert_model_path = "/Volumes/cmidev/default/preventech/DBX_hack/google-bertbert-base-uncased"
# Initialize the tokenizer (example with BERT)
tokenizer = AutoTokenizer.from_pretrained(bert_model_path)

# Function to tokenize and encode the text data in the dictionaries
def encode_texts(text_dicts, tokenizer):
    # Concatenate the values of the specified keys into a single string for each dictionary
    # Ensure all values are converted to strings before joining
    texts = [' '.join([str(text_dict[key]) for key in ['OEM_CODE', 'DEALER', 'DISTR', 'FC', 'ENGINE_NAME_DESC', 'FAILCODE', 'SHOPORDERNUM', 'CLEANED_CORRECTION']]) for text_dict in text_dicts]
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
# Tokenize and encode X_train and X_test
X_train_encoded = encode_texts(X_train, tokenizer)
X_test_encoded = encode_texts(X_test, tokenizer)

# COMMAND ----------

def decode_texts(encoded_texts, tokenizer):
    decoded_texts = [tokenizer.decode(encoded_text, skip_special_tokens=True) for encoded_text in encoded_texts]
    return decoded_texts

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
    num_train_epochs=80,              # Total number of training epochs
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

# Save the model
model.save_pretrained('/Volumes/cmidev/default/preventech/DBX_hack/ccims_bert_base_uncased_trained_model_synthetic_v2')

# COMMAND ----------

from transformers import BertForSequenceClassification
 
# Load the model
model = BertForSequenceClassification.from_pretrained('/Volumes/cmidev/default/preventech/DBX_hack/ccims_bert_base_uncased_trained_model_synthetic_v2')
 
 
from transformers import BertTokenizer
 
# Assuming 'model' is your BertForSequenceClassification model
# and 'tokenizer' is an instance of BertTokenizer or similar compatible with your model
 
input_text = "SRT190CQ overlap with 190CLCheck engine light on, Connected to Cummins and inspected codes, Fault code active for turbo speed sensor, Opened eds to troubleshoot, Inspected if any other fault code was triggering turbo speed code, Inspected sensor supply voltage, voltage was in spec, Inspect intake manifold pressure sensor for carbon build up, removed sensor to inspect, no build up found, Inspect for signal wire open circuit, Removed harness from ECM and ohmed signal to signal from sensor connector to ecm connector, That was less than 10 ohms, Inspected return for a open circuit, ohmed from sensor connector to ecm connector and that was also less than 10 ohms, Checked for turbo sensor shorted to ground, It was greater than 100k ohms, Inspected signal pin and ohmed out signal pin at ecm to inspect for a pin to pin short, All other pins on ecm connecter are not shorted to signal pin for turbo speed sensor connector, Checked ecm calibration history, Update is available but not relavent to active code, Advised replace turbo speed senor, Speed sensor ordered, Replaced speed sensor"
 
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
label_encoder.inverse_transform(predictions)

# COMMAND ----------

model

# COMMAND ----------

import torch
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

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Also move the inputs to the same device as the model
inputs = {k: v.to(device) for k, v in inputs.items()}

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

claims_train = df_final.select('CLAIM_ID_SEQ').rdd.flatMap(lambda x: x).collect()
claims_str = ",".join([str(claim) for claim in claims_train])

query = f"select * from cmidev.cbu_ccims_poc.synth_data_claims_feature where FAILURE_MODE_BUCKET != 'Not Inspected' AND CLAIM_ID_SEQ not in ({claims_str}) limit 24"

test_claims = spark.sql(query)

final_test_df = test_claims

docs = final_test_df.select('DOCUMENT').rdd.flatMap(lambda x: x).collect()
test_claims_list = [str(doc).lower() for doc in docs]
final_test_df_pd = final_test_df.toPandas()
claim_id = final_test_df_pd["CLAIM_ID_SEQ"].tolist()
actual_label = final_test_df_pd['FAILURE_MODE_BUCKET'].tolist()
predicted_label_fh = []

for input_text, claim_no, actual_l in zip(test_claims_list, claim_id, actual_label):
    print("claim_no:", claim_no)
    try:
        inputs = tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs.logits

        predictions = torch.argmax(logits, dim=-1)
        label_fh = label_encoder.inverse_transform(predictions.cpu().numpy())
        predicted_label_fh.append(label_fh[0])
        print("processing claim id:", claim_no, "----actual label:", actual_l, "----Prediction:", label_fh[0])
    except Exception as e:
        print(f"Error processing claim id {claim_no}: {str(e)}")

# COMMAND ----------

import pandas as pd
result_df = pd.DataFrame()
result_df['claim_id'] = claim_id
result_df['actual_label'] = actual_label
result_df['predicted_label_few'] = predicted_label_fh

# display(result_df)

 

# COMMAND ----------

#Calculate accuracy pretrained on cleaned correction
correct=0
for actual, predicted in zip(actual_label,predicted_label_fh):
    if actual==predicted:
        correct+=1
print("Accuracy of pretrained model on cleaned correction% =",correct*100/len(actual_label))