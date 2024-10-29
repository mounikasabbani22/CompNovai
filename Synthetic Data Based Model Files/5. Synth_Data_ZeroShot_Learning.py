# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Create a "Self-Managed" Vector Search Index
# MAGIC
# MAGIC In the previous demo, we chunked the raw PDF document pages into small sections, computed the embeddings, and saved it as a Delta Lake table. Our dataset is now ready. 
# MAGIC
# MAGIC Next, we'll configure Databricks Vector Search to ingest data from this table.
# MAGIC
# MAGIC Vector search index uses a Vector search endpoint to serve the embeddings (you can think about it as your Vector Search API endpoint). <br/>
# MAGIC Multiple Indexes can use the same endpoint. Let's start by creating one.
# MAGIC
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to;*
# MAGIC
# MAGIC * Set up an endpoint for Vector Search.
# MAGIC
# MAGIC * Store the embeddings and their metadata using the Vector Search.
# MAGIC
# MAGIC * Inspect the Vector Search endpoint and index using the UI. 
# MAGIC
# MAGIC * Retrieve documents from the vector store using similarity search.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12 14.3.x-scala2.12**
# MAGIC
# MAGIC
# MAGIC **🚨 Important:** This demonstration relies on the resources established in the previous one. Please ensure you have completed the prior demonstration before starting this one.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Install required libraries.

# COMMAND ----------

# MAGIC %pip install -U --quiet mlflow==2.14.3 databricks-vectorsearch==0.40 transformers==4.43.3 "unstructured[pdf,docx]==0.14.10" langchain==0.2.11 langchain-community==0.2.10 pydantic==2.8.2 flashrank==0.2.8 pyspark==3.1.2 PyMuPDF accelerate
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ./_helper_functions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a "Self-Managed" Vector Search Index
# MAGIC
# MAGIC Setting up a Databricks Vector Search index involves a few key steps. First, you need to decide on the method of providing vector embeddings. Databricks supports three options: 
# MAGIC
# MAGIC * providing a source Delta table containing text data
# MAGIC * **providing a source Delta table that contains pre-calculated embeddings**
# MAGIC * using the Direct Vector API to create an index on embeddings stored in a Delta table
# MAGIC
# MAGIC In this demo, we will go with the first method. 
# MAGIC
# MAGIC Next, we will **create a vector search endpoint**. And in the final step, we will **create a vector search index** from a Delta table. 
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup a Vector Search Endpoint
# MAGIC
# MAGIC The first step for creating a Vector Search index is to create a compute endpoint. This endpoint serves the vector search index. You can query and update the endpoint using the REST API or the SDK. 
# MAGIC
# MAGIC **🚨IMPORTANT: Vector Search endpoints must be created before running the rest of the demo. Endpoint names should be in this format; `vs_endpoint_x`. The endpoint will be assigned by username.**
# MAGIC
# MAGIC **💡 Note: In the code block below, `create_endpoint` function is commented out. If you have right permissions, you can create an endpoint using that function.**

# COMMAND ----------

# assign vs search endpoint by username
vs_endpoint_prefix = "vs_endpoint_synthetic"
vs_endpoint_fallback = "vs_endpoint_synthetic_claims_data"

vs_endpoint_name = vs_endpoint_fallback #try this
print(f"Vector Endpoint name: {vs_endpoint_name}. In case of any issues, replace variable `vs_endpoint_name` with `vs_endpoint_fallback` in demos and labs.")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

vsc = VectorSearchClient(disable_notice=True)

# COMMAND ----------

if vs_endpoint_name not in [e["name"] for e in vsc.list_endpoints()["endpoints"]]:
    vsc.create_endpoint(name=vs_endpoint_name, endpoint_type="STANDARD")

# COMMAND ----------

vs_endpoint_name

# COMMAND ----------

vs_index_fullname= f"cmidev.cbu_ccims_poc.synthetic_claim_text_embeddings_index"

# COMMAND ----------

# MAGIC %md
# MAGIC ### View the Endpoint
# MAGIC
# MAGIC After the endpoint is created, you can view your endpoint on the [Vector Search Endpoints UI](#/setting/clusters/vector-search). Click on the endpoint name to see all indexes that are served by the endpoint.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Connect Delta Table with Vector Search Endpoint
# MAGIC
# MAGIC After creating the endpoint, we can create the **vector search index**. The vector search index is created from a Delta table and is optimized to provide real-time approximate nearest neighbor searches. The goal of the search is to identify documents that are similar to the query. 
# MAGIC
# MAGIC **Vector search indexes appear in and are governed by Unity Catalog.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Search for Similar Content
# MAGIC
# MAGIC That's all we have to do. Databricks will automatically capture and synchronize new entries in your Delta Lake Table.
# MAGIC
# MAGIC Note that depending on your dataset size and model size, index creation can take a few seconds to start and index your embeddings.
# MAGIC
# MAGIC Let's give it a try and search for similar content.
# MAGIC
# MAGIC **📌 Note:** `similarity_search` also supports a filter parameter. This is useful to add a security layer to your RAG system: you can filter out some sensitive content based on who is doing the call (for example filter on a specific department based on the user preference).
# MAGIC

# COMMAND ----------

#Test case 1 with known failure mode bucket
claim='"1608.0","27620","1693","687.0","X15 3 2017","ELTS","1033900"," MESSAGE:Service Support Center Cummins Inc, Receiving Location 35, 910 S Marr Road, Columbus, IN 47201 ****IMPORTANT**EPR**PART REQUEST - Review CUMMINS MATERIAL RETURN TICKET/s for specific part numbers to be returned **EPR Core Part***Goto QSOL Warranty Click on Request Core Return Process** ***Enhanced Parts Return Warning*** UPS: Visit https://row,ups,com/Default,aspx?Company=cumminsssc&LoginId=SSCship&Password=SSCpart1 to print your return label, FEDEX: 332426788 Important: Always print and include a copy of the parts return tag inside the box,FOLLOWING PARTS TO BE RETURNED:535130700 HOOK UP COMPUTER TO CHECKED CODES FINDING NONE ACTIVE STARTED TROUBLESHOOTING 687 AS IT HAD 7 COUNTS WITHIN THE LAST 25 HOURS , OPEN EDS AND FOLLOW PROMPTS 1, CHECKED FOR OTHER CODES FINDING NONE OF THE LISTED CODES PRESENT 2, REMOVED TURBO SPEED SENSOR AND INSPECTED FINDING NO DAMAGE 3, FOUND A GOOD 5V SIGNAL AT THE TURBO SPEED SENSOR HARNESS4, REMOVED INTAKE TEMP SENSOR FINDING NO DAMAGE BUT IT DID HAVE SOOT ACCUMULATION SO I CLEANED IT 5, REMOVED ECM CONNECTOR AND TURBO SPEED CONNECTOR AND CHECKED PINS TO THE ECM FINDING ALL WHERE AT 0,3OHMS IN SPEC 6, CHECKED GROUND TO ENGINE BLOCK AND FOUND IT WAS IN SPEC AT OL7, CHECKED PIN CONTINUITY WITH ALL OTHER PINS FINDING NO CONTINUITY WITH ANY OTHER PINS 8, CHECKED CALIBRATION AND FOUND THERE ARE MULTIPLE NEWER CALIBRATIONS AVAILABLE BUT NONE RELATED TO THESE CODES 9, EDS LEADS ME TO REPLACE THE TURBO SPEED SENSOR REPLACED SENSOR AND WENT FOR A ROAD TEST WITH NO CODES COMING BACK ALL GOOD AT THIS TIME EPR Part Number:535130700 Tracking Number:1z4r54w6880002738,CHECK ENGINE LIGHT YELLOWTurbocharger Speed SensorINDETERMINATE1084534 MONDAY 17JUN2019 11:27:31 AM"'

# COMMAND ----------

#Test case 2
claim = '{"OEM_CODE":1608,"DEALER":"99876","DISTR":"1692","FC":"2147.0,1897.0","ENGINE_NAME_DESC":"X15 3 2017","FAILCODE":"THAC","SHOPORDERNUM":"1473300","CLEANED_CORRECTION":" 6262020 6:23:34 AM 3025 customer called that cel is back on drove to cust locationBRENNER OIL CO 2147 3RD STNILES, MI 49120due to cel comming back on when truck put under load, hooked up computer and pulled codes again, active code again for vgt, fc 1897 this time, performed ts and found actuator failed, drained coolant and removed old actuator, insp turbo, good, installed and calabrated new one, refilled coolant, cleared all fault codes,performed regen, all temps and pressures look good, power washed engine road tested unit and no faults returned, road tested to confirm that cel didnt come on since this is the second trip out to customer for cel issue,CHECK ENGINE LIGHT YELLOWTURBOCHARGER ACTUATORFAULT LAMP23451DANKO MONDAY 29JUN2020 07:59:36 AM"}'

# COMMAND ----------

#Test case 3
#Failmode bucket: TSS - INTERMITTENT SIGNAL
claim_context='{"OEM_CODE": 1522, "DEALER": "25054", "DISTR" : "1758", "FC" : "345.0,4631.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "THAC", "SHOPORDERNUM": "1032965", "FAILURE_MODE_BUCKET": "TSS - ADJUSTED/DAMAGED BY CUSTOMER", "CLEANED_CORRECTION": "Replaced faulty throttle actuator due to TSS adjusted/damaged by customer. Performed road test post-repair."}'

# COMMAND ----------

#Test case 4
#ACT(E) LOOSE APS MAGNET
claim='{"OEM_CODE":2758,"DEALER":"19958","DISTR":"1693","FC":"1897.0","ENGINE_NAME_DESC":"B6.7 2017","FAILCODE":"THAC","SHOPORDERNUM":"5998900","FAILURE_MODE_BUCKET":"ACT(E) LOOSE APS MAGNET","CLEANED_CORRECTION":" drain and fill coolant replaced vgt actuator and calibrated, send on road test and run regen, EPR Part Number:549487800 Tracking Number:775883091309,CHECK ENGINE LIGHT YELLOWTurbocharger ActuatorDATA LINK19958BBAKER SUNDAY 04AUG2019 04:09:58 PM"}'

# COMMAND ----------

#Zero Shot Learning
import mlflow
from mlflow.deployments import get_deploy_client

def Predict_ZeroShotLearning(question):
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    # question = "What is predicted 'failure mode bucket' having below claim details:"+str(claim) +"Show probability of each failure mode bucket"

    data = {
        "prompt": (
            "You are a diesel engine service assistant, who is expert in the turbochargers. You will be given a warranty claim text, along with other key details of the engine.  Here is an example of such claim:"+claim_context+" For this claim, the FAILURE MODE BUCKET is 'TSS - ADJUSTED/DAMAGED BY CUSTOMER'.By reading those details, you need to predict failure mode bucket of the turbocharger."
        ),
        "temperature": 0.5,
        "max_tokens": 1000,
        "n": 1,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.2,
    }
    
    response = deploy_client.predict(endpoint="databricks-gte-large-en", inputs={"input": [question]})
    embeddings = [e["embedding"] for e in response.data]
    print(embeddings)
    
    return embeddings

# COMMAND ----------

claim_contexts= spark.sql("select DOCUMENT from cmidev.cbu_ccims_poc.synth_data_claims_feature where FAILURE_MODE_BUCKET != 'Not Inspected'").toPandas()


# COMMAND ----------

display(claim_contexts)

# COMMAND ----------

def GetSimilarDocuments(embeddings):
  # get similar 5 documents.
  results = vsc.get_index(vs_endpoint_name, vs_index_fullname).similarity_search(
    query_vector=embeddings[0],
    columns=["CLAIM_ID_SEQ", "DOCUMENT_CHUNK"],
    num_results=10)

  # format result to align with reranker lib format. 
  passages = []
  similar_claim_id=[]
  claim_proba=[]
  documents=[]
  # print("results:",results)
  for doc in results.get("result", {}).get("data_array", []):
      # print(doc)
      new_doc = {"CLAIM_ID_SEQ": doc[0], "text": doc[1], "probability": doc[2]}
      similar_claim_id.append(doc[0])
      claim_proba.append(doc[2])
      passages.append(new_doc)
      documents.append(doc[1].lower())

  # pprint(passages)
  # print("Similar claims are:",similar_claim_id)
  # print("Similar claims probabilty:",claim_proba)
  import pandas as pd
  df_sim_claims= pd.DataFrame()
  df_sim_claims['Similar_Claim_ID']=similar_claim_id
  df_sim_claims['Similar_Claim_Proba']=claim_proba
  df_sim_claims['Similar_Claim_Details']=documents
  df_sim_claims= df_sim_claims.sort_values(by='Similar_Claim_Proba', ascending=False)
  return df_sim_claims,documents,similar_claim_id
# display(df_sim_claims)

# COMMAND ----------

def GetFailureModeBucket(similar_claim_id):
    # Assuming similar_claim_id is a list of IDs
    formatted_ids = ",".join([f"'{id}'" for id in similar_claim_id])
    print("Formatted_IDS", formatted_ids)
    query = f"""
    SELECT distinct(CLAIM_ID_SEQ), CLEANED_CORRECTION, FAILURE_MODE_BUCKET 
    FROM cmidev.cbu_ccims_poc.synth_data_claims_feature
    WHERE CLAIM_ID_SEQ IN ({formatted_ids})
    """
    claims_with_failmode = spark.sql(query)
    # display(claims_with_failmode)
    return claims_with_failmode#['FAILURE_MODE_BUCKET'].iloc[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Re-ranking Search Results
# MAGIC
# MAGIC For re-ranking the results, we will use a very light library. [**`flashrank`**](https://github.com/PrithivirajDamodaran/FlashRank) is an open-source reranking library based on SoTA cross-encoders. The library supports multiple models, and in this example we will use `rank-T5` model. 
# MAGIC
# MAGIC After re-ranking you can review the results to check if the order of the results has changed. 
# MAGIC
# MAGIC **💡Note:** Re-ranking order varies based on the model used!

# COMMAND ----------

bert_model_path = "/Volumes/cmidev/default/preventech/DBX_hack/google-bertbert-base-uncased"
from transformers import AutoTokenizer, AutoModelForCausalLM, Pipeline, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained(bert_model_path, local_files_only=True, revision="no_timm")

# COMMAND ----------

bert_model_path = "/Volumes/cmidev/default/preventech/DBX_hack/google-bertbert-base-uncased"
from transformers import AutoTokenizer, AutoModelForCausalLM, Pipeline, AutoModelForSeq2SeqLM


from transformers import AutoTokenizer, AutoModelForQuestionAnswering
def GetSortedDocuments(question,documents):
    tokenizer = AutoTokenizer.from_pretrained(bert_model_path, local_files_only=True, revision="no_timm")
    model = AutoModelForQuestionAnswering.from_pretrained(bert_model_path, local_files_only=True, revision="no_timm")



    # Format the input for the tokenizer correctly
    # Combine question and each document into a single string
    inputs = tokenizer(
        [question + " " + doc for doc in documents],  # Combine question and document
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    outputs = model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    import torch
    # Calculate the scores for each document:
    scores = torch.sum(start_scores * end_scores, dim=1)

    # Convert the scores to a 1D integer tensor:
    scores = scores.type(torch.IntTensor)

    # Sort the documents in descending order based on their scores:
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_documents = [documents[i] for i in sorted_indices]

    # Print the sorted documents:
    # print(sorted_documents)
    return sorted_documents

# COMMAND ----------

def GetSimilarDocumentDetails(similar_claim_id,df_sim_claims,sorted_documents):
    get_index = []
    similar_claim_id_new = []
    for document in sorted_documents:
        index = df_sim_claims[df_sim_claims['Similar_Claim_Details'] == document].index.tolist()
        # print(index)
        # for j in index:
        #     if j not in get_index:
        get_index.extend(index)
    # print("GET_INDEX", get_index)

    for id in similar_claim_id:
        id = int(id)
        similar_claim_id_new.append(id)

    claims=[]
    claims = ([similar_claim_id_new[i] for i in get_index])
    best_matching_doc_fail_mode = GetFailureModeBucket(claims)
    return best_matching_doc_fail_mode

# COMMAND ----------

#Approach with max count of fail modes

import numpy as np
from pyspark.sql.functions import col
def MostPopulatedFailModeBucket(best_matching_doc_fail_mode):
    fail_mode_counts = best_matching_doc_fail_mode.groupBy("FAILURE_MODE_BUCKET").count()
    max_count_fail_mode = fail_mode_counts.orderBy(col("count").desc()).first()["FAILURE_MODE_BUCKET"]
    # print(max_count_fail_mode)
    return max_count_fail_mode

# COMMAND ----------

#Approach with top most matching document's fail mode
def TopMostMatchingFailModeBucket(best_matching_doc_fail_mode):

    fail_mode_counts = best_matching_doc_fail_mode.select("FAILURE_MODE_BUCKET").collect()[0]
    # print(fail_mode_counts)
    return fail_mode_counts

# COMMAND ----------

# label_zh = (TopMostMatchingFailModeBucket(best_matching_doc_fail_mode_zh))

# COMMAND ----------

def ConcatenateFields(test_claims):
    from pyspark.sql.functions import concat_ws, struct
    
    final_test_df = test_claims.withColumn("DOCUMENT", struct("OEM_CODE", "DEALER", "DISTR", "FC", "ENGINE_NAME_DESC", "FAILCODE", "SHOPORDERNUM",  "CLEANED_CORRECTION"))
 
    return final_test_df

# COMMAND ----------

test_claims = spark.sql("select * from cmidev.cbu_ccims_poc.synth_data_claims_feature")
final_test_df=ConcatenateFields(test_claims)

# COMMAND ----------

pip install mlflow[databricks]

# COMMAND ----------

#Put everything together to call required functions on multiple claims
test_claims = spark.sql("select * from cmidev.cbu_ccims_poc.synth_data_claims_feature")
final_test_df=ConcatenateFields(test_claims)
predicted_label_zh=[]
predicted_label_fh=[]
predicted_label_oh=[]

test_claims_list = final_test_df.select("DOCUMENT").collect()

claim_id = [row.CLAIM_ID_SEQ for row in test_claims.select("CLAIM_ID_SEQ").collect()]
actual_label = [row.FAILURE_MODE_BUCKET for row in final_test_df.select("FAILURE_MODE_BUCKET").collect()]

print(test_claims_list)
for test_claim, claim_no, actual_l in zip(test_claims_list,claim_id,actual_label):

    document_text = str(test_claim.DOCUMENT)
    question = "What is predicted 'failure mode bucket' having below claim details: " + document_text + " Show probability of each failure mode bucket"
    print(document_text)

   
    embeddings_zeroshot = Predict_ZeroShotLearning(question)

    df_sim_claims_zh,documents_zh,similar_claim_id_zh = GetSimilarDocuments(embeddings_zeroshot)
   
    # Zero Shot Learning
    sorted_documents_zh=GetSortedDocuments(question,documents_zh)
    best_matching_doc_fail_mode_zh=GetSimilarDocumentDetails(similar_claim_id_zh,df_sim_claims_zh,sorted_documents_zh)
    label_zh = (TopMostMatchingFailModeBucket(best_matching_doc_fail_mode_zh))
    predicted_label_zh.append(label_zh[0])
    print("Zero Shot Learning ### claim id:", claim_no, "Actual Label:",actual_l, "predicted Label:",label_zh )



# COMMAND ----------


import pandas as pd
result_df = pd.DataFrame()
result_df['claim_id'] = claim_id
result_df['actual_label'] = actual_label
result_df['predicted_label_zero'] = predicted_label_zh

display(result_df)

# COMMAND ----------

zeroshot_learning = result_df.to_csv("/Volumes/cmidev/default/preventech/DBX_hack/synth_data_zeroshot_learning.csv")
zeroshot_learning

# COMMAND ----------

#Calculate accuracy zero shot
correct=0
for actual, predicted in zip(actual_label,predicted_label_zh):
    if actual==predicted:
        correct+=1
print("Accuracy of Zero Shot% =",correct*100/len(actual_label))
