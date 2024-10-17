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

# %run ../Includes/Classroom-Setup-03
# %run ../_helper_functions


# COMMAND ----------

# MAGIC %run ./_helper_functions

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this demo, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

# print(f"Username:          {DA.username}")
# print(f"Catalog Name:      {DA.catalog_name}")
# print(f"Schema Name:       {DA.schema_name}")
# print(f"Working Directory: {DA.paths.working_dir}")
# print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo Overview
# MAGIC
# MAGIC As seen in the diagram below, in this demo we will focus on the Vector Search indexing section (highlighted in orange).  
# MAGIC
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/genai/genai-as-01-rag-pdf-self-managed-3.png" width="100%">
# MAGIC
# MAGIC
# MAGIC

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
# vs_endpoint_name = vs_endpoint_prefix+str(get_fixed_integer(DA.unique_name("_")))
vs_endpoint_name = vs_endpoint_fallback #try this
print(f"Vector Endpoint name: {vs_endpoint_name}. In case of any issues, replace variable `vs_endpoint_name` with `vs_endpoint_fallback` in demos and labs.")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

vsc = VectorSearchClient(disable_notice=True)

# COMMAND ----------

# IF YOU HAVE ENDPOINT CREATION PERMISSIONS, UNCOMMENT THIS CODE AND RUN IT TO CREATE AN ENDPOINT
# Define the endpoint name
# VECTOR_SEARCH_ENDPOINT_NAME = "swapnali_vector_search_endpoint" #swapnali added
if vs_endpoint_name not in [e["name"] for e in vsc.list_endpoints()["endpoints"]]:
    vsc.create_endpoint(name=vs_endpoint_name, endpoint_type="STANDARD")

# COMMAND ----------

vs_endpoint_name

# COMMAND ----------

# # check the status of the endpoint
# wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name)
# # wait_for_vs_endpoint_to_be_ready(vsc, "vs_endpoint_9") #swapnali
# print(f"Endpoint named {vs_endpoint_name} is ready.")

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

# # the table we'd like to index
# # source_table_fullname = f"{DA.catalog_name}.{DA.schema_name}.pdf_text_embeddings"
# #Swapnali added below
# # source_table_fullname = f"{DA.catalog_name}.{DA.schema_name}.pdf_text_embeddings_lab"
# source_table_fullname = f"cmidev.cbu_ccims_poc.synthetic_data_claim_text_embeddings_gte"

# # where we want to store our index
# # vs_index_fullname = f"{DA.catalog_name}.{DA.schema_name}.pdf_text_self_managed_vs_index_lab"
# vs_index_fullname= f"cmidev.cbu_ccims_poc.synthetic_claim_text_embeddings_index"
# # create or sync the index
# if not index_exists(vsc, vs_endpoint_name, vs_index_fullname):
#   print(f"Creating index {vs_index_fullname} on endpoint {vs_endpoint_name}...")
#   vsc.create_delta_sync_index(
#     endpoint_name=vs_endpoint_name,
#     index_name=vs_index_fullname,
#     source_table_name=source_table_fullname,
#     pipeline_type="TRIGGERED", #Sync needs to be manually triggered
#     primary_key="id",
#     embedding_dimension=1024, #Match your model embedding size (gte)
#     embedding_vector_column="embedding"
#   )
# else:
#   # trigger a sync to update our vs content with the new data saved in the table
#   vsc.get_index(vs_endpoint_name, vs_index_fullname).sync()

# # let's wait for the index to be ready and all our embeddings to be created and indexed
# wait_for_index_to_be_ready(vsc, vs_endpoint_name, vs_index_fullname)

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
    
    
#     response = deploy_client.predict(  
#     endpoint="databricks-gte-large-en",  
#     inputs={"input":  [  
#         "You are a diesel engine service assistant, who is expert in the turbochargers. You will be given a warranty claim text, along with other key details of the engine. By reading those details, you need to predict failure mode bucket of the turbocharger.",    
#         question 
#       ]}  
# )  

    response = deploy_client.predict(endpoint="databricks-gte-large-en", inputs={"input": [question]})
    embeddings = [e["embedding"] for e in response.data]
    print(embeddings)
    # response = deploy_client.predict(endpoint="databricks-gte-large-en", inputs={"input": [question, data]})  
    # embeddings = [e["embedding"] for e in response.data] 
    # print(embeddings)
    #----
    # Define the inputs for the model  
    # inputs = {  
    # "input": [  
    #     "You are a diesel engine service assistant, who is expert in the turbochargers. You will be given a warranty claim text, along with other key details of the engine. By reading those details, you need to predict failure mode bucket of the turbocharger.",   vf
    #     question
    # ]  
    # }  
    # # Make a prediction using the deployed model  
    # response = deploy_client.predict(endpoint="databricks-gte-large-en", inputs=inputs)  
  
    # # Extract the embeddings from the response  
    # embeddings = [e["embedding"] for e in response.data]  
    #----
    return embeddings

# COMMAND ----------

claim_contexts= spark.sql("select DOCUMENT from cmidev.cbu_ccims_poc.synth_data_claims_feature where FAILURE_MODE_BUCKET != 'Not Inspected'").toPandas()
# claim_contexts= spark.sql("select * from cmidev.cbu_ccims_poc.ccims_claim_inspection_feature where FAILURE_MODE_BUCKET != 'Not Inspected' limit 200").toPandas()

# COMMAND ----------

display(claim_contexts)

# COMMAND ----------

# context_claims_list_temp = claim_contexts["DOCUMENT"].tolist()
# len(context_claims_list_temp)

# COMMAND ----------

# import mlflow.deployments

# deploy_client = mlflow.deployments.get_deploy_client("databricks")

# question = "What is predicted 'failure mode bucket' having below claim details:"+str(claim) +"Show probability of each failure mode bucket"
# response = deploy_client.predict(endpoint="databricks-gte-large-en", inputs={"input": [question]})
# embeddings = [e["embedding"] for e in response.data]
# print(embeddings)

# COMMAND ----------

#Hugging Face Access Token: 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
#Token Name: hug_access_token_gf591
# Created by : Swapnali on 9/24/24

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

# embeddings = Predict(claim)
# df_sim_claims,documents = GetSimilarDocuments(embeddings)
# df_sim_claims= df_sim_claims.sort_values(by='Similar_Claim_Proba', ascending=False)
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

# sorted DOCS ['replaced turbocharger assembly due to c26 failure. performed stationary regen and road test.', 'replaced faulty turbo speed sensor due to oil contamination. performed stationary regen and road test.', 'replaced faulty turbo speed sensor due to oil contamination. performed stationary regen and road test.', 'replaced turbo speed sensor after c26 failure. performed stationary regen and road test.', 'replaced faulty turbo speed sensor due to tss weld process defect. performed stationary regen and road test.', 'replaced faulty turbo speed sensor due to tss weld process defect. performed stationary regen and road test.', 'waiting for supplier analysis of faulty turbo speed sensor due to act(e) supplier analysis not conducted. performed stationary regen and road test.', 'replaced faulty turbo speed sensor due to act(e) supplier manufacturing fault. performed stationary regen and road test.', 'waiting for supplier analysis of faulty turbo speed sensor due to act(e) waiting supplier analysis. performed stationary regen and road test.', 'waiting for supplier analysis of faulty turbo speed sensor due to act(e) waiting supplier analysis. performed stationary regen and road test.']
# similar_claim_id_zh [17695654.0, 17695642.0, 17695633.0, 17695625.0, 17695637.0, 17695645.0, 17695581.0, 17695595.0, 17695597.0, 17695585.0]
# df_sim_claims_zh    Similar_Claim_ID  ...                              Similar_Claim_Details
# 0        17695654.0  ...  waiting for supplier analysis of faulty turbo ...
# 1        17695642.0  ...  replaced faulty turbo speed sensor due to act(...
# 2        17695633.0  ...  replaced faulty turbo speed sensor due to oil ...
# 3        17695625.0  ...  replaced faulty turbo speed sensor due to oil ...
# 4        17695637.0  ...  waiting for supplier analysis of faulty turbo ...
# 5        17695645.0  ...  waiting for supplier analysis of faulty turbo ...
# 6        17695581.0  ...  replaced turbo speed sensor after c26 failure....
# 7        17695595.0  ...  replaced faulty turbo speed sensor due to tss ...
# 8        17695597.0  ...  replaced faulty turbo speed sensor due to tss ...
# 9        17695585.0  ...  replaced turbocharger assembly due to c26 fail...

# [10 rows x 3 columns]


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

# get_index

# COMMAND ----------

# print(get_index)


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
    
    # df_final_updated = test_claims.withColumn("FC", concat_ws(",", "FC_1","FC_2","FC_3","FC_4","FC_5"))
    final_test_df = test_claims.withColumn("DOCUMENT", struct("OEM_CODE", "DEALER", "DISTR", "FC", "ENGINE_NAME_DESC", "FAILCODE", "SHOPORDERNUM",  "CLEANED_CORRECTION"))
    # display("-----",type(final_test_df))
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
# claim_id = test_claims.select("CLAIM_ID_SEQ").collect()
# actual_label = final_test_df.select("FAILURE_MODE_BUCKET").collect()
claim_id = [row.CLAIM_ID_SEQ for row in test_claims.select("CLAIM_ID_SEQ").collect()]
actual_label = [row.FAILURE_MODE_BUCKET for row in final_test_df.select("FAILURE_MODE_BUCKET").collect()]

print(test_claims_list)
for test_claim, claim_no, actual_l in zip(test_claims_list,claim_id,actual_label):
    # question = "What is predicted 'failure mode bucket' having below claim details: "+str(test_claim) +" Show probability of each failure mode bucket"
    # print(test_claim)
    # document_text = str(test_claim.DOCUMENT)
    # document_dict = test_claim.DOCUMENT.asDict()

    # Concatenate all values within the DOCUMENT struct into a single string
    # document_text = ", ".join([str(value) for value in document_dict.values()])

    # print(type(document_text))

    # question = "What is predicted 'failure mode bucket' having below claim details: " + test_claim + " Show probability of each failure mode bucket"
    document_text = str(test_claim.DOCUMENT)
    question = "What is predicted 'failure mode bucket' having below claim details: " + document_text + " Show probability of each failure mode bucket"
    print(document_text)

   
    embeddings_zeroshot = Predict_ZeroShotLearning(question)
    # embeddings_fewshot= Predict_FewShotLearning(question)
    # embeddings_oneshot= Predict_OneShotLearning(question)
    # embeddings = Predict_FewShotLearning(test_claim,question)

    df_sim_claims_zh,documents_zh,similar_claim_id_zh = GetSimilarDocuments(embeddings_zeroshot)
    # df_sim_claims_fh,documents_fh,similar_claim_id_fh = GetSimilarDocuments(embeddings_fewshot)
    # df_sim_claims_oh,documents_oh,similar_claim_id_oh = GetSimilarDocuments(embeddings_oneshot)

    # Zero Shot Learning
    # claims_with_failmode =GetFailureModeBucket(similar_claim_id)
    sorted_documents_zh=GetSortedDocuments(question,documents_zh)
    # print("sorted DOCS", sorted_documents_zh)
    # print("similar_claim_id_zh", similar_claim_id_zh)
    # print("df_sim_claims_zh", df_sim_claims_zh)
    # print("sorted_documents_zh", sorted_documents_zh)
    best_matching_doc_fail_mode_zh=GetSimilarDocumentDetails(similar_claim_id_zh,df_sim_claims_zh,sorted_documents_zh)
    # display(best_matching_doc_fail_mode_zh)
    label_zh = (TopMostMatchingFailModeBucket(best_matching_doc_fail_mode_zh))
    
    # predicted_label.append(list(label[0].asDict().values())[0])
    predicted_label_zh.append(label_zh[0])
    print("Zero Shot Learning ### claim id:", claim_no, "Actual Label:",actual_l, "predicted Label:",label_zh )



# COMMAND ----------

# label_fh

# COMMAND ----------

# predicted_label_fh

# COMMAND ----------


import pandas as pd
result_df = pd.DataFrame()
result_df['claim_id'] = claim_id
result_df['actual_label'] = actual_label
result_df['predicted_label_zero'] = predicted_label_zh
# result_df['predicted_label_one'] = predicted_label_oh
# result_df['predicted_label_few'] = predicted_label_fh

display(result_df)

# COMMAND ----------

#Calculate accuracy zero shot
correct=0
for actual, predicted in zip(actual_label,predicted_label_zh):
    if actual==predicted:
        correct+=1
print("Accuracy of Zero Shot% =",correct*100/len(actual_label))

# COMMAND ----------

# from flashrank import Ranker, RerankRequest

# # Ensure the model file exists at this path or update the path accordingly
# # ranker_path ="dbfs://FileStore/rank-T5-flan"
# # ranker_path = "dbfs://FileStore/FlashRank_main.zip"
# cache_dir = f"{DA.paths.working_dir.replace('dbfs:/', '/dbfs')}/opt"
# # cache_dir = f"{ranker_path.replace('dbfs:/', '/dbfs')}/opt"

# ranker = Ranker(model_name="rank-T5-flan", cache_dir=cache_dir)

# rerankrequest = RerankRequest(query=question, passages=passages)
# results = ranker.rerank(rerankrequest)
# print(*results[:3], sep="\n\n")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Clean up Classroom
# MAGIC
# MAGIC **🚨 Warning:** Please refrain from deleting the catalog and tables created in this demo, as they are required for upcoming demonstrations. To clean up the classroom assets, execute the classroom clean-up script provided in the final demo.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, the objective was to generate embeddings from documents and store them in Vector Search. The initial step involved creating a Vector Search index, which required the establishment of a compute endpoint and the creation of an index that is synchronized with a source Delta table. Following this, we conducted a search for the stored indexes using a sample input query.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>