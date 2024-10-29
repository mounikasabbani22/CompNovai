# Databricks notebook source
# MAGIC %pip install --upgrade databricks-sdk==0.24.0 mlflow==2.13.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

import json
import pandas as pd
import time
from typing import List
import pyspark.sql.functions as F
from matplotlib import pyplot as plt

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from databricks.sdk.errors.platform import NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied
from databricks.sdk.service.vectorsearch import EndpointType, VectorIndexType, DeltaSyncVectorIndexSpecResponse, EmbeddingSourceColumn, PipelineType

import mlflow
from mlflow.metrics.genai import make_genai_metric, EvaluationExample

# COMMAND ----------

WORKSPACE_IDENTIFIER = spark.conf.get("spark.databricks.workspaceUrl").split(".")[0].replace("-", "")
VS_CATALOG = f"cmidev.cbu_ccims_poc.synth_data_claims_feature" # catalog shared across all workspace students
VS_ENDPOINT = "vs_endpoint_synthetic_claims_data" 
VS_INDEX = "cmidev.cbu_ccims_poc.synthetic_claim_text_embeddings_index"

DBRX_ENDPOINT = "databricks-dbrx-instruct"
LLAMA3_ENDPOINT = "databricks-meta-llama-3-70b-instruct"
GENERATED_QUESTIONS = "generated_questions"
GOLD_CHUNKS_FLAT = "spark_docs_gold_flat"
VS_INDEX_FULL_NAME = f"{VS_CATALOG}.{SCHEMA}.{VS_INDEX}"

# COMMAND ----------

src = f"{CATALOG}.{SCHEMA}.{GOLD_CHUNKS_FLAT}"
dest = f"{VS_CATALOG}.{SCHEMA}.{GOLD_CHUNKS_FLAT}"

if not spark.catalog.tableExists(dest):
    try:
        _ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {VS_CATALOG}")
        _ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {VS_CATALOG}.{SCHEMA}")
        _ = spark.sql(f"CREATE TABLE {dest} AS SELECT * FROM {src}")
        _ = spark.sql(f"GRANT USE_CATALOG, USE_SCHEMA, SELECT ON CATALOG {VS_CATALOG} to `account users`")
        _ = spark.sql(f"ALTER TABLE {dest} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    except Exception as e:
        print(f"Table creation failed. Check your permissions: {e}")
else:
    print(f"Table {dest} found!")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC First, create the vector search endpoints. These should be created for you in the classroom environment, but if you run this in your own environment with the correct permissions this will still work.

# COMMAND ----------

w = WorkspaceClient()

def create_endpoint(endpoint:str, w:WorkspaceClient) -> None:
    """
    Creates an endpoint in the specified workspace.

    This function interacts with a given workspace client to create an endpoint at the specified location.
    
    Parameters:
    ----------
    endpoint : str
        The endpoint to be created.
        
    w : WorkspaceClient
        An instance of WorkspaceClient which provides the necessary methods and context to interact with the workspace where the endpoint will be created.
        
    Returns:
    -------
    None
        This function does not return any value. It performs the endpoint creation as a side effect.
    """
    try:
        w.vector_search_endpoints.get_endpoint(endpoint)
        print(f"Endpoint {endpoint} exists. Skipping endpoint creation.")
    except NotFound as e:
        print(f"Endpoint {endpoint} doesn't exist! Creating. May take up to 20 minutes. Note this block will fail if the endpoint isn't pre-created and you don't have the proper endpoint creation permissions.")
        w.vector_search_endpoints.create_endpoint(
            name=endpoint,
            endpoint_type=EndpointType("STANDARD")
        )

create_endpoint(VS_ENDPOINT, w)

# COMMAND ----------

print(f"Checking the creation of {VS_ENDPOINT} and waiting for it to be provisioned.")
w.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(VS_ENDPOINT)
print("All endpoints created!")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Check if the vector search index exists. If not, create it.

# COMMAND ----------

def create_index(endpoint:str, w:WorkspaceClient, sync_index:bool=False) -> None:
    """
    Creates an index in the specified endpoint.

    This function interacts with a given workspace client to create an index at the specified endpoint.
    
    Parameters:
    ----------
    endpoint : str
        The endpoint where the index should be created.
        
    w : WorkspaceClient
        An instance of WorkspaceClient which provides the necessary methods and context to interact with the workspace where the index will be created.

    sync_index : bool
        Whether the index should be synced if the endpoint already exists.
        
    Returns:
    -------
    None
        This function does not return any value. It performs the index creation as a side effect.
    """
    for index in w.vector_search_indexes.list_indexes(VS_ENDPOINT):
        if index.name == VS_INDEX_FULL_NAME.lower() and not sync_index:
            print(f"Found existing index in endpoint {endpoint}. Skipping index syncing.")
            return
    try:
        w.vector_search_indexes.sync_index(f"{VS_CATALOG}.{SCHEMA}.{VS_INDEX}")
        print(f"Found existing index in endpoint {endpoint}. Synced index.")
    except ResourceDoesNotExist as e:
        print(f"Index in endpoint {endpoint} not found. Creating index.")
        try:
            w.vector_search_indexes.create_index(
                name=f"{VS_CATALOG}.{SCHEMA}.{VS_INDEX}",
                endpoint_name=endpoint,
                primary_key="uuid",
                index_type=VectorIndexType("DELTA_SYNC"),
                delta_sync_index_spec=DeltaSyncVectorIndexSpecResponse(
                    embedding_source_columns=[
                        EmbeddingSourceColumn(
                            name="content",
                            embedding_model_endpoint_name="databricks-bge-large-en"
                        )],
                    pipeline_type=PipelineType("TRIGGERED"),
                    source_table=f"{VS_CATALOG}.{SCHEMA}.{GOLD_CHUNKS_FLAT}"
                )
            )
        except PermissionDenied as e:
            print(f"You do not have permission to create an index. Skipping this for this notebook. You'll create an index in the lab. {e}")
    except BadRequest as e:
            print(f"Index not ready to sync for endpoint {endpoint}. Skipping for now. {e}")


create_index(VS_ENDPOINT, w)

# COMMAND ----------

# MAGIC %md
# MAGIC Define helper functions

# COMMAND ----------

def get_relevant_documents(question:str, index_name:str, k:int = 3, filters:str = None, max_retries:int = 3) -> List[dict]:
    """
    This function searches through the supplied vector index name and returns relevant documents 
    """
    docs = w.vector_search_indexes.query_index(
        index_name=index_name,
        columns=["uuid", "content", "category", "filepath"],
        filters_json=filters,
        num_results=k,
        query_text=question
    )
    docs_pd = pd.DataFrame(docs.result.data_array)
    docs_pd.columns = [_c.name for _c in docs.manifest.columns]
    return json.loads(docs_pd.to_json(orient="records"))

# COMMAND ----------

import time 

while w.vector_search_indexes.get_index(VS_INDEX_FULL_NAME).status.ready is not True:
    print("Vector search index is not ready yet...")
    time.sleep(30)

print("Vector search index is ready")

# COMMAND ----------

# MAGIC %md
# MAGIC It may take ~6 mins for the vector index to be ready. You can check the sync status under Compute -> Vector Search.

# COMMAND ----------

get_relevant_documents("How can I specify a list of columns to keep from my dataframe?", VS_INDEX_FULL_NAME, k=10)

# COMMAND ----------

# MAGIC %md
# MAGIC Take a look at the dataset below. The data consists of pyspark documentation html pages which have been parsed and chunked. The columns are:
# MAGIC - filepath : the original path on dbfs to the html file
# MAGIC - content : the text or html (if it is a table) of the chunk
# MAGIC - category : what type of chunk it is
# MAGIC - char_length : how many characters the chunk is
# MAGIC - chunk_num : what number (starting from 0) chunk it is within the html file
# MAGIC - uuid : a unique identifier for each chunk

# COMMAND ----------

display(spark.table(f"{CATALOG}.{SCHEMA}.spark_docs_gold_flat"))

# COMMAND ----------

candidates = (
    spark
        .table(f"{CATALOG}.{SCHEMA}.spark_docs_gold_flat")
        .filter(F.col("chunk_num") == F.lit(0))
        .filter(F.col("char_length") > 450)
        .filter(~F.col("content").startswith("Source code"))
)

candidates_pd = candidates.toPandas()
display(candidates_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate questions using LLM

# COMMAND ----------

SYSTEM_PROMPT = """You are a pyspark user who is going to ask a pyspark usage question related to the provided context. DO NOT use the function's name in your question. Answer only with your question related to the provided context encased in single quotes. Here are some examples of good questions:

Good Question in single quotes WITHOUT USING FUNCTION NAME:
'What function do I use to pick which columns to keep from my DataFrame?'

Good Question in single quotes WITHOUT USING FUNCTION NAME:
'How can I expand an array column to a single row per array entry?'
"""

USER_PROMPT = """{CONTEXT}

Good Question in single quotes WITHOUT USING FUNCTION NAME:
"""

# COMMAND ----------

def _send_llm_request(context : str) -> str:
        messages = [
            ChatMessage(
                role=ChatMessageRole("system"),
                content=SYSTEM_PROMPT
            ),
            ChatMessage(
                role=ChatMessageRole("user"),
                content=USER_PROMPT.format(CONTEXT=context)
            )
        ]
        response = w.serving_endpoints.query(
            name=DBRX_ENDPOINT,
            messages=messages
        )
        return response.choices[0].message.content

# COMMAND ----------

# MAGIC %md
# MAGIC The following cell might take ~6 mins to run.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Below, we use a few example generated questions from the dataframe above.

# COMMAND ----------

example_generated_question1 = candidates_pd[candidates_pd["content"].str.startswith("LinearDataGenerator")]["generated_question"].values[0]
print(example_generated_question1)

# COMMAND ----------

example_context2 = candidates_pd[candidates_pd["content"].str.startswith("pyspark.pandas.DataFrame.applymap")]["content"].values[0]
print(example_context2)

# COMMAND ----------

example_question2 = candidates_pd[candidates_pd["content"].str.startswith("pyspark.pandas.DataFrame.applymap")]["generated_question"].values[0]
print(example_question2)

# COMMAND ----------

example2 = EvaluationExample(
    input=example_context2,
    output=(example_question2),
    score=8,
    justification=(
        "The generated question is broad and doesn't directly reference the function name in the context."
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we can define custom metrics to assess the quality of the generated questions. We will provide guidance to the LLM how to assign a score to each generated question.

# COMMAND ----------

eval_sample = {"OEM_CODE":1608,"DEALER":"22887","DISTR":"1694","FC":"3382.0,3446.0,5177.0,1939.0","ENGINE_NAME_DESC":"X15 1 2017","FAILCODE":"THAC","SHOPORDERNUM":"3037900","FAILURE_MODE_BUCKET":"ACT(E) ELECTROLYTIC C26 FAILURE","CLEANED_CORRECTION":" Drained the coolant and replaced the turbo actuator, Filled withcoolant and Ran a regen and test drove all codes went inactive,Cleared the inactive codesHad code 5177 and TS in EDS, 1939 can and battery voltage were inspec, Replaced turbo act per EDS, All codes went inactive, cause failturbo actuator EPR Part Number:637477200RX Tracking Number:770339114459,CHECK ENGINE LIGHT  YELLOWTURBOCHARGER ACTUATORSEIZED, STUCK, SCORED, SCUFFED, SPUNTURBOCHARGER ACTUATORSHORTED ELECTRICAL ONLYAWILLIS FRIDAY 28OCT2022 10:07:54 AM **IMPORTANT**EPR**PART REQUEST - Review PRINT RETURN TAG for return instructions **Go to QSOL/Warranty/Parts Return and refer to memo**"} 


synthetic_sample = {"OEM_CODE":1457,"DEALER":"24989","DISTR":"1693","FC":"687.0,4616.0","ENGINE_NAME_DESC":"X15 3 2017","FAILCODE":"ELTS","SHOPORDERNUM":"1032900","FAILURE_MODE_BUCKET":"ACT(E) ELECTROLYTIC C26 FAILURE","CLEANED_CORRECTION":"Replaced turbo speed sensor after C26 failure. Performed stationary regen and road test."}

# COMMAND ----------

eval_pd = pd.DataFrame([eval_sample])
synthetic_pd = pd.DataFrame([synthetic_sample])

# COMMAND ----------

example1 = EvaluationExample(
    input=eval_sample,
    output="ACT(E) ELECTROLYTIC C26 FAILURE",
    score=4,
    justification=(
        "The generated question references 'provided utils'. The questions should be standalone. Additionally it uses a lot of exact terminology directly from the context."
    )
)

# COMMAND ----------

example2 = EvaluationExample(
    input=synthetic_sample,
    output="ACT(E) ELECTROLYTIC C26 FAILURE",
    score=4,
    justification=(
        "The generated question references 'provided utils'. The questions should be standalone. Additionally it uses a lot of exact terminology directly from the context."
    )
)

# COMMAND ----------

example1

# COMMAND ----------

question_quality = make_genai_metric(
  name="SyntheticDataQuality",
  definition=(
      "Measures the quality of the synthetic data by comparing it to real corporate data."),
  grading_prompt=(
      """Synthetic Data Quality: If the synthetic data seems unrealistic, overly specific, or re-uses too much terminology from real data, we will give it a low score. If the synthetic data is broad, plausible, and closely mimics the general patterns of the real data, we will give it a high score.
      - Score 0: The synthetic data doesn't resemble real-world data at all.
      - Score 2: The synthetic data is too specific, doesn't sound realistic, and uses overly technical or irrelevant details.
      - Score 4: The synthetic data somewhat resembles real-world data but still uses too much unnecessary detail or specific terminology.
      - Score 7: The synthetic data closely resembles real-world data, though minor inconsistencies are present.
      - Score 10: The synthetic data looks exactly like real-world data, with natural language and plausible details."""
  ),
  model=f"endpoints:/databricks-gte-large-en",
  examples=[example1, example2],
  parameters={"temperature": 0.0},
  aggregations=["mean", "variance"],
  greater_is_better=True,
)

# COMMAND ----------

eval_df = pd.DataFrame(
  {
  "inputs": [
    {"OEM_CODE":1608,"DEALER":"22887","DISTR":"1694","FC":"3382.0,3446.0,5177.0,1939.0","ENGINE_NAME_DESC":"X15 1 2017","FAILCODE":"THAC","SHOPORDERNUM":"3037900","FAILURE_MODE_BUCKET":"ACT(E) ELECTROLYTIC C26 FAILURE","CLEANED_CORRECTION":" Drained the coolant and replaced the turbo actuator, Filled withcoolant and Ran a regen and test drove all codes went inactive,Cleared the inactive codesHad code 5177 and TS in EDS, 1939 can and battery voltage were inspec, Replaced turbo act per EDS, All codes went inactive, cause failturbo actuator EPR Part Number:637477200RX Tracking Number:770339114459,CHECK ENGINE LIGHT  YELLOWTURBOCHARGER ACTUATORSEIZED, STUCK, SCORED, SCUFFED, SPUNTURBOCHARGER ACTUATORSHORTED ELECTRICAL ONLYAWILLIS FRIDAY 28OCT2022 10:07:54 AM **IMPORTANT**EPR**PART REQUEST - Review PRINT RETURN TAG for return instructions **Go to QSOL/Warranty/Parts Return and refer to memo**"}
  ],

  "predictions": [
    {"ACT(E) ELECTROLYTIC C26 FAILURE"}
  ]
  }
)

# COMMAND ----------

synthetic_df = pd.DataFrame(
  {
  "inputs": [
    {"OEM_CODE":1457,"DEALER":"24989","DISTR":"1693","FC":"687.0,4616.0","ENGINE_NAME_DESC":"X15 3 2017","FAILCODE":"ELTS","SHOPORDERNUM":"1032900","FAILURE_MODE_BUCKET":"ACT(E) ELECTROLYTIC C26 FAILURE", "CLEANED_CORRECTION":"Replaced turbo speed sensor after C26 failure. Performed stationary regen and road test."}
  ],

  "predictions": [
    {"ACT(E) ELECTROLYTIC C26 FAILURE"}
  ]
}
)

# COMMAND ----------

import mlflow
import pandas as pd

# Convert 'inputs' and 'predictions' into separate DataFrames
eval_inputs = pd.DataFrame(eval_df['inputs'].values.tolist())
eval_predictions = pd.DataFrame([list(pred)[0] for pred in eval_df['predictions'].values], columns=['predictions'])

synthetic_inputs = pd.DataFrame(synthetic_df['inputs'].values.tolist())
synthetic_predictions = pd.DataFrame([list(pred)[0] for pred in synthetic_df['predictions'].values], columns=['predictions'])

print("eval_inputs", eval_inputs)
print("synthetic_inputs", synthetic_inputs)

print("eval_predictions", eval_predictions)
print("synthetic_predictions", synthetic_predictions)

print("eval_df_full", eval_df_full.columns)
print("synthetic_df_full", synthetic_df_full.columns)

# Combine inputs with predictions
eval_df_full = pd.concat([eval_inputs, eval_predictions], axis=1)
synthetic_df_full = pd.concat([synthetic_inputs, synthetic_predictions], axis=1)

print("eval_df_full", eval_df_full)
print("synthetic_df_full", synthetic_df_full)

# Print out the shapes to verify matching row counts
print("Eval DF shape:", eval_df_full.shape)
print("Synthetic DF shape:", synthetic_df_full.shape)

# Define evaluation function using mlflow.evaluate
def evaluate_model(eval_df_full, synthetic_df_full):
    with mlflow.start_run():
        # Log evaluation results for real data
        eval_metrics = mlflow.evaluate(
            model_type="question-answering",  # assuming you're comparing data, not model
            data=eval_df_full,
            targets='FAILURE_MODE_BUCKET',
            predictions="predictions",
            # model_type="classifier",  # Change this to regression if needed
            evaluators=["default"],
        )

        # Log evaluation results for synthetic data
        synthetic_metrics = mlflow.evaluate(
            model_type="question-answering",
            data=synthetic_df_full,
            targets='FAILURE_MODE_BUCKET',
            predictions="predictions",
            # model_type="classifier",
            evaluators=["default"],
        )

        return eval_metrics, synthetic_metrics

# Call the evaluation function
real_metrics, synthetic_metrics = evaluate_model(eval_df_full, synthetic_df_full)

# Output the metrics
print("Real Data Metrics: ", real_metrics)
print("Synthetic Data Metrics: ", synthetic_metrics)


# COMMAND ----------

real_metrics.metrics

# COMMAND ----------

synthetic_metrics.metrics

# COMMAND ----------

# MAGIC %md
# MAGIC # Compute evaluation metrics using MLflow
# MAGIC The cell below could take ~8mins to complete.

# COMMAND ----------

with mlflow.start_run() as run:
    question_eval = mlflow.evaluate(
        data=eval_pd,
        model_type="question-answering",
        predictions="predictions",
        extra_metrics=[question_quality], # This now includes our custom metric!
      )

# COMMAND ----------

eval_results_table = pd.DataFrame.from_dict(question_eval.tables["eval_results_table"])
eval_results_table

# COMMAND ----------

question_eval.metrics

# COMMAND ----------

(
    spark
        .createDataFrame(candidates_pd)
        .select("uuid","generated_question","content")
        .write
        .mode("overwrite")
        .option("overwriteSchema","true")
        .saveAsTable(f"{CATALOG}.{SCHEMA}.{GENERATED_QUESTIONS}")
)

# COMMAND ----------

generated_questions = (spark.table("select * from cmidev.cbu_ccims_poc.synth_data_claims_feature"))
train_df, eval_df = generated_questions.randomSplit([0.8,0.2], seed=41)
display(train_df)

# COMMAND ----------

# Write both train and eval dataframes out 
(
    train_df
        .write
        .mode("overwrite")
        .option("overwriteSchema","true")
        .saveAsTable(f"{CATALOG}.{SCHEMA}.{GENERATED_QUESTIONS}_train")
)

(
    eval_df
        .write
        .mode("overwrite")
        .option("overwriteSchema","true")
        .saveAsTable(f"{CATALOG}.{SCHEMA}.{GENERATED_QUESTIONS}_eval")
)

# COMMAND ----------

eval_pd = eval_df.toPandas()

display(eval_pd)

# COMMAND ----------

def get_relevant_doc_ids(question : str) -> list[str]:
    docs = get_relevant_documents(question, index_name=VS_INDEX_FULL_NAME, k=10)
    return [_x["uuid"] for _x in docs]
    
get_relevant_doc_ids("How can I read CSV files using Structured Streaming in PySpark?")

# COMMAND ----------

eval_pd["retrieved_docs"] = eval_pd["generated_question"].transform(get_relevant_doc_ids)

# COMMAND ----------

with mlflow.start_run() as run:
    eval_results = mlflow.evaluate(
        data = eval_pd,
        model_type="retriever",
        targets="uuid",
        predictions="retrieved_docs",
        evaluators="default",
        extra_metrics=[mlflow.metrics.recall_at_k(i) for i in range(1,10,1)]
    )

# COMMAND ----------

plt.plot([eval_results.metrics[f"recall_at_{i}/mean"] for i in range(1,10,1)], label="recall")
plt.title("Recall at k")
plt.xlabel("k")
plt.legend()
plt.show()
