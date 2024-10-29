# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Preparing Data for RAG
# MAGIC This notebook takes care of below: 
# MAGIC
# MAGIC * Split the document data into chunks that are at least as small as the maximum context window of the LLM to be used later.
# MAGIC
# MAGIC * Choose an embedding model.
# MAGIC
# MAGIC * Compute embeddings for each of the chunks using a Databricks-managed embedding model.
# MAGIC
# MAGIC * Use the chunking strategy to divide up the context information to be provided to a model.
# MAGIC
# MAGIC * Store embeddings into a table for further use
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12 14.3.x-scala2.12**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Setup Installation
# MAGIC
# MAGIC Install required libraries.

# COMMAND ----------

# MAGIC %pip install --quiet PyMuPDF mlflow==2.14.3 transformers==4.44.0 "unstructured[pdf,docx]==0.14.10" llama-index==0.10.62 pydantic==2.8.2 accelerate
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install --upgrade urllib3 transformers
# MAGIC %pip uninstall -y transformers
# MAGIC %pip install transformers

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract PDF Content as Text Chunks
# MAGIC
# MAGIC As the first step, we need to ingest PDF files and divide the content into chunks. PDF files are already downloaded during the course step and stored in **datasets path**.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Creating Spark Dataframe for Synthetic Data 

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

# Start Spark session
# spark = SparkSession.builder.appName("MLflow Data Evaluation").getOrCreate()

# Define the schema with DOCUMENT as a struct
schema = StructType([
    StructField("CLAIM_ID_SEQ", IntegerType(), True),
    StructField("OEM_CODE", IntegerType(), True),
    StructField("DEALER", StringType(), True),
    StructField("DISTR", StringType(), True),
    StructField("FC", StringType(), True),
    StructField("ENGINE_NAME_DESC", StringType(), True),
    StructField("FAILCODE", StringType(), True),
    StructField("SHOPORDERNUM", StringType(), True),
    StructField("FAILURE_MODE_BUCKET", StringType(), True),
    StructField("CLEANED_CORRECTION", StringType(), True),
    StructField("DOCUMENT", StructType([
        StructField("OEM_CODE", IntegerType(), True),
        StructField("DEALER", StringType(), True),
        StructField("DISTR", StringType(), True),
        StructField("FC", StringType(), True),
        StructField("ENGINE_NAME_DESC", StringType(), True),
        StructField("FAILCODE", StringType(), True),
        StructField("SHOPORDERNUM", StringType(), True),
        StructField("FAILURE_MODE_BUCKET", StringType(), True),
        StructField("CLEANED_CORRECTION", StringType(), True)
        ]), True)
])

# COMMAND ----------

from pyspark.sql import SparkSession

# create a SparkSession
spark = SparkSession.builder.appName("Turbocharger Warranty Claims").getOrCreate()

# define the data
data = [
(17695581, 1457, "24989", "1693", "687.0,4616.0", "X15 3 2017", "ELTS", "1032900", "ACT(E) ELECTROLYTIC C26 FAILURE", "Replaced turbo speed sensor after C26 failure. Performed stationary regen and road test.", {"OEM_CODE": 1457, "DEALER": "24989", "DISTR" : "1693", "FC" : "687.0,4616.0", "ENGINE_NAME_DESC": "X15 3 2017", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032900", "FAILURE_MODE_BUCKET": "ACT(E) ELECTROLYTIC C26 FAILURE", "CLEANED_CORRECTION": "Replaced turbo speed sensor after C26 failure. Performed stationary regen and road test."}),

(17695582, 1458, "24990", "1694", "123.0,4567.0", "ISX12 2020", "THAS", "1032901", "ACT(E) ELECTROLYTIC C26 FAILURE", "Replaced turbocharger assembly due to C26 failure. Completed functional test post-repair.", {"OEM_CODE": 1458, "DEALER": "24990", "DISTR" : "1694", "FC" : "123.0,4567.0", "ENGINE_NAME_DESC": "ISX12 2020", "FAILCODE": "THAS", "SHOPORDERNUM": "1032901", "FAILURE_MODE_BUCKET": "ACT(E) ELECTROLYTIC C26 FAILURE", "CLEANED_CORRECTION": "Replaced turbocharger assembly due to C26 failure. Completed functional test post-repair."}),

(17695583, 1459, "24991", "1695", "901.0,4211.0", "X15 3 2018", "ELTS", "1032902", "ACT(E) ELECTROLYTIC C26 FAILURE", "Replaced turbocharger actuator after C26 failure. Performed stationary regen and road test.", {"OEM_CODE": 1459, "DEALER": "24991", "DISTR" : "1695", "FC" : "901.0,4211.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032902", "FAILURE_MODE_BUCKET": "ACT(E) ELECTROLYTIC C26 FAILURE", "CLEANED_CORRECTION": "Replaced turbocharger actuator after C26 failure. Performed stationary regen and road test."}),

(17695584, 1460, "24992", "1696", "234.0,3987.0", "ISX15 2019", "THAS", "1032903", "ACT(E) ELECTROLYTIC C26 FAILURE", "Replaced speed sensor due to C26 failure. Completed functional test post-repair.", {"OEM_CODE": 1460, "DEALER": "24992", "DISTR" : "1696", "FC" : "234.0,3987.0", "ENGINE_NAME_DESC": "ISX15 2019", "FAILCODE": "THAS", "SHOPORDERNUM": "1032903", "FAILURE_MODE_BUCKET": "ACT(E) ELECTROLYTIC C26 FAILURE", "CLEANED_CORRECTION": "Replaced speed sensor due to C26 failure. Completed functional test post-repair."}),

(17695585, 1461, "24993", "1697", "567.0,4563.0", "X15 3 2017", "ELTS", "1032904", "ACT(E) ELECTROLYTIC C26 FAILURE", "Replaced turbocharger assembly due to C26 failure. Performed stationary regen and road test.", {"OEM_CODE": 1461, "DEALER": "24993", "DISTR" : "1697", "FC" : "567.0,4563.0", "ENGINE_NAME_DESC": "X15 3 2017", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032904", "FAILURE_MODE_BUCKET": "ACT(E) ELECTROLYTIC C26 FAILURE", "CLEANED_CORRECTION": "Replaced turbocharger assembly due to C26 failure. Performed stationary regen and road test."}),

(17695588, 1464, "24996", "1700", "789.0,4573.0", "X15 3 2018", "THAC", "1032907", "CONSEQUENTIAL DAMAGE", "Replaced faulty throttle actuator due to consequential damage. Performed road test post-repair.", {"OEM_CODE": 1464, "DEALER": "24996", "DISTR" : "1700", "FC" : "789.0,4573.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "THAC", "SHOPORDERNUM": "1032907", "FAILURE_MODE_BUCKET": "CONSEQUENTIAL DAMAGE", "CLEANED_CORRECTION": "Replaced faulty throttle actuator due to consequential damage. Performed road test post-repair."}),

(17695589, 1465, "24997", "1701", "890.0,4574.0", "ISX12 2020", "THAS", "1032908", "CONSEQUENTIAL DAMAGE", "Replaced faulty turbocharger speed sensor due to consequential damage. Performed functional test post-repair.", {"OEM_CODE": 1465, "DEALER": "24997", "DISTR" : "1701", "FC" : "890.0,4574.0", "ENGINE_NAME_DESC": "ISX12 2020", "FAILCODE": "THAS", "SHOPORDERNUM": "1032908", "FAILURE_MODE_BUCKET": "CONSEQUENTIAL DAMAGE", "CLEANED_CORRECTION": "Replaced faulty turbocharger speed sensor due to consequential damage. Performed functional test post-repair."}),

(17695590, 1466, "24998", "1702", "345.0,4211.0", "X15 3 2017", "THAC", "1032909", "CONSEQUENTIAL DAMAGE", "Replaced faulty fuel injector due to consequential damage. Performed stationary regen and road test.", {"OEM_CODE": 1466, "DEALER": "24998", "DISTR" : "1702", "FC" : "345.0,4211.0", "ENGINE_NAME_DESC": "X15 3 2017", "FAILCODE": "THAC", "SHOPORDERNUM": "1032909", "FAILURE_MODE_BUCKET": "CONSEQUENTIAL DAMAGE", "CLEANED_CORRECTION": "Replaced faulty fuel injector due to consequential damage. Performed stationary regen and road test."}),

(17695591, 1467, "24999", "1703", "123.0,3987.0", "ISX15 2019", "THAS", "1032910", "CONSEQUENTIAL DAMAGE", "Replaced faulty engine control module due to consequential damage. Completed functional test post-repair.", {"OEM_CODE": 1467, "DEALER": "24999", "DISTR" : "1703", "FC" : "123.0,3987.0", "ENGINE_NAME_DESC": "ISX15 2019", "FAILCODE": "THAS", "SHOPORDERNUM": "1032910", "FAILURE_MODE_BUCKET": "CONSEQUENTIAL DAMAGE", "CLEANED_CORRECTION": "Replaced faulty engine control module due to consequential damage. Completed functional test post-repair."}),

(17695592, 1468, "25000", "1704", "567.0,4563.0", "X15 3 2018", "THAC", "1032911", "CONSEQUENTIAL DAMAGE", "Replaced faulty turbocharger actuator due to consequential damage. Performed road test post-repair.", {"OEM_CODE": 1468, "DEALER": "25000", "DISTR" : "1704", "FC" : "567.0,4563.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "THAC", "SHOPORDERNUM": "1032911", "FAILURE_MODE_BUCKET": "CONSEQUENTIAL DAMAGE", "CLEANED_CORRECTION": "Replaced faulty turbocharger actuator due to consequential damage. Performed road test post-repair."}),

(17695594, 1470, "25002", "1706", "567.0,4579.0", "ISX12 2020", "THAS", "1032913", "TSS - WELD PROCESS DEFECT", "Replaced faulty turbocharger speed sensor due to TSS weld process defect. Performed functional test post-repair.", {"OEM_CODE": 1470, "DEALER": "25002", "DISTR" : "1706", "FC" : "567.0,4579.0", "ENGINE_NAME_DESC": "ISX12 2020", "FAILCODE": "THAS", "SHOPORDERNUM": "1032913", "FAILURE_MODE_BUCKET": "TSS - WELD PROCESS DEFECT", "CLEANED_CORRECTION": "Replaced faulty turbocharger speed sensor due to TSS weld process defect. Performed functional test post-repair."}),

(17695595, 1471, "25003", "1707", "678.0,4580.0", "X15 3 2017", "ELTS", "1032914", "TSS - WELD PROCESS DEFECT", "Replaced faulty turbo speed sensor due to TSS weld process defect. Performed stationary regen and road test.", {"OEM_CODE": 1471, "DEALER": "25003", "DISTR" : "1707", "FC" : "678.0,4580.0", "ENGINE_NAME_DESC": "X15 3 2017", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032914", "FAILURE_MODE_BUCKET": "TSS - WELD PROCESS DEFECT", "CLEANED_CORRECTION": "Replaced faulty turbo speed sensor due to TSS weld process defect. Performed stationary regen and road test."}),

(17695596, 1472, "25004", "1708", "234.0,4211.0", "ISX15 2019", "THAS", "1032915", "TSS - WELD PROCESS DEFECT", "Replaced faulty turbocharger actuator due to TSS weld process defect. Completed functional test post-repair.", {"OEM_CODE": 1472, "DEALER": "25004", "DISTR" : "1708", "FC" : "234.0,4211.0", "ENGINE_NAME_DESC": "ISX15 2019", "FAILCODE": "THAS", "SHOPORDERNUM": "1032915", "FAILURE_MODE_BUCKET": "TSS - WELD PROCESS DEFECT", "CLEANED_CORRECTION": "Replaced faulty turbocharger actuator due to TSS weld process defect. Completed functional test post-repair."}),

(17695597, 1473, "25005", "1709", "901.0,3987.0", "X15 3 2018", "ELTS", "1032916", "TSS - WELD PROCESS DEFECT", "Replaced faulty turbo speed sensor due to TSS weld process defect. Performed stationary regen and road test.", {"OEM_CODE": 1473, "DEALER": "25005", "DISTR" : "1709", "FC" : "901.0,3987.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032916", "FAILURE_MODE_BUCKET": "TSS - WELD PROCESS DEFECT", "CLEANED_CORRECTION": "Replaced faulty turbo speed sensor due to TSS weld process defect. Performed stationary regen and road test."}),

(17695598, 1474, "25006", "1710", "456.0,4563.0", "ISX12 2020", "THAS", "1032917", "TSS - WELD PROCESS DEFECT", "Replaced faulty turbocharger speed sensor due to TSS weld process defect. Completed functional test post-repair.", {"OEM_CODE": 1474, "DEALER": "25006", "DISTR" : "1710", "FC" : "456.0,4563.0", "ENGINE_NAME_DESC": "ISX12 2020", "FAILCODE": "THAS", "SHOPORDERNUM": "1032917", "FAILURE_MODE_BUCKET": "TSS - WELD PROCESS DEFECT", "CLEANED_CORRECTION": "Replaced faulty turbocharger speed sensor due to TSS weld process defect. Completed functional test post-repair."}),

(17695599, 1475, "25007", "1711", "234.0,4211.0", "ISX15 2020", "ELTS", "1032918", "TSS - INTERMITTENT SIGNAL", "Replaced faulty turbocharger speed sensor due to TSS intermittent signal. Completed functional test post-repair.", {"OEM_CODE": 1475, "DEALER": "25007", "DISTR" : "1711", "FC" : "234.0,4211.0", "ENGINE_NAME_DESC": "ISX15 2020", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032918", "FAILURE_MODE_BUCKET": "TSS - INTERMITTENT SIGNAL", "CLEANED_CORRECTION": "Replaced faulty turbocharger speed sensor due to TSS intermittent signal. Completed functional test post-repair."}),

(17695600, 1476, "25008", "1712", "456.0,3987.0", "X15 3 2017", "THAC", "1032919", "TSS - INTERMITTENT SIGNAL", "Replaced faulty engine control module due to TSS intermittent signal. Performed stationary regen and road test.", {"OEM_CODE": 1476, "DEALER": "25008", "DISTR" : "1712", "FC" : "456.0,3987.0", "ENGINE_NAME_DESC": "X15 3 2017", "FAILCODE": "THAC", "SHOPORDERNUM": "1032919", "FAILURE_MODE_BUCKET": "TSS - INTERMITTENT SIGNAL", "CLEANED_CORRECTION": "Replaced faulty engine control module due to TSS intermittent signal. Performed stationary regen and road test."}),

(17695601, 1477, "25009", "1713", "789.0,4563.0", "ISX12 2019", "ELTS", "1032920", "TSS - INTERMITTENT SIGNAL", "Replaced faulty throttle actuator due to TSS intermittent signal. Completed functional test post-repair.", {"OEM_CODE": 1477, "DEALER": "25009", "DISTR" : "1713", "FC" : "789.0,4563.0", "ENGINE_NAME_DESC": "ISX12 2019", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032920", "FAILURE_MODE_BUCKET": "TSS - INTERMITTENT SIGNAL", "CLEANED_CORRECTION": "Replaced faulty throttle actuator due to TSS intermittent signal. Completed functional test post-repair."}),

(17695602, 1478, "25010", "1714", "456.0,4587.0", "ISX12 2019", "ELTS", "1032921", "ACT(E) DAMAGE INDUCED (THERMAL OVERSTRESS)", "Replaced faulty engine speed sensor due to thermal overstress. Completed stationary regen post-repair.", {"OEM_CODE": 1478, "DEALER": "25010", "DISTR" : "1714", "FC" : "456.0,4587.0", "ENGINE_NAME_DESC": "ISX12 2019", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032921", "FAILURE_MODE_BUCKET": "ACT(E) DAMAGE INDUCED (THERMAL OVERSTRESS)", "CLEANED_CORRECTION": "Replaced faulty engine speed sensor due to thermal overstress. Completed stationary regen post-repair."}),

(17695603, 1479, "25011", "1715", "789.0,4211.0", "X15 3 2018", "THAS", "1032922", "ACT(E) DAMAGE INDUCED (THERMAL OVERSTRESS)", "Replaced faulty turbocharger actuator due to thermal overstress. Performed road test post-repair.", {"OEM_CODE": 1479, "DEALER": "25011", "DISTR" : "1715", "FC" : "789.0,4211.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "THAS", "SHOPORDERNUM": "1032922", "FAILURE_MODE_BUCKET": "ACT(E) DAMAGE INDUCED (THERMAL OVERSTRESS)", "CLEANED_CORRECTION": "Replaced faulty turbocharger actuator due to thermal overstress. Performed road test post-repair."}),

(17695604, 1480, "25012", "1716", "901.0,3987.0", "ISX15 2020", "ELTS", "1032923", "ACT(E) DAMAGE INDUCED (THERMAL OVERSTRESS)", "Replaced faulty engine control module due to thermal overstress. Completed functional test post-repair.", {"OEM_CODE": 1480, "DEALER": "25012", "DISTR" : "1716", "FC" : "901.0,3987.0", "ENGINE_NAME_DESC": "ISX15 2020", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032923", "FAILURE_MODE_BUCKET": "ACT(E) DAMAGE INDUCED (THERMAL OVERSTRESS)", "CLEANED_CORRECTION": "Replaced faulty engine control module due to thermal overstress. Completed functional test post-repair."}),

(17695605, 1481, "25013", "1717", "234.0,4563.0", "X15 3 2017", "THAS", "1032924", "ACT(E) DAMAGE INDUCED (THERMAL OVERSTRESS)", "Replaced faulty turbo speed sensor due to thermal overstress. Performed stationary regen and road test.", {"OEM_CODE": 1481, "DEALER": "25013", "DISTR" : "1717", "FC" : "234.0,4563.0", "ENGINE_NAME_DESC": "X15 3 2017", "FAILCODE": "THAS", "SHOPORDERNUM": "1032924", "FAILURE_MODE_BUCKET": "ACT(E) DAMAGE INDUCED (THERMAL OVERSTRESS)", "CLEANED_CORRECTION": "Replaced faulty turbo speed sensor due to thermal overstress. Performed stationary regen and road test."}),

(17695608, 1484, "25016", "1720", "123.0,4593.0", "X15 3 2018", "THAC", "1032927", "ACT(E) LOOSE APS MAGNET", "Replaced faulty throttle actuator due to loose APS magnet. Performed road test post-repair.", {"OEM_CODE": 1484, "DEALER": "25016", "DISTR" : "1720", "FC" : "123.0,4593.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "THAC", "SHOPORDERNUM": "1032927", "FAILURE_MODE_BUCKET": "ACT(E) LOOSE APS MAGNET", "CLEANED_CORRECTION": "Replaced faulty throttle actuator due to loose APS magnet. Performed road test post-repair."}),

(17695609, 1485, "25017", "1721", "234.0,4594.0", "ISX12 2020", "THAS", "1032928", "ACT(E) LOOSE APS MAGNET", "Replaced faulty turbocharger speed sensor due to loose APS magnet. Performed functional test post-repair.", {"OEM_CODE": 1485, "DEALER": "25017", "DISTR" : "1721", "FC" : "234.0,4594.0", "ENGINE_NAME_DESC": "ISX12 2020", "FAILCODE": "THAS", "SHOPORDERNUM": "1032928", "FAILURE_MODE_BUCKET": "ACT(E) LOOSE APS MAGNET", "CLEANED_CORRECTION": "Replaced faulty turbocharger speed sensor due to loose APS magnet. Performed functional test post-repair."}),

(17695610, 1486, "25018", "1722", "789.0,4211.0", "X15 3 2017", "THAC", "1032929", "ACT(E) LOOSE APS MAGNET", "Replaced faulty engine control module due to loose APS magnet. Completed stationary regen post-repair.", {"OEM_CODE": 1486, "DEALER": "25018", "DISTR" : "1722", "FC" : "789.0,4211.0", "ENGINE_NAME_DESC": "X15 3 2017", "FAILCODE": "THAC", "SHOPORDERNUM": "1032929", "FAILURE_MODE_BUCKET": "ACT(E) LOOSE APS MAGNET", "CLEANED_CORRECTION": "Replaced faulty engine control module due to loose APS magnet. Completed stationary regen post-repair."}),

(17695611, 1487, "25019", "1723", "901.0,3987.0", "ISX15 2019", "THAS", "1032930", "ACT(E) LOOSE APS MAGNET", "Replaced faulty throttle actuator due to loose APS magnet. Performed road test post-repair.", {"OEM_CODE": 1487, "DEALER": "25019", "DISTR" : "1723", "FC" : "901.0,3987.0", "ENGINE_NAME_DESC": "ISX15 2019", "FAILCODE": "THAS", "SHOPORDERNUM": "1032930", "FAILURE_MODE_BUCKET": "ACT(E) LOOSE APS MAGNET", "CLEANED_CORRECTION": "Replaced faulty throttle actuator due to loose APS magnet. Performed road test post-repair."}),

(17695612, 1488, "25020", "1724", "456.0,4563.0", "X15 3 2020", "THAC", "1032931", "ACT(E) LOOSE APS MAGNET", "Replaced faulty turbo speed sensor due to loose APS magnet. Performed stationary regen and road test.", {"OEM_CODE": 1488, "DEALER": "25020", "DISTR" : "1724", "FC" : "456.0,4563.0", "ENGINE_NAME_DESC": "X15 3 2020", "FAILCODE": "THAC", "SHOPORDERNUM": "1032931", "FAILURE_MODE_BUCKET": "ACT(E) LOOSE APS MAGNET", "CLEANED_CORRECTION": "Replaced faulty turbo speed sensor due to loose APS magnet. Performed stationary regen and road test."}),

(17695612, 1488, "25020", "1724", "567.0,4597.0", "ISX12 2019", "ELTS", "1032931", "TURBINE/NOZZLE FOD", "Replaced faulty engine speed sensor due to turbine/nozzle FOD. Completed stationary regen post-repair.", {"OEM_CODE": 1488, "DEALER": "25020", "DISTR" : "1724", "FC" : "567.0,4597.0", "ENGINE_NAME_DESC": "ISX12 2019", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032931", "FAILURE_MODE_BUCKET": "TURBINE/NOZZLE FOD", "CLEANED_CORRECTION": "Replaced faulty engine speed sensor due to turbine/nozzle FOD. Completed stationary regen post-repair."}),

(17695614, 1490, "25022", "1726", "789.0,4599.0", "ISX12 2020", "THAS", "1032933", "TURBINE/NOZZLE FOD", "Replaced faulty turbocharger speed sensor due to turbine/nozzle FOD. Performed functional test post-repair.", {"OEM_CODE": 1490, "DEALER": "25022", "DISTR" : "1726", "FC" : "789.0,4599.0", "ENGINE_NAME_DESC": "ISX12 2020", "FAILCODE": "THAS", "SHOPORDERNUM": "1032933", "FAILURE_MODE_BUCKET": "TURBINE/NOZZLE FOD", "CLEANED_CORRECTION": "Replaced faulty turbocharger speed sensor due to turbine/nozzle FOD. Performed functional test post-repair."}),

(17695615, 1491, "25023", "1727", "234.0,4211.0", "X15 3 2018", "ELTS", "1032934", "TURBINE/NOZZLE FOD", "Replaced faulty throttle actuator due to turbine/nozzle FOD. Performed road test post-repair.", {"OEM_CODE": 1491, "DEALER": "25023", "DISTR" : "1727", "FC" : "234.0,4211.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032934", "FAILURE_MODE_BUCKET": "TURBINE/NOZZLE FOD", "CLEANED_CORRECTION": "Replaced faulty throttle actuator due to turbine/nozzle FOD. Performed road test post-repair."}),

(17695616, 1492, "25024", "1728", "901.0,3987.0", "ISX15 2019", "THAS", "1032935", "TURBINE/NOZZLE FOD", "Replaced faulty engine control module due to turbine/nozzle FOD. Completed functional test post-repair.", {"OEM_CODE": 1492, "DEALER": "25024", "DISTR" : "1728", "FC" : "901.0,3987.0", "ENGINE_NAME_DESC": "ISX15 2019", "FAILCODE": "THAS", "SHOPORDERNUM": "1032935", "FAILURE_MODE_BUCKET": "TURBINE/NOZZLE FOD", "CLEANED_CORRECTION": "Replaced faulty engine control module due to turbine/nozzle FOD. Completed functional test post-repair."}),

(17695617, 1493, "25025", "1729", "456.0,4563.0", "X15 3 2020", "ELTS", "1032936", "TURBINE/NOZZLE FOD", "Replaced faulty turbo speed sensor due to turbine/nozzle FOD. Performed stationary regen and road test.", {"OEM_CODE": 1493, "DEALER": "25025", "DISTR" : "1729", "FC" : "456.0,4563.0", "ENGINE_NAME_DESC": "X15 3 2020", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032936", "FAILURE_MODE_BUCKET": "TURBINE/NOZZLE FOD", "CLEANED_CORRECTION": "Replaced faulty turbo speed sensor due to turbine/nozzle FOD. Performed stationary regen and road test."}),

(17695612, 1488, "25020", "1724", "567.0,4597.0", "ISX12 2019", "ELTS", "1032931", "TURBINE/NOZZLE FOD", "Replaced faulty engine speed sensor due to turbine/nozzle FOD. Completed stationary regen post-repair.", {"OEM_CODE": 1488, "DEALER": "25020", "DISTR" : "1724", "FC" : "567.0,4597.0", "ENGINE_NAME_DESC": "ISX12 2019", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032931", "FAILURE_MODE_BUCKET": "TURBINE/NOZZLE FOD", "CLEANED_CORRECTION": "Replaced faulty engine speed sensor due to turbine/nozzle FOD. Completed stationary regen post-repair."}),

(17695613, 1489, "25021", "1725", "678.0,4598.0", "X15 3 2018", "THAC", "1032932", "TURBINE/NOZZLE FOD", "Replaced faulty throttle actuator due to turbine/nozzle FOD. Performed road test post-repair.", {"OEM_CODE": 1489, "DEALER": "25021", "DISTR" : "1725", "FC" : "678.0,4598.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "THAC", "SHOPORDERNUM": "1032932", "FAILURE_MODE_BUCKET": "TURBINE/NOZZLE FOD", "CLEANED_CORRECTION": "Replaced faulty throttle actuator due to turbine/nozzle FOD. Performed road test post-repair."}),

(17695614, 1490, "25022", "1726", "234.0,4600.0", "ISX15 2019", "ELTS", "1032933", "TURBINE/NOZZLE FOD", "Replaced faulty turbocharger speed sensor due to turbine/nozzle FOD. Performed functional test post-repair.", {"OEM_CODE": 1490, "DEALER": "25022", "DISTR" : "1726", "FC" : "234.0,4600.0", "ENGINE_NAME_DESC": "ISX15 2019", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032933", "FAILURE_MODE_BUCKET": "TURBINE/NOZZLE FOD", "CLEANED_CORRECTION": "Replaced faulty turbocharger speed sensor due to turbine/nozzle FOD. Performed functional test post-repair."}),

(17695615, 1491, "25023", "1727", "901.0,4601.0", "X15 3 2020", "THAC", "1032934", "TURBINE/NOZZLE FOD", "Replaced faulty engine control module due to turbine/nozzle FOD. Completed stationary regen post-repair.", {"OEM_CODE": 1491, "DEALER": "25023", "DISTR" : "1727", "FC" : "901.0,4601.0", "ENGINE_NAME_DESC": "X15 3 2020", "FAILCODE": "THAC", "SHOPORDERNUM": "1032934", "FAILURE_MODE_BUCKET": "TURBINE/NOZZLE FOD", "CLEANED_CORRECTION": "Replaced faulty engine control module due to turbine/nozzle FOD. Completed stationary regen post-repair."}),

(17695616, 1492, "25024", "1728", "456.0,4602.0", "ISX12 2019", "ELTS", "1032935", "TURBINE/NOZZLE FOD", "Replaced faulty turbo speed sensor due to turbine/nozzle FOD. Performed road test post-repair.", {"OEM_CODE": 1492, "DEALER": "25024", "DISTR" : "1728", "FC" : "456.0,4602.0", "ENGINE_NAME_DESC": "ISX12 2019", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032935", "FAILURE_MODE_BUCKET": "TURBINE/NOZZLE FOD", "CLEANED_CORRECTION": "Replaced faulty turbo speed sensor due to turbine/nozzle FOD. Performed road test post-repair."}),

(17695623, 1499, "25031", "1735", "789.0,4608.0", "X15 3 2018", "THAC", "1032942", "ACT(E) CONSEQUENTIAL DAMAGE", "Replaced faulty throttle actuator due to consequential damage. Performed road test post-repair.", {"OEM_CODE": 1499, "DEALER": "25031", "DISTR" : "1735", "FC" : "789.0,4608.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "THAC", "SHOPORDERNUM": "1032942", "FAILURE_MODE_BUCKET": "ACT(E) CONSEQUENTIAL DAMAGE", "CLEANED_CORRECTION": "Replaced faulty throttle actuator due to consequential damage. Performed road test post-repair."}),

(17695628, 1504, "25036", "1740", "345.0,4613.0", "ISX12 2019", "ELTS", "1032947", "ACT(E) CONSEQUENTIAL DAMAGE", "Replaced faulty engine speed sensor due to consequential damage. Completed stationary regen post-repair.", {"OEM_CODE": 1504, "DEALER": "25036", "DISTR" : "1740", "FC" : "345.0,4613.0", "ENGINE_NAME_DESC": "ISX12 2019", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032947", "FAILURE_MODE_BUCKET": "ACT(E) CONSEQUENTIAL DAMAGE", "CLEANED_CORRECTION": "Replaced faulty engine speed sensor due to consequential damage. Completed stationary regen post-repair."}),

(17695629, 1505, "25037", "1741", "901.0,4614.0", "X15 3 2020", "THAC", "1032948", "ACT(E) CONSEQUENTIAL DAMAGE", "Replaced faulty turbocharger speed sensor due to consequential damage. Performed functional test post-repair.", {"OEM_CODE": 1505, "DEALER": "25037", "DISTR" : "1741", "FC" : "901.0,4614.0", "ENGINE_NAME_DESC": "X15 3 2020", "FAILCODE": "THAC", "SHOPORDERNUM": "1032948", "FAILURE_MODE_BUCKET": "ACT(E) CONSEQUENTIAL DAMAGE", "CLEANED_CORRECTION": "Replaced faulty turbocharger speed sensor due to consequential damage. Performed functional test post-repair."}),

(17695630, 1506, "25038", "1742", "234.0,4615.0", "ISX15 2019", "ELTS", "1032949", "ACT(E) CONSEQUENTIAL DAMAGE", "Replaced faulty engine control module due to consequential damage. Completed stationary regen post-repair.", {"OEM_CODE": 1506, "DEALER": "25038", "DISTR" : "1742", "FC" : "234.0,4615.0", "ENGINE_NAME_DESC": "ISX15 2019", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032949", "FAILURE_MODE_BUCKET": "ACT(E) CONSEQUENTIAL DAMAGE", "CLEANED_CORRECTION": "Replaced faulty engine control module due to consequential damage. Completed stationary regen post-repair."}),

(17695631, 1507, "25039", "1743", "456.0,4616.0", "X15 3 2018", "THAC", "1032950", "ACT(E) CONSEQUENTIAL DAMAGE", "Replaced faulty throttle actuator due to consequential damage. Performed road test post-repair.", {"OEM_CODE": 1507, "DEALER": "25039", "DISTR" : "1743", "FC" : "456.0,4616.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "THAC", "SHOPORDERNUM": "1032950", "FAILURE_MODE_BUCKET": "ACT(E) CONSEQUENTIAL DAMAGE", "CLEANED_CORRECTION": "Replaced faulty throttle actuator due to consequential damage. Performed road test post-repair."}),

(17695624, 1500, "25032", "1736", "890.0,4609.0", "ISX12 2020", "THAS", "1032943", "ACT(E) BROKEN SOLDER JOINT", "Replaced faulty turbocharger speed sensor due to broken solder joint. Performed functional test post-repair.", {"OEM_CODE": 1500, "DEALER": "25032", "DISTR" : "1736", "FC" : "890.0,4609.0", "ENGINE_NAME_DESC": "ISX12 2020", "FAILCODE": "THAS", "SHOPORDERNUM": "1032943", "FAILURE_MODE_BUCKET": "ACT(E) BROKEN SOLDER JOINT", "CLEANED_CORRECTION": "Replaced faulty turbocharger speed sensor due to broken solder joint. Performed functional test post-repair."}),

(17695629, 1505, "25037", "1741", "456.0,4614.0", "X15 3 2018", "THAC", "1032948", "ACT(E) BROKEN SOLDER JOINT", "Replaced faulty throttle actuator due to broken solder joint. Performed road test post-repair.", {"OEM_CODE": 1505, "DEALER": "25037", "DISTR" : "1741", "FC" : "456.0,4614.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "THAC", "SHOPORDERNUM": "1032948", "FAILURE_MODE_BUCKET": "ACT(E) BROKEN SOLDER JOINT", "CLEANED_CORRECTION": "Replaced faulty throttle actuator due to broken solder joint. Performed road test post-repair."}),

(17695630, 1506, "25038", "1742", "234.0,4615.0", "ISX15 2019", "THAS", "1032949", "ACT(E) BROKEN SOLDER JOINT", "Replaced faulty engine control module due to broken solder joint. Completed stationary regen post-repair.", {"OEM_CODE": 1506, "DEALER": "25038", "DISTR" : "1742", "FC" : "234.0,4615.0", "ENGINE_NAME_DESC": "ISX15 2019", "FAILCODE": "THAS", "SHOPORDERNUM": "1032949", "FAILURE_MODE_BUCKET": "ACT(E) BROKEN SOLDER JOINT", "CLEANED_CORRECTION": "Replaced faulty engine control module due to broken solder joint. Completed stationary regen post-repair."}),

(17695631, 1507, "25039", "1743", "901.0,4616.0", "X15 3 2020", "THAC", "1032950", "ACT(E) BROKEN SOLDER JOINT", "Replaced faulty turbo speed sensor due to broken solder joint. Performed stationary regen and road test.", {"OEM_CODE": 1507, "DEALER": "25039", "DISTR" : "1743", "FC" : "901.0,4616.0", "ENGINE_NAME_DESC": "X15 3 2020", "FAILCODE": "THAC", "SHOPORDERNUM": "1032950", "FAILURE_MODE_BUCKET": "ACT(E) BROKEN SOLDER JOINT", "CLEANED_CORRECTION": "Replaced faulty turbo speed sensor due to broken solder joint. Performed stationary regen and road test."}),

(17695632, 1508, "25040", "1744", "567.0,4617.0", "ISX12 2019", "THAS", "1032951", "ACT(E) BROKEN SOLDER JOINT", "Replaced faulty engine speed sensor due to broken solder joint. Completed functional test post-repair.", {"OEM_CODE": 1508, "DEALER": "25040", "DISTR" : "1744", "FC" : "567.0,4617.0", "ENGINE_NAME_DESC": "ISX12 2019", "FAILCODE": "THAS", "SHOPORDERNUM": "1032951", "FAILURE_MODE_BUCKET": "ACT(E) BROKEN SOLDER JOINT", "CLEANED_CORRECTION": "Replaced faulty engine speed sensor due to broken solder joint. Completed functional test post-repair."}),

(17695625, 1501, "25033", "1737", "901.0,4610.0", "X15 3 2017", "ELTS", "1032944", "OIL CONTAMINATION", "Replaced faulty turbo speed sensor due to oil contamination. Performed stationary regen and road test.", {"OEM_CODE": 1501, "DEALER": "25033", "DISTR" : "1737", "FC" : "901.0,4610.0", "ENGINE_NAME_DESC": "X15 3 2017", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032944", "FAILURE_MODE_BUCKET": "OIL CONTAMINATION", "CLEANED_CORRECTION": "Replaced faulty turbo speed sensor due to oil contamination. Performed stationary regen and road test."}),

(17695630, 1506, "25038", "1742", "567.0,4615.0", "ISX12 2020", "THAS", "1032949", "OIL CONTAMINATION", "Replaced faulty turbocharger speed sensor due to oil contamination. Performed functional test post-repair.", {"OEM_CODE": 1506, "DEALER": "25038", "DISTR" : "1742", "FC" : "567.0,4615.0", "ENGINE_NAME_DESC": "ISX12 2020", "FAILCODE": "THAS", "SHOPORDERNUM": "1032949", "FAILURE_MODE_BUCKET": "OIL CONTAMINATION", "CLEANED_CORRECTION": "Replaced faulty turbocharger speed sensor due to oil contamination. Performed functional test post-repair."}),

(17695631, 1507, "25039", "1743", "234.0,4616.0", "X15 3 2018", "ELTS", "1032950", "OIL CONTAMINATION", "Replaced faulty throttle actuator due to oil contamination. Performed road test post-repair.", {"OEM_CODE": 1507, "DEALER": "25039", "DISTR" : "1743", "FC" : "234.0,4616.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032950", "FAILURE_MODE_BUCKET": "OIL CONTAMINATION", "CLEANED_CORRECTION": "Replaced faulty throttle actuator due to oil contamination. Performed road test post-repair."}),

(17695632, 1508, "25040", "1744", "789.0,4617.0", "ISX15 2019", "THAS", "1032951", "OIL CONTAMINATION", "Replaced faulty engine control module due to oil contamination. Completed stationary regen post-repair.", {"OEM_CODE": 1508, "DEALER": "25040", "DISTR" : "1744", "FC" : "789.0,4617.0", "ENGINE_NAME_DESC": "ISX15 2019", "FAILCODE": "THAS", "SHOPORDERNUM": "1032951", "FAILURE_MODE_BUCKET": "OIL CONTAMINATION", "CLEANED_CORRECTION": "Replaced faulty engine control module due to oil contamination. Completed stationary regen post-repair."}),

(17695633, 1509, "25041", "1745", "456.0,4618.0", "X15 3 2020", "ELTS", "1032952", "OIL CONTAMINATION", "Replaced faulty turbo speed sensor due to oil contamination. Performed stationary regen and road test.", {"OEM_CODE": 1509, "DEALER": "25041", "DISTR" : "1745", "FC" : "456.0,4618.0", "ENGINE_NAME_DESC": "X15 3 2020", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032952", "FAILURE_MODE_BUCKET": "OIL CONTAMINATION", "CLEANED_CORRECTION": "Replaced faulty turbo speed sensor due to oil contamination. Performed stationary regen and road test."}),

(17695638, 1514, "25046", "1750", "456.0,4623.0", "X15 3 2020", "THAC", "1032957", "TSS - BODY FAILURE", "Replaced faulty throttle actuator due to TSS body failure. Performed road test post-repair.", {"OEM_CODE": 1514, "DEALER": "25046", "DISTR" : "1750", "FC" : "456.0,4623.0", "ENGINE_NAME_DESC": "X15 3 2020", "FAILCODE": "THAC", "SHOPORDERNUM": "1032957", "FAILURE_MODE_BUCKET": "TSS - BODY FAILURE", "CLEANED_CORRECTION": "Replaced faulty throttle actuator due to TSS body failure. Performed road test post-repair."}),

(17695639, 1515, "25047", "1751", "234.0,4624.0", "ISX12 2019", "THAS", "1032958", "TSS - BODY FAILURE", "Replaced faulty engine speed sensor due to TSS body failure. Completed stationary regen post-repair.", {"OEM_CODE": 1515, "DEALER": "25047", "DISTR" : "1751", "FC" : "234.0,4624.0", "ENGINE_NAME_DESC": "ISX12 2019", "FAILCODE": "THAS", "SHOPORDERNUM": "1032958", "FAILURE_MODE_BUCKET": "TSS - BODY FAILURE", "CLEANED_CORRECTION": "Replaced faulty engine speed sensor due to TSS body failure. Completed stationary regen post-repair."}),

(17695640, 1516, "25048", "1752", "789.0,4625.0", "X15 3 2018", "THAC", "1032959", "TSS - BODY FAILURE", "Replaced faulty turbo speed sensor due to TSS body failure. Performed stationary regen and road test.", {"OEM_CODE": 1516, "DEALER": "25048", "DISTR" : "1752", "FC" : "789.0,4625.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "THAC", "SHOPORDERNUM": "1032959", "FAILURE_MODE_BUCKET": "TSS - BODY FAILURE", "CLEANED_CORRECTION": "Replaced faulty turbo speed sensor due to TSS body failure. Performed stationary regen and road test."}),

(17695641, 1517, "25049", "1753", "901.0,4626.0", "ISX15 2020", "THAS", "1032960", "TSS - BODY FAILURE", "Replaced faulty turbocharger speed sensor due to TSS body failure. Performed functional test post-repair.", {"OEM_CODE": 1517, "DEALER": "25049", "DISTR" : "1753", "FC" : "901.0,4626.0", "ENGINE_NAME_DESC": "ISX15 2020", "FAILCODE": "THAS", "SHOPORDERNUM": "1032960", "FAILURE_MODE_BUCKET": "TSS - BODY FAILURE", "CLEANED_CORRECTION": "Replaced faulty turbocharger speed sensor due to TSS body failure. Performed functional test post-repair."}),

(17695634, 1510, "25042", "1746", "901.0,4619.0", "ISX12 2019", "ELTS", "1032953", "ACT(E) SUPPLIER MANUFACTURING FAULT", "Replaced faulty engine speed sensor due to ACT(E) supplier manufacturing fault. Completed stationary regen post-repair.", {"OEM_CODE": 1510, "DEALER": "25042", "DISTR" : "1746", "FC" : "901.0,4619.0", "ENGINE_NAME_DESC": "ISX12 2019", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032953", "FAILURE_MODE_BUCKET": "ACT(E) SUPPLIER MANUFACTURING FAULT", "CLEANED_CORRECTION": "Replaced faulty engine speed sensor due to ACT(E) supplier manufacturing fault. Completed stationary regen post-repair."}),

(17695639, 1515, "25047", "1751", "567.0,4624.0", "ISX12 2020", "THAS", "1032958", "ACT(E) SUPPLIER MANUFACTURING FAULT", "Replaced faulty turbocharger speed sensor due to ACT(E) supplier manufacturing fault. Performed functional test post-repair.", {"OEM_CODE": 1515, "DEALER": "25047", "DISTR" : "1751", "FC" : "567.0,4624.0", "ENGINE_NAME_DESC": "ISX12 2020", "FAILCODE": "THAS", "SHOPORDERNUM": "1032958", "FAILURE_MODE_BUCKET": "ACT(E) SUPPLIER MANUFACTURING FAULT", "CLEANED_CORRECTION": "Replaced faulty turbocharger speed sensor due to ACT(E) supplier manufacturing fault. Performed functional test post-repair."}),

(17695640, 1516, "25048", "1752", "234.0,4625.0", "X15 3 2018", "ELTS", "1032959", "ACT(E) SUPPLIER MANUFACTURING FAULT", "Replaced faulty throttle actuator due to ACT(E) supplier manufacturing fault. Performed road test post-repair.", {"OEM_CODE": 1516, "DEALER": "25048", "DISTR" : "1752", "FC" : "234.0,4625.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032959", "FAILURE_MODE_BUCKET": "ACT(E) SUPPLIER MANUFACTURING FAULT", "CLEANED_CORRECTION": "Replaced faulty throttle actuator due to ACT(E) supplier manufacturing fault. Performed road test post-repair."}),

(17695641, 1517, "25049", "1753", "789.0,4626.0", "ISX15 2019", "THAS", "1032960", "ACT(E) SUPPLIER MANUFACTURING FAULT", "Replaced faulty engine control module due to ACT(E) supplier manufacturing fault. Completed stationary regen post-repair.", {"OEM_CODE": 1517, "DEALER": "25049", "DISTR" : "1753", "FC" : "789.0,4626.0", "ENGINE_NAME_DESC": "ISX15 2019", "FAILCODE": "THAS", "SHOPORDERNUM": "1032960", "FAILURE_MODE_BUCKET": "ACT(E) SUPPLIER MANUFACTURING FAULT", "CLEANED_CORRECTION": "Replaced faulty engine control module due to ACT(E) supplier manufacturing fault. Completed stationary regen post-repair."}),

(17695642, 1518, "25050", "1754", "456.0,4627.0", "X15 3 2020", "ELTS", "1032961", "ACT(E) SUPPLIER MANUFACTURING FAULT", "Replaced faulty turbo speed sensor due to ACT(E) supplier manufacturing fault. Performed stationary regen and road test.", {"OEM_CODE": 1518, "DEALER": "25050", "DISTR" : "1754", "FC" : "456.0,4627.0", "ENGINE_NAME_DESC": "X15 3 2020", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032961", "FAILURE_MODE_BUCKET": "ACT(E) SUPPLIER MANUFACTURING FAULT", "CLEANED_CORRECTION": "Replaced faulty turbo speed sensor due to ACT(E) supplier manufacturing fault. Performed stationary regen and road test."}),

(17695635, 1511, "25043", "1747", "123.0,4620.0", "X15 3 2018", "THAC", "1032954", "ACT(E) CORROSION OF HOUSING AT CONNECTOR", "Replaced faulty throttle actuator due to ACT(E) corrosion of housing at connector. Performed road test post-repair.", {"OEM_CODE": 1511, "DEALER": "25043", "DISTR" : "1747", "FC" : "123.0,4620.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "THAC", "SHOPORDERNUM": "1032954", "FAILURE_MODE_BUCKET": "ACT(E) CORROSION OF HOUSING AT CONNECTOR", "CLEANED_CORRECTION": "Replaced faulty throttle actuator due to ACT(E) corrosion of housing at connector. Performed road test post-repair."}),

(17695640, 1516, "25048", "1752", "678.0,4625.0", "X15 3 2019", "ELTS", "1032959", "ACT(E) CORROSION OF HOUSING AT CONNECTOR", "Replaced faulty engine speed sensor due to ACT(E) corrosion of housing at connector. Completed stationary regen post-repair.", {"OEM_CODE": 1516, "DEALER": "25048", "DISTR" : "1752", "FC" : "678.0,4625.0", "ENGINE_NAME_DESC": "X15 3 2019", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032959", "FAILURE_MODE_BUCKET": "ACT(E) CORROSION OF HOUSING AT CONNECTOR", "CLEANED_CORRECTION": "Replaced faulty engine speed sensor due to ACT(E) corrosion of housing at connector. Completed stationary regen post-repair."}),

(17695641, 1517, "25049", "1753", "234.0,4626.0", "ISX12 2020", "THAS", "1032960", "ACT(E) CORROSION OF HOUSING AT CONNECTOR", "Replaced faulty turbocharger speed sensor due to ACT(E) corrosion of housing at connector. Performed functional test post-repair.", {"OEM_CODE": 1517, "DEALER": "25049", "DISTR" : "1753", "FC" : "234.0,4626.0", "ENGINE_NAME_DESC": "ISX12 2020", "FAILCODE": "THAS", "SHOPORDERNUM": "1032960", "FAILURE_MODE_BUCKET": "ACT(E) CORROSION OF HOUSING AT CONNECTOR", "CLEANED_CORRECTION": "Replaced faulty turbocharger speed sensor due to ACT(E) corrosion of housing at connector. Performed functional test post-repair."}),

(17695642, 1518, "25050", "1754", "789.0,4627.0", "X15 3 2018", "THAC", "1032961", "ACT(E) CORROSION OF HOUSING AT CONNECTOR", "Replaced faulty throttle actuator due to ACT(E) corrosion of housing at connector. Performed road test post-repair.", {"OEM_CODE": 1518, "DEALER": "25050", "DISTR" : "1754", "FC" : "789.0,4627.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "THAC", "SHOPORDERNUM": "1032961", "FAILURE_MODE_BUCKET": "ACT(E) CORROSION OF HOUSING AT CONNECTOR", "CLEANED_CORRECTION": "Replaced faulty throttle actuator due to ACT(E) corrosion of housing at connector. Performed road test post-repair."}),

(17695643, 1519, "25051", "1755", "456.0,4628.0", "ISX15 2019", "ELTS", "1032962", "ACT(E) CORROSION OF HOUSING AT CONNECTOR", "Replaced faulty engine control module due to ACT(E) corrosion of housing at connector. Completed stationary regen post-repair.", {"OEM_CODE": 1519, "DEALER": "25051", "DISTR" : "1755", "FC" : "456.0,4628.0", "ENGINE_NAME_DESC": "ISX15 2019", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032962", "FAILURE_MODE_BUCKET": "ACT(E) CORROSION OF HOUSING AT CONNECTOR", "CLEANED_CORRECTION": "Replaced faulty engine control module due to ACT(E) corrosion of housing at connector. Completed stationary regen post-repair."}),

(17695636, 1512, "25044", "1748", "234.0,4621.0", "ISX12 2020", "THAS", "1032955", "NOZZLE RING ASSY RIVET(S) BROKEN/WORN", "Replaced faulty turbocharger speed sensor due to nozzle ring assy rivet(s) broken/worn. Performed functional test post-repair.", {"OEM_CODE": 1512, "DEALER": "25044", "DISTR" : "1748", "FC" : "234.0,4621.0", "ENGINE_NAME_DESC": "ISX12 2020", "FAILCODE": "THAS", "SHOPORDERNUM": "1032955", "FAILURE_MODE_BUCKET": "NOZZLE RING ASSY RIVET(S) BROKEN/WORN", "CLEANED_CORRECTION": "Replaced faulty turbocharger speed sensor due to nozzle ring assy rivet(s) broken/worn. Performed functional test post-repair."}),

(17695641, 1517, "25049", "1753", "789.0,4626.0", "X15 3 2018", "THAC", "1032960", "NOZZLE RING ASSY RIVET(S) BROKEN/WORN", "Replaced faulty throttle actuator due to nozzle ring assy rivet(s) broken/worn. Performed road test post-repair.", {"OEM_CODE": 1517, "DEALER": "25049", "DISTR" : "1753", "FC" : "789.0,4626.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "THAC", "SHOPORDERNUM": "1032960", "FAILURE_MODE_BUCKET": "NOZZLE RING ASSY RIVET(S) BROKEN/WORN", "CLEANED_CORRECTION": "Replaced faulty throttle actuator due to nozzle ring assy rivet(s) broken/worn. Performed road test post-repair."}),

(17695642, 1518, "25050", "1754", "456.0,4627.0", "ISX15 2019", "THAS", "1032961", "NOZZLE RING ASSY RIVET(S) BROKEN/WORN", "Replaced faulty engine control module due to nozzle ring assy rivet(s) broken/worn. Completed stationary regen post-repair.", {"OEM_CODE": 1518, "DEALER": "25050", "DISTR" : "1754", "FC" : "456.0,4627.0", "ENGINE_NAME_DESC": "ISX15 2019", "FAILCODE": "THAS", "SHOPORDERNUM": "1032961", "FAILURE_MODE_BUCKET": "NOZZLE RING ASSY RIVET(S) BROKEN/WORN", "CLEANED_CORRECTION": "Replaced faulty engine control module due to nozzle ring assy rivet(s) broken/worn. Completed stationary regen post-repair."}),


(17695643, 1519, "25051", "1755", "901.0,4628.0", "X15 3 2020", "THAC", "1032962", "NOZZLE RING ASSY RIVET(S) BROKEN/WORN", "Replaced faulty turbo speed sensor due to nozzle ring assy rivet(s) broken/worn. Performed stationary regen and road test.", {"OEM_CODE": 1519, "DEALER": "25051", "DISTR" : "1755", "FC" : "901.0,4628.0", "ENGINE_NAME_DESC": "X15 3 2020", "FAILCODE": "THAC", "SHOPORDERNUM": "1032962", "FAILURE_MODE_BUCKET": "NOZZLE RING ASSY RIVET(S) BROKEN/WORN", "CLEANED_CORRECTION": "Replaced faulty turbo speed sensor due to nozzle ring assy rivet(s) broken/worn. Performed stationary regen and road test."}),

(17695644, 1520, "25052", "1756", "123.0,4629.0", "ISX12 2019", "THAS", "1032963", "NOZZLE RING ASSY RIVET(S) BROKEN/WORN", "Replaced faulty engine speed sensor due to nozzle ring assy rivet(s) broken/worn. Completed functional test post-repair.", {"OEM_CODE": 1520, "DEALER": "25052", "DISTR" : "1756", "FC" : "123.0,4629.0", "ENGINE_NAME_DESC": "ISX12 2019", "FAILCODE": "THAS", "SHOPORDERNUM": "1032963", "FAILURE_MODE_BUCKET": "NOZZLE RING ASSY RIVET(S) BROKEN/WORN", "CLEANED_CORRECTION": "Replaced faulty engine speed sensor due to nozzle ring assy rivet(s) broken/worn. Completed functional test post-repair."}),

(17695637, 1513, "25045", "1749", "345.0,4622.0", "X15 3 2017", "ELTS", "1032956", "ACT(E) WAITING SUPPLIER ANALYSIS", "Waiting for supplier analysis of faulty turbo speed sensor due to ACT(E) waiting supplier analysis. Performed stationary regen and road test.", {"OEM_CODE": 1513, "DEALER": "25045", "DISTR" : "1749", "FC" : "345.0,4622.0", "ENGINE_NAME_DESC": "X15 3 2017", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032956", "FAILURE_MODE_BUCKET": "ACT(E) WAITING SUPPLIER ANALYSIS", "CLEANED_CORRECTION": "Waiting for supplier analysis of faulty turbo speed sensor due to ACT(E) waiting supplier analysis. Performed stationary regen and road test."}),

(17695642, 1518, "25050", "1754", "890.0,4627.0", "ISX12 2020", "THAS", "1032961", "ACT(E) WAITING SUPPLIER ANALYSIS", "Waiting for supplier analysis of faulty turbocharger assembly due to ACT(E) waiting supplier analysis. Performed functional test post-repair.", {"OEM_CODE": 1518, "DEALER": "25050", "DISTR" : "1754", "FC" : "890.0,4627.0", "ENGINE_NAME_DESC": "ISX12 2020", "FAILCODE": "THAS", "SHOPORDERNUM": "1032961", "FAILURE_MODE_BUCKET": "ACT(E) WAITING SUPPLIER ANALYSIS", "CLEANED_CORRECTION": "Waiting for supplier analysis of faulty turbocharger assembly due to ACT(E) waiting supplier analysis. Performed functional test post-repair."}),

(17695643, 1519, "25051", "1755", "234.0,4628.0", "X15 3 2018", "ELTS", "1032962", "ACT(E) WAITING SUPPLIER ANALYSIS", "Waiting for supplier analysis of faulty throttle actuator due to ACT(E) waiting supplier analysis. Performed road test post-repair.", {"OEM_CODE": 1519, "DEALER": "25051", "DISTR" : "1755", "FC" : "234.0,4628.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032962", "FAILURE_MODE_BUCKET": "ACT(E) WAITING SUPPLIER ANALYSIS", "CLEANED_CORRECTION": "Waiting for supplier analysis of faulty throttle actuator due to ACT(E) waiting supplier analysis. Performed road test post-repair."}),

(17695644, 1520, "25052", "1756", "456.0,4629.0", "ISX15 2019", "THAS", "1032963", "ACT(E) WAITING SUPPLIER ANALYSIS", "Waiting for supplier analysis of faulty engine control module due to ACT(E) waiting supplier analysis. Completed stationary regen post-repair.", {"OEM_CODE": 1520, "DEALER": "25052", "DISTR" : "1756", "FC" : "456.0,4629.0", "ENGINE_NAME_DESC": "ISX15 2019", "FAILCODE": "THAS", "SHOPORDERNUM": "1032963", "FAILURE_MODE_BUCKET": "ACT(E) WAITING SUPPLIER ANALYSIS", "CLEANED_CORRECTION": "Waiting for supplier analysis of faulty engine control module due to ACT(E) waiting supplier analysis. Completed stationary regen post-repair."}),

(17695645, 1521, "25053", "1757", "901.0,4630.0", "X15 3 2020", "ELTS", "1032964", "ACT(E) WAITING SUPPLIER ANALYSIS", "Waiting for supplier analysis of faulty turbo speed sensor due to ACT(E) waiting supplier analysis. Performed stationary regen and road test.", {"OEM_CODE": 1521, "DEALER": "25053", "DISTR" : "1757", "FC" : "901.0,4630.0", "ENGINE_NAME_DESC": "X15 3 2020", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032964", "FAILURE_MODE_BUCKET": "ACT(E) WAITING SUPPLIER ANALYSIS", "CLEANED_CORRECTION": "Waiting for supplier analysis of faulty turbo speed sensor due to ACT(E) waiting supplier analysis. Performed stationary regen and road test."}),

(17695644, 1520, "25052", "1756", "123.0,4629.0", "X15 3 2020", "THAS", "1032963", "TSS - CABLE FRETTING", "Replaced faulty turbocharger assembly due to TSS cable fretting. Performed functional test post-repair.", {"OEM_CODE": 1520, "DEALER": "25052", "DISTR" : "1756", "FC" : "123.0,4629.0", "ENGINE_NAME_DESC": "X15 3 2020", "FAILCODE": "THAS", "SHOPORDERNUM": "1032963", "FAILURE_MODE_BUCKET": "TSS - CABLE FRETTING", "CLEANED_CORRECTION": "Replaced faulty turbocharger assembly due to TSS cable fretting. Performed functional test post-repair."}),

(17695649, 1525, "25057", "1761", "678.0,4634.0", "X15 3 2020", "THAC", "1032968", "TSS - CABLE FRETTING", "Replaced faulty throttle actuator due to TSS cable fretting. Performed road test post-repair.", {"OEM_CODE": 1525, "DEALER": "25057", "DISTR" : "1761", "FC" : "678.0,4634.0", "ENGINE_NAME_DESC": "X15 3 2020", "FAILCODE": "THAC", "SHOPORDERNUM": "1032968", "FAILURE_MODE_BUCKET": "TSS - CABLE FRETTING", "CLEANED_CORRECTION": "Replaced faulty throttle actuator due to TSS cable fretting. Performed road test post-repair."}),

(17695650, 1526, "25058", "1762", "234.0,4635.0", "ISX12 2019", "THAS", "1032969", "TSS - CABLE FRETTING", "Replaced faulty engine speed sensor due to TSS cable fretting. Completed functional test post-repair.", {"OEM_CODE": 1526, "DEALER": "25058", "DISTR" : "1762", "FC" : "234.0,4635.0", "ENGINE_NAME_DESC": "ISX12 2019", "FAILCODE": "THAS", "SHOPORDERNUM": "1032969", "FAILURE_MODE_BUCKET": "TSS - CABLE FRETTING", "CLEANED_CORRECTION": "Replaced faulty engine speed sensor due to TSS cable fretting. Completed functional test post-repair."}),

(17695651, 1527, "25059", "1763", "456.0,4636.0", "X15 3 2018", "THAC", "1032970", "TSS - CABLE FRETTING", "Replaced faulty turbo speed sensor due to TSS cable fretting. Performed stationary regen and road test.", {"OEM_CODE": 1527, "DEALER": "25059", "DISTR" : "1763", "FC" : "456.0,4636.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "THAC", "SHOPORDERNUM": "1032970", "FAILURE_MODE_BUCKET": "TSS - CABLE FRETTING", "CLEANED_CORRECTION": "Replaced faulty turbo speed sensor due to TSS cable fretting. Performed stationary regen and road test."}),

(17695652, 1528, "25060", "1764", "789.0,4637.0", "ISX15 2020", "THAS", "1032971", "TSS - CABLE FRETTING", "Replaced faulty engine control module due to TSS cable fretting. Completed stationary regen post-repair.", {"OEM_CODE": 1528, "DEALER": "25060", "DISTR" : "1764", "FC" : "789.0,4637.0", "ENGINE_NAME_DESC": "ISX15 2020", "FAILCODE": "THAS", "SHOPORDERNUM": "1032971", "FAILURE_MODE_BUCKET": "TSS - CABLE FRETTING", "CLEANED_CORRECTION": "Replaced faulty engine control module due to TSS cable fretting. Completed stationary regen post-repair."}),

(17695645, 1521, "25053", "1757", "234.0,4630.0", "ISX12 2019", "ELTS", "1032964", "TSS - BODY CRACKED/BROKEN", "Replaced faulty engine speed sensor due to TSS body cracked/broken. Completed stationary regen post-repair.", {"OEM_CODE": 1521, "DEALER": "25053", "DISTR" : "1757", "FC" : "234.0,4630.0", "ENGINE_NAME_DESC": "ISX12 2019", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032964", "FAILURE_MODE_BUCKET": "TSS - BODY CRACKED/BROKEN", "CLEANED_CORRECTION": "Replaced faulty engine speed sensor due to TSS body cracked/broken. Completed stationary regen post-repair."}),

(17695650, 1526, "25058", "1762", "789.0,4635.0", "ISX12 2020", "THAS", "1032969", "TSS - BODY CRACKED/BROKEN", "Replaced faulty turbocharger speed sensor due to TSS body cracked/broken. Performed functional test post-repair.", {"OEM_CODE": 1526, "DEALER": "25058", "DISTR" : "1762", "FC" : "789.0,4635.0", "ENGINE_NAME_DESC": "ISX12 2020", "FAILCODE": "THAS", "SHOPORDERNUM": "1032969", "FAILURE_MODE_BUCKET": "TSS - BODY CRACKED/BROKEN", "CLEANED_CORRECTION": "Replaced faulty turbocharger speed sensor due to TSS body cracked/broken. Performed functional test post-repair."}),

(17695651, 1527, "25059", "1763", "456.0,4636.0", "X15 3 2018", "ELTS", "1032970", "TSS - BODY CRACKED/BROKEN", "Replaced faulty throttle actuator due to TSS body cracked/broken. Performed road test post-repair.", {"OEM_CODE": 1527, "DEALER": "25059", "DISTR" : "1763", "FC" : "456.0,4636.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032970", "FAILURE_MODE_BUCKET": "TSS - BODY CRACKED/BROKEN", "CLEANED_CORRECTION": "Replaced faulty throttle actuator due to TSS body cracked/broken. Performed road test post-repair."}),

(17695652, 1528, "25060", "1764", "901.0,4637.0", "ISX15 2019", "THAS", "1032971", "TSS - BODY CRACKED/BROKEN", "Replaced faulty engine control module due to TSS body cracked/broken. Completed stationary regen post-repair.", {"OEM_CODE": 1528, "DEALER": "25060", "DISTR" : "1764", "FC" : "901.0,4637.0", "ENGINE_NAME_DESC": "ISX15 2019", "FAILCODE": "THAS", "SHOPORDERNUM": "1032971", "FAILURE_MODE_BUCKET": "TSS - BODY CRACKED/BROKEN", "CLEANED_CORRECTION": "Replaced faulty engine control module due to TSS body cracked/broken. Completed stationary regen post-repair."}),

(17695653, 1529, "25061", "1765", "123.0,4638.0", "X15 3 2020", "ELTS", "1032972", "TSS - BODY CRACKED/BROKEN", "Replaced faulty turbo speed sensor due to TSS body cracked/broken. Performed stationary regen and road test.", {"OEM_CODE": 1529, "DEALER": "25061", "DISTR" : "1765", "FC" : "123.0,4638.0", "ENGINE_NAME_DESC": "X15 3 2020", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032972", "FAILURE_MODE_BUCKET": "TSS - BODY CRACKED/BROKEN", "CLEANED_CORRECTION": "Replaced faulty turbo speed sensor due to TSS body cracked/broken. Performed stationary regen and road test."}),

(17695646, 1522, "25054", "1758", "345.0,4631.0", "X15 3 2018", "THAC", "1032965", "TSS - ADJUSTED/DAMAGED BY CUSTOMER", "Replaced faulty throttle actuator due to TSS adjusted/damaged by customer. Performed road test post-repair.", {"OEM_CODE": 1522, "DEALER": "25054", "DISTR" : "1758", "FC" : "345.0,4631.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "THAC", "SHOPORDERNUM": "1032965", "FAILURE_MODE_BUCKET": "TSS - ADJUSTED/DAMAGED BY CUSTOMER", "CLEANED_CORRECTION": "Replaced faulty throttle actuator due to TSS adjusted/damaged by customer. Performed road test post-repair."}),

(17695651, 1527, "25059", "1763", "890.0,4636.0", "X15 3 2019", "ELTS", "1032970", "TSS - ADJUSTED/DAMAGED BY CUSTOMER", "Replaced faulty engine speed sensor due to TSS adjusted/damaged by customer. Completed stationary regen post-repair.", {"OEM_CODE": 1527, "DEALER": "25059", "DISTR" : "1763", "FC" : "890.0,4636.0", "ENGINE_NAME_DESC": "X15 3 2019", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032970", "FAILURE_MODE_BUCKET": "TSS - ADJUSTED/DAMAGED BY CUSTOMER", "CLEANED_CORRECTION": "Replaced faulty engine speed sensor due to TSS adjusted/damaged by customer. Completed stationary regen post-repair."}),

(17695652, 1528, "25060", "1764", "234.0,4637.0", "ISX12 2020", "THAS", "1032971", "TSS - ADJUSTED/DAMAGED BY CUSTOMER", "Replaced faulty turbocharger speed sensor due to TSS adjusted/damaged by customer. Performed functional test post-repair.", {"OEM_CODE": 1528, "DEALER": "25060", "DISTR" : "1764", "FC" : "234.0,4637.0", "ENGINE_NAME_DESC": "ISX12 2020", "FAILCODE": "THAS", "SHOPORDERNUM": "1032971", "FAILURE_MODE_BUCKET": "TSS - ADJUSTED/DAMAGED BY CUSTOMER", "CLEANED_CORRECTION": "Replaced faulty turbocharger speed sensor due to TSS adjusted/damaged by customer. Performed functional test post-repair."}),

(17695653, 1529, "25061", "1765", "456.0,4638.0", "X15 3 2018", "THAC", "1032972", "TSS - ADJUSTED/DAMAGED BY CUSTOMER", "Replaced faulty throttle actuator due to TSS adjusted/damaged by customer. Performed road test post-repair.", {"OEM_CODE": 1529, "DEALER": "25061", "DISTR" : "1765", "FC" : "456.0,4638.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "THAC", "SHOPORDERNUM": "1032972", "FAILURE_MODE_BUCKET": "TSS - ADJUSTED/DAMAGED BY CUSTOMER", "CLEANED_CORRECTION": "Replaced faulty throttle actuator due to TSS adjusted/damaged by customer. Performed road test post-repair."}),

(17695654, 1530, "25062", "1766", "901.0,4639.0", "ISX15 2019", "ELTS", "1032973", "TSS - ADJUSTED/DAMAGED BY CUSTOMER", "Replaced faulty engine control module due to TSS adjusted/damaged by customer. Completed stationary regen post-repair.", {"OEM_CODE": 1530, "DEALER": "25062", "DISTR" : "1766", "FC" : "901.0,4639.0", "ENGINE_NAME_DESC": "ISX15 2019", "FAILCODE": "ELTS", "SHOPORDERNUM": "1032973", "FAILURE_MODE_BUCKET": "TSS - ADJUSTED/DAMAGED BY CUSTOMER", "CLEANED_CORRECTION": "Replaced faulty engine control module due to TSS adjusted/damaged by customer. Completed stationary regen post-repair."}),

(17695647, 1523, "25055", "1759", "456.0,4632.0", "ISX12 2020", "THAS", "1032966", "ACT(E) SUPPLIER ANALYSIS NOT CONDUCTED", "Waiting for supplier analysis of faulty turbocharger speed sensor due to ACT(E) supplier analysis not conducted. Performed functional test post-repair.", {"OEM_CODE": 1523, "DEALER": "25055", "DISTR" : "1759", "FC" : "456.0,4632.0", "ENGINE_NAME_DESC": "ISX12 2020", "FAILCODE": "THAS", "SHOPORDERNUM": "1032966", "FAILURE_MODE_BUCKET": "ACT(E) SUPPLIER ANALYSIS NOT CONDUCTED", "CLEANED_CORRECTION": "Waiting for supplier analysis of faulty turbocharger speed sensor due to ACT(E) supplier analysis not conducted. Performed functional test post-repair."}),

(17695652, 1528, "25060", "1764", "901.0,4637.0", "X15 3 2018", "THAC", "1032971", "ACT(E) SUPPLIER ANALYSIS NOT CONDUCTED", "Waiting for supplier analysis of faulty throttle actuator due to ACT(E) supplier analysis not conducted. Performed road test post-repair.", {"OEM_CODE": 1528, "DEALER": "25060", "DISTR" : "1764", "FC" : "901.0,4637.0", "ENGINE_NAME_DESC": "X15 3 2018", "FAILCODE": "THAC", "SHOPORDERNUM": "1032971", "FAILURE_MODE_BUCKET": "ACT(E) SUPPLIER ANALYSIS NOT CONDUCTED", "CLEANED_CORRECTION": "Waiting for supplier analysis of faulty throttle actuator due to ACT(E) supplier analysis not conducted. Performed road test post-repair."}),

(17695653, 1529, "25061", "1765", "234.0,4638.0", "ISX15 2019", "THAS", "1032972", "ACT(E) SUPPLIER ANALYSIS NOT CONDUCTED", "Waiting for supplier analysis of faulty engine control module due to ACT(E) supplier analysis not conducted. Completed stationary regen post-repair.", {"OEM_CODE": 1529, "DEALER": "25061", "DISTR" : "1765", "FC" : "234.0,4638.0", "ENGINE_NAME_DESC": "ISX15 2019", "FAILCODE": "THAS", "SHOPORDERNUM": "1032972", "FAILURE_MODE_BUCKET": "ACT(E) SUPPLIER ANALYSIS NOT CONDUCTED", "CLEANED_CORRECTION": "Waiting for supplier analysis of faulty engine control module due to ACT(E) supplier analysis not conducted. Completed stationary regen post-repair."}),

(17695654, 1530, "25062", "1766", "567.0,4639.0", "X15 3 2020", "THAC", "1032973", "ACT(E) SUPPLIER ANALYSIS NOT CONDUCTED", "Waiting for supplier analysis of faulty turbo speed sensor due to ACT(E) supplier analysis not conducted. Performed stationary regen and road test.", {"OEM_CODE": 1530, "DEALER": "25062", "DISTR" : "1766", "FC" : "567.0,4639.0", "ENGINE_NAME_DESC": "X15 3 2020", "FAILCODE": "THAC", "SHOPORDERNUM": "1032973", "FAILURE_MODE_BUCKET": "ACT(E) SUPPLIER ANALYSIS NOT CONDUCTED", "CLEANED_CORRECTION": "Waiting for supplier analysis of faulty turbo speed sensor due to ACT(E) supplier analysis not conducted. Performed stationary regen and road test."}),

(17695655, 1531, "25063", "1767", "789.0,4640.0", "ISX12 2019", "THAS", "1032974", "ACT(E) SUPPLIER ANALYSIS NOT CONDUCTED", "Waiting for supplier analysis of faulty engine speed sensor due to ACT(E) supplier analysis not conducted. Completed functional test post-repair.", {"OEM_CODE": 1531, "DEALER": "25063", "DISTR" : "1767", "FC" : "789.0,4640.0", "ENGINE_NAME_DESC": "ISX12 2019", "FAILCODE": "THAS", "SHOPORDERNUM": "1032974", "FAILURE_MODE_BUCKET": "ACT(E) SUPPLIER ANALYSIS NOT CONDUCTED", "CLEANED_CORRECTION": "Waiting for supplier analysis of faulty engine speed sensor due to ACT(E) supplier analysis not conducted. Completed functional test post-repair."})

]

df = spark.createDataFrame(data, schema=schema)

display(df)

# COMMAND ----------


# table_name = f"cmidev.cbu_ccims_poc.ccims_claim_inspection_feature"

# # df =spark.readStream.table(table_name)

# df = spark.read.table(table_name)


# COMMAND ----------

# MAGIC %md
# MAGIC Let's view the content of one of the articles.

# COMMAND ----------

df.count()

# COMMAND ----------

from pyspark.sql.functions import col

# Remove rows having Not Inspected as the failure mode bucket value
df = df.filter(~(col('FAILURE_MODE_BUCKET') == 'Not Inspected'))
df.count()

# COMMAND ----------

# print("Total % of rows having valid failure mode bucket is:",(44702/115972)*100)

# COMMAND ----------

# Function to clean up the extracted text (optional)
import re
def clean_extracted_text(text):
    text = re.sub(r'\n', '', text)
    return re.sub(r' ?\.', '.', text)


def pprint(obj):
  import pprint
  pprint.pprint(obj, compact=True, indent=1, width=100)

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %pip install transformers
# MAGIC
# MAGIC from transformers import AutoTokenizer, AutoModel
# MAGIC import torch

# COMMAND ----------

pprint(df.select("DOCUMENT").show(1))

# COMMAND ----------

def extract_doc_text(df,row_id):
    txt = df.collect()[row_id]
    return txt    

extract_doc_text(df,row_id=0)

# COMMAND ----------

# !pip install llama-index
import io
import os
import pandas as pd 

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.core.utils import set_global_tokenizer
from transformers import AutoTokenizer
from typing import Iterator
from pyspark.sql.functions import col, udf, length, pandas_udf, explode
from unstructured.partition.auto import partition


@pandas_udf("array<string>")
def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    # set llama2 as tokenizer
    set_global_tokenizer(
      AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    )
    # sentence splitter from llama_index to split on sentences
    splitter = SentenceSplitter(chunk_size=100, chunk_overlap=20)
    def extract_and_split(txt):
      # txt = extract_doc_text(b)
      nodes = splitter.get_nodes_from_documents([Document(text=txt)])
      return [n.text for n in nodes]

    for x in batch_iter:
        yield x.apply(extract_and_split)

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Convert STRUCT format of Document into array format

# COMMAND ----------

from pyspark.sql.functions import array, col

# Correct way to create an array from existing columns in PySpark
df_transformed = df.withColumn("DOCUMENT_ARRAY", array(
    col("DOCUMENT.OEM_CODE"),
    col("DOCUMENT.DEALER"),
    col("DOCUMENT.DISTR"),
    col("DOCUMENT.FC"),
    col("DOCUMENT.ENGINE_NAME_DESC"),
    col("DOCUMENT.FAILCODE"),
    col("DOCUMENT.SHOPORDERNUM"),
    col("DOCUMENT.FAILURE_MODE_BUCKET"),
    col("DOCUMENT.CLEANED_CORRECTION")
))

display(df_transformed)

# COMMAND ----------

# MAGIC %md
# MAGIC Perform chunking of each field in document array for each claim

# COMMAND ----------

from pyspark.sql.functions import explode

df_chunks = (df_transformed
                .withColumn("DOCUMENT_CHUNK", explode(df_transformed["DOCUMENT_ARRAY"]))
                .selectExpr('CLAIM_ID_SEQ', 'DOCUMENT_CHUNK')
            )
display(df_chunks.head(36))

# COMMAND ----------

# MAGIC %md
# MAGIC As you can see, for each claim ID, there are 9 chunks created, each chunk is for each field in the array

# COMMAND ----------

from mlflow.deployments import get_deploy_client


# gte-large-en Foundation models are available using the /serving-endpoints/databricks-gte-large-en/invocations api. 
deploy_client = get_deploy_client("databricks")

# NOTE: if you change your embedding model here, make sure you change it in the query step too
embeddings = deploy_client.predict(endpoint="databricks-gte-large-en", inputs={"input": ["What is Apache Spark?"]})
pprint(embeddings)

# COMMAND ----------

from mlflow.exceptions import MlflowException
import logging

# COMMAND ----------

@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    def get_embeddings(batch):
        # NOTE: this will fail if an exception is thrown during embedding creation (add try/except if needed) 
         try:
            # Convert the batch to a list of strings
            string_batch = [str(item) for item in batch]  # Ensure all items are strings
            response = deploy_client.predict(endpoint="databricks-gte-large-en", inputs={"input": string_batch})
            return [e["embedding"] for e in response.data]
         except MlflowException as e:
            logging.error(f"Error occurred while generating embeddings: {e}")
            return [None] * len(batch)  # Return None for failed embeddings
    # splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())

    return pd.Series(all_embeddings)

# COMMAND ----------

df_chunks.columns

# COMMAND ----------

import pyspark.sql.functions as F


df_chunk_emd = (df_chunks
                .withColumn("embedding", get_embedding("DOCUMENT_CHUNK"))
                .selectExpr("CLAIM_ID_SEQ", "DOCUMENT_CHUNK", "embedding")
                )
display(df_chunk_emd.head(10))

# COMMAND ----------

df_chunk_emd.count()

# COMMAND ----------

display(df_chunk_emd)

# COMMAND ----------

df_emd_pd = df_chunk_emd.toPandas()
len(df_emd_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Embeddings to a Delta Table
# MAGIC
# MAGIC Now that the embeddings are ready, let's create a Delta table and store the embeddings in this table.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM cmidev.cbu_ccims_poc.synthetic_data_claim_text_embeddings_gte;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS cmidev.cbu_ccims_poc.synthetic_data_claim_text_embeddings_gte (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   CLAIM_ID_SEQ STRING,
# MAGIC   DOCUMENT_CHUNK STRING,
# MAGIC   embedding ARRAY <FLOAT>
# MAGIC   -- NOTE: the table has to be CDC because VectorSearch is using DLT that is requiring CDC state
# MAGIC   ) TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM cmidev.cbu_ccims_poc.synthetic_data_claim_text_embeddings_gte;

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

# Step 1: Read the existing embeddings table
df_embeddings = spark.sql("select * from cmidev.cbu_ccims_poc.synthetic_data_claim_text_embeddings_gte")

embedding_table_name = f"cmidev.cbu_ccims_poc.synthetic_data_claim_text_embeddings_gte"

# Step 2: Add an "id" column with auto-increment using monotonically_increasing_id
df_with_id = df_embeddings.withColumn("id", monotonically_increasing_id())

# Step 3: Overwrite the table with the new column added
df_with_id.write.mode("overwrite").option("overwriteSchema", "True").saveAsTable(embedding_table_name)


# COMMAND ----------

# MAGIC %md
# MAGIC Batch operation to write data from df_chunk_emd into the embeddings delta table

# COMMAND ----------

embedding_table_name = f"cmidev.cbu_ccims_poc.synthetic_data_claim_text_embeddings_gte"
df_chunk_emd.write.mode("overwrite").option("overwriteSchema", "True").saveAsTable(embedding_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC CONVERT TO DELTA cmidev.cbu_ccims_poc.synthetic_data_claim_text_embeddings_gte
