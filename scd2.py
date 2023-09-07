from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql.functions import hash, lit, concat

sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# SCD2 implemented in PySpark. A record is considered as new/updated if
# at least on of the columns in columns_to_check is changed. Hashing of
# these columns was added, to make comparison of these columns between
# already existing records and new records faster.

columns_to_check = [
    "order_status",
    "payment_status",
    "address",
    "quote",
    "phone_number"
]

# these are mock values, they are normally coming from GlueJob arguments
path = "data/orders/"
target_path = "data/target/scd2/"
LOAD_TYPE = "first_load"
HASH_RELOAD = False


df = spark.read.option("header", "true").parquet(path)

df = (
    df.withColumn("scd_hash", hash(*columns_to_check))
    .withColumn("uq_key", concat(col("id"), col("lastmodifieddate")))
)

if LOAD_TYPE == "first_load":
    df = (
        df.withColumn("scd_create_date", col("createddate"))
        .withColumn("scd_end_date", to_timestamp(lit("9999-12-31 23:59:59")))
        .withColumn("scd_is_active", lit(1))
    )
    df.write.parquet(target_path)

elif LOAD_TYPE == "incremental":
    df_target = spark.read.option("header", "true").parquet(target_path)

    # this parameter should be set to true only once after changing the columns
    # used to calculate hash (since previous hash is no longer relevant)
    if HASH_RELOAD:
        df_target.withColumn("scd_hash", hash(*columns_to_check))
    
    df_inserts = df.join(df_target, on="uq_key", how="left_anti").select(df["*"])
    df_inserts = (
        df_inserts.withColumn("scd_create_date", col("createddate"))
        .withColumn("scd_end_date", to_timestamp(lit("9999-12-31 23:59:59")))
        .withColumn("scd_is_active", lit(1))
    )

    df_updates = (
        df.alias("stg")
        .join(df_target.alias("tgt"), (df_stg["uq_key"]==df_target["uq_key"]), how="inner")
        .where(
            (df_target["scd_is_active"] == 1) & 
            # if hash is different, it means that at least one of the columns used to calculate has has changed
            (df_stg["scd_hash"] != df_target["scd_hash"])
        )
        # adding prefixes to stage and target columns, to differentiate them later
        .select(
            *[col(f"stg.{c}").alias(f"stg_{c}") for c in df_stg.columns]
            + [col(f"tgt.{c}").alias(f"tgt_{c}") for c in df_tgt.columns]
        )
    )

    df_active_updates = (
        df_updates.select(
            # select all the columns with alias stg_ and then remove this alias
            *[col(column).alias(column.replace("stg_", "")) for column in df_updates.columns if "stg_" in column]
        )
        .withColumn("scd_create_date", col("lastmodifieddate"))
        .withColumn("scd_end_date", to_timestamp(lit("9999-12-31 23:59:59")))
        .withColumn("scd_is_active", lit(1))
    )

    df_inactive_updates = (
        df_updates.withColumn("tgt_scd_end_date", col("lastmodifieddate"))
        .select(
        # select all the columns with alias tgt_ and then remove this alias
            *[col(column).alias(column.replace("tgt_", "")) for column in df_updates.columns if "tgt_" in column]
        )
        .withColumn("scd_is_active", lit(0))
    )

    df_upsert = df_inserts.union(df_active_updates).union(df_inactive_updates)
    upsert_count = df_upsert.count()

    # Saving. The best option is to use Hudi, as it allows to do an upsert without 
    # reloading all existing data