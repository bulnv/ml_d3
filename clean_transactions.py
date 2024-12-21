#!/usr/bin/env python3
import sys
import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_s3_credentials():
    """Get S3 credentials from environment variables"""
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    endpoint = os.getenv('S3_ENDPOINT', 'storage.yandexcloud.net')

    if not access_key or not secret_key:
        logger.error("AWS credentials not found in environment variables")
        sys.exit(1)

    return access_key, secret_key, endpoint

def create_spark_session(app_name="TransactionDataCleaning", access_key=None, secret_key=None, endpoint=None):
    """
    Create and configure Spark session for local execution
    """
    try:
        builder = SparkSession.builder \
            .appName(app_name) \
            .config("spark.executor.memory", "10g") \
            .config("spark.executor.cores", "4") \
            .config("spark.executor.instances", "3") \
            .config("spark.executor.memoryOverhead", "2g") \
            .config("spark.driver.memory", "4g") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

        # Add S3 configurations if credentials are provided
        if access_key and secret_key:
            builder = builder \
                .config("spark.hadoop.fs.s3a.access.key", access_key) \
                .config("spark.hadoop.fs.s3a.secret.key", secret_key) \
                .config("spark.hadoop.fs.s3a.endpoint", endpoint) \
                .config("spark.hadoop.fs.s3a.path.style.access", "true") \
                .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true") \
                .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                        "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
                .config("spark.hadoop.fs.s3a.signing-algorithm", "S3SignerType") \
                .config("spark.hadoop.fs.s3a.change.detection.mode", "none") \
                .config("spark.hadoop.fs.s3a.impl.disable.cache", "true") \
                .config("spark.hadoop.fs.s3a.committer.magic.enabled", "true")

        spark = builder.getOrCreate()

        # Set log level after creation
        spark.sparkContext.setLogLevel("WARN")

        logger.info("Spark session created successfully")
        return spark
    except Exception as e:
        logger.error(f"Failed to create Spark session: {str(e)}")
        raise

def load_data(spark, input_path):
    """
    Load data with predefined schema
    """
    try:
        schema = StructType([
            StructField("transaction_id", LongType(), True),
            StructField("tx_datetime", StringType(), True),
            StructField("customer_id", LongType(), True),
            StructField("terminal_id", LongType(), True),
            StructField("tx_amount", DoubleType(), True),
            StructField("tx_time_seconds", LongType(), True),
            StructField("tx_time_days", LongType(), True),
            StructField("tx_fraud", IntegerType(), True),
            StructField("tx_fraud_scenario", IntegerType(), True)
        ])

        df = spark.read.csv(
            input_path,
            schema=schema,
            header=False,
            mode='PERMISSIVE',
            nullValue='null'
        )

        # Cache the DataFrame for better performance
        df.cache()
        count = df.count()  # Force cache computation
        logger.info(f"Data loaded successfully. Row count: {count:,}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {input_path}: {str(e)}")
        raise

def clean_numeric_outliers(df, column, lower_percentile=0.01, upper_percentile=0.99):
    """
    Clean numeric outliers using percentile method with error handling
    """
    try:
        percentiles = df.approxQuantile(column, [lower_percentile, 0.5, upper_percentile], 0.01)
        lower_bound, median, upper_bound = percentiles

        cleaned_df = df.withColumn(
            column,
            F.when(
                (F.col(column) < lower_bound) | (F.col(column) > upper_bound),
                median
            ).otherwise(F.col(column))
        )

        return cleaned_df
    except Exception as e:
        logger.error(f"Error cleaning outliers in column {column}: {str(e)}")
        raise

def clean_dataset(df):
    """
    Clean the dataset based on identified issues
    """
    try:
        logger.info("Starting data cleaning process")

        # 1. Remove complete null rows
        df_cleaned = df.dropna(how='all')
        logger.info(f"Removed complete null rows. Remaining: {df_cleaned.count():,}")

        # 2. Remove duplicate transaction IDs (keep first occurrence)
        df_cleaned = df_cleaned.dropDuplicates(['transaction_id'])
        logger.info(f"Removed duplicate transactions. Remaining: {df_cleaned.count():,}")

        # 3. Clean numeric outliers
        df_cleaned = clean_numeric_outliers(df_cleaned, 'tx_amount')

        # 4. Handle invalid terminal_ids
        df_cleaned = df_cleaned.withColumn(
            'terminal_id',
            F.when(F.col('terminal_id').isNull(), -1).otherwise(F.col('terminal_id'))
        )

        # 5. Validate and clean tx_datetime
        df_cleaned = df_cleaned.withColumn(
            'tx_datetime',
            F.to_timestamp(F.col('tx_datetime'))
        ).filter(F.col('tx_datetime').isNotNull())

        # 6. Ensure tx_fraud and tx_fraud_scenario consistency
        df_cleaned = df_cleaned.withColumn(
            'tx_fraud_scenario',
            F.when(F.col('tx_fraud') == 0, 0).otherwise(F.col('tx_fraud_scenario'))
        )

        # 7. Remove remaining invalid records
        df_cleaned = df_cleaned.filter(
            (F.col('transaction_id').isNotNull()) &
            (F.col('tx_datetime').isNotNull()) &
            (F.col('tx_amount').isNotNull()) &
            (F.col('tx_fraud').isNotNull())
        )

        # Cache the cleaned DataFrame
        df_cleaned.cache()
        cleaned_count = df_cleaned.count()
        logger.info(f"Data cleaning completed. Final count: {cleaned_count:,}")

        return df_cleaned
    except Exception as e:
        logger.error(f"Error during data cleaning: {str(e)}")
        raise

def save_to_parquet(df, output_path):
    """
    Save DataFrame to parquet format with partitioning and error handling
    """
    try:
        # Add year and month columns for partitioning
        df_to_save = df.withColumn(
            "year_month",
            F.date_format(F.col("tx_datetime"), "yyyy-MM")
        )

        # Save with partitioning
        df_to_save.write \
            .partitionBy("year_month") \
            .mode("overwrite") \
            .option("compression", "snappy") \
            .option("maxRecordsPerFile", "1000000") \
            .parquet(output_path+'cleaned.parquet')

        logger.info(f"Data successfully saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving data to {output_path}: {str(e)}")
        raise

def main():
    """
    Main execution function
    """
    if len(sys.argv) != 3:
        logger.error("Invalid arguments. Usage: script.py <input_path> <output_path>")
        logger.error("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    spark = None

    try:
        # Get S3 credentials
        access_key, secret_key, endpoint = get_s3_credentials()

        # Initialize Spark with credentials
        spark = create_spark_session(
            access_key=access_key,
            secret_key=secret_key,
            endpoint=endpoint
        )

        # Load data
        df_original = load_data(spark, input_path)

        # Clean data
        df_cleaned = clean_dataset(df_original)

        # Log metrics
        original_count = df_original.count()
        cleaned_count = df_cleaned.count()
        logger.info(f"Data Quality Metrics:")
        logger.info(f"Original records: {original_count:,}")
        logger.info(f"Cleaned records: {cleaned_count:,}")
        logger.info(f"Removed records: {original_count - cleaned_count:,}")

        # Save cleaned data
        save_to_parquet(df_cleaned, output_path)

        logger.info("Data processing completed successfully")

    except Exception as e:
        logger.error(f"Error during data processing: {str(e)}")
        if spark:
            spark.stop()
        sys.exit(1)
    finally:
        if spark:
            # Uncache DataFrames
            try:
                df_original.unpersist()
                df_cleaned.unpersist()
            except:
                pass
            spark.stop()
            logger.info("Spark session stopped")

if __name__ == "__main__":
    main()
