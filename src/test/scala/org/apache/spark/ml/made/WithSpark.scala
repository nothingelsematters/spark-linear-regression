package org.apache.spark.ml.made

import org.apache.spark.sql.{SQLContext, SparkSession}

trait WithSpark {
  lazy val sparkSession: SparkSession = WithSpark.sparkSession
  lazy val sqlContext: SQLContext = WithSpark.sqlContext
}

object WithSpark {
  private lazy val sparkSession = SparkSession
    .builder
    .appName("Linear Regression Testing")
    .master("local[4]")
    .getOrCreate()

  private lazy val sqlContext = sparkSession.sqlContext
}
