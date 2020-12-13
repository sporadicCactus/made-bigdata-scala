import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel

import org.apache.spark.sql.SparkSession

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer}

object Homework extends App {
  val df = spark.read.option("inferSchema", "true").option("header", "true").csv("tripadvisor_hotel_reviews.csv")

  val preprocessingPipe = new Pipeline()
    .setStages(Array(
        new RegexTokenizer()
        .setInputCol("Review")
        .setOutputCol("tokenized")
        .setPattern("\\W+"),
        new HashingTF()
            .setInputCol("tokenized")
            .setOutputCol("tf")
            .setBinary(true)
            .setNumFeatures(1000),
        new HashingTF()
            .setInputCol("tokenized")
            .setOutputCol("tf2")
            .setNumFeatures(1000),
        new IDF()
            .setInputCol("tf2")
            .setOutputCol("tfidf")
    ))

  val Array(train, test) = df.randomSplit(Array(0.8, 0.2))


  val pipe = preprocessingPipe.fit(train)

  val trainFeatures = pipe.transform(train).cache()
  val testFeatures = pipe.transform(test)

  trainFeatures.show
}
