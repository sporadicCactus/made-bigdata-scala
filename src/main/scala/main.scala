import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel

import org.apache.spark.sql.SparkSession

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, RandomHyperplanesLSH}

import org.apache.spark.ml.evaluation.RegressionEvaluator


object Homework extends App {

  val spark = SparkSession.builder()
    .master("local[*]")
    .appName("spark-homework")
    .getOrCreate()

  import spark.implicits._

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

  val testFeaturesWithIndex = testFeatures.withColumn("id", monotonicallyIncreasingId()).cache()

  val metrics = new RegressionEvaluator()
    .setLabelCol("Rating")
    .setPredictionCol("predict")
    .setMetricName("rmse")

  val numHashesArray = Array(5, 7, 9)
  val distThresholdsArray = Array(0.7, 0.8, 0.9)
  val results = {for {numHashes <- numHashesArray; distThreshold <- distThresholdsArray} yield (numHashes, distThreshold)}
    .map( pair => {
    val numHashes = pair._1
    val distThreshold = pair._2

    val rhp = new RandomHyperplanesLSH()
        .setInputCol("tfidf")
        .setOutputCol("buckets")
        .setNumHashTables(numHashes)
        .fit(trainFeatures)

    val neighbors = rhp.approxSimilarityJoin(trainFeatures, testFeaturesWithIndex, distThreshold)

    val predictions = neighbors
        .withColumn("similarity", (lit(1) - col("distCol")))
        .groupBy("datasetB.id")
        .agg(
            (sum(col("similarity") * col("datasetA.Rating")) / sum(col("similarity"))).as("predict"),
            count("datasetA.Rating").as("numNeighbors")
        )

    val forMetric = testFeaturesWithIndex.join(predictions, Seq("id"))

    val meanNumNeighbors = forMetric.select(avg("numNeighbors")).collect.head(0)

    val metric = metrics.evaluate(forMetric)

    val res = (numHashes, distThreshold, metric, meanNumNeighbors)
    res
    })
    for {res <- results} println(res) 
}
