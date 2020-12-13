package org.apache.spark.ml.feature

import scala.util.Random
import scala.math.{signum, sqrt}

import org.apache.hadoop.fs.Path

import org.apache.spark.annotation.Since
import org.apache.spark.ml.linalg.{Vector, Vectors, VectorUDT}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.HasSeed
import org.apache.spark.ml.util._
import org.apache.spark.sql.types.StructType


@Since("2.1.0")
class RandomHyperplanesLSHModel private[ml](
    override val uid: String,
    private[ml] val randPlanes: Array[Vector])
  extends LSHModel[RandomHyperplanesLSHModel] {

  /** @group setParam */
  @Since("2.4.0")
  override def setInputCol(value: String): this.type = super.set(inputCol, value)

  /** @group setParam */
  @Since("2.4.0")
  override def setOutputCol(value: String): this.type = super.set(outputCol, value)

  @Since("2.1.0")
  override protected[ml] def hashFunction(elems: Vector): Array[Vector] = {
    require(elems.nonZeroIterator.nonEmpty, "Must have at least 1 non zero entry.")
    val hashValues = randPlanes.map { case plane =>
      signum(
        elems.nonZeroIterator.map { case (i, v) =>
          v * plane(i)
        }.sum
      )
    }
    hashValues.map(Vectors.dense(_))
  }

  @Since("2.1.0")
  override protected[ml] def keyDistance(x: Vector, y: Vector): Double = {
    val xIter = x.nonZeroIterator.map(_._1)
    val yIter = y.nonZeroIterator.map(_._1)
    if (xIter.isEmpty) {
      require(yIter.hasNext, "The union of two input sets must have at least 1 elements")
      return 1.0
    } else if (yIter.isEmpty) {
      return 1.0
    }

    var xIndex = xIter.next
    var yIndex = yIter.next
    var xLength: Double = 0
    var yLength: Double = 0
    var xyProduct: Double = 0

    while (xIndex != -1 && yIndex != -1) {
      if (xIndex == yIndex) {
        val xValue = x(xIndex)
        val yValue = y(yIndex)
        xyProduct += xValue*yValue
        xLength += xValue*xValue;
        yLength += yValue*yValue;
        xIndex = if (xIter.hasNext) { xIter.next } else -1
        yIndex = if (yIter.hasNext) { yIter.next } else -1
      } else if (xIndex > yIndex) {
        val yValue = y(yIndex);
        yLength += yValue*yValue;
        yIndex = if (yIter.hasNext) { yIter.next } else -1
      } else {
        val xValue = x(xIndex);
        xLength += xValue*xValue;
        xIndex = if (xIter.hasNext) { xIter.next } else -1
      }
    }

    require(xLength*yLength > 0, "Both inputs should be non-empty")
    val cosSimilarity = xyProduct / sqrt(xLength*yLength)
    1 - cosSimilarity 
  }

  @Since("2.1.0")
  override protected[ml] def hashDistance(x: Seq[Vector], y: Seq[Vector]): Double = {
    x.iterator.zip(y.iterator).map(vectorPair =>
      vectorPair._1.toArray.zip(vectorPair._2.toArray).count(pair => pair._1 != pair._2)
    ).min
  }

  @Since("2.1.0")
  override def copy(extra: ParamMap): RandomHyperplanesLSHModel = {
    val copied = new RandomHyperplanesLSHModel(uid, randPlanes).setParent(parent)
    copyValues(copied, extra)
  }

  @Since("2.1.0")
  override def write: MLWriter = new RandomHyperplanesLSHModel.RandomHyperplanesLSHModelWriter(this)

  @Since("3.0.0")
  override def toString: String = {
    s"RandomHyperplanesLSHModel: uid=$uid, numHashTables=${$(numHashTables)}"
  }
}

/**
 * LSH class for Jaccard distance.
 *
 * The input can be dense or sparse vectors, but it is more efficient if it is sparse. For example,
 *    `Vectors.sparse(10, Array((2, 1.0), (3, 1.0), (5, 1.0)))`
 * means there are 10 elements in the space. This set contains elements 2, 3, and 5. Also, any
 * input vector must have at least 1 non-zero index, and all non-zero values are
 * treated as binary "1" values.
 *
 * References:
 * <a href="https://en.wikipedia.org/wiki/MinHash">Wikipedia on MinHash</a>
 */
@Since("2.1.0")
class RandomHyperplanesLSH(override val uid: String) extends LSH[RandomHyperplanesLSHModel] with HasSeed {

  @Since("2.1.0")
  override def setInputCol(value: String): this.type = super.setInputCol(value)

  @Since("2.1.0")
  override def setOutputCol(value: String): this.type = super.setOutputCol(value)

  @Since("2.1.0")
  override def setNumHashTables(value: Int): this.type = super.setNumHashTables(value)

  @Since("2.1.0")
  def this() = {
    this(Identifiable.randomUID("rhp-lsh"))
  }

  /** @group setParam */
  @Since("2.1.0")
  def setSeed(value: Long): this.type = set(seed, value)

  @Since("2.1.0")
  override protected[ml] def createRawLSHModel(inputDim: Int): RandomHyperplanesLSHModel = {
    val rand = new Random($(seed))
    val randPlanes: Array[Vector] = Array.fill($(numHashTables)) {
        Vectors.dense(Array.fill(inputDim)(rand.nextDouble))
      }
    new RandomHyperplanesLSHModel(uid, randPlanes)
  }

  @Since("2.1.0")
  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(inputCol), new VectorUDT)
    validateAndTransformSchema(schema)
  }

  @Since("2.1.0")
  override def copy(extra: ParamMap): this.type = defaultCopy(extra)
}

//@Since("2.1.0")
//object RandomHyperplanesLSH extends DefaultParamsReadable[RandomHyperplanesLSH] {
//  // A large prime smaller than sqrt(2^63 − 1)
//  private[ml] val HASH_PRIME = 2038074743
//
//  @Since("2.1.0")
//  override def load(path: String): RandomHyperplanesLSH = super.load(path)
//}

@Since("2.1.0")
object RandomHyperplanesLSHModel extends MLReadable[RandomHyperplanesLSHModel] {

  @Since("2.1.0")
  override def read: MLReader[RandomHyperplanesLSHModel] = new RandomHyperplanesLSHModelReader

  @Since("2.1.0")
  override def load(path: String): RandomHyperplanesLSHModel = super.load(path)

  private[RandomHyperplanesLSHModel] class RandomHyperplanesLSHModelWriter(instance: RandomHyperplanesLSHModel)
    extends MLWriter {

    private case class Data(randPlanes: Array[Vector])

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      //val data = Data(instance.randPlanes.flatMap(_.toArray))
      val data = Data(instance.randPlanes);
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class RandomHyperplanesLSHModelReader extends MLReader[RandomHyperplanesLSHModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[RandomHyperplanesLSHModel].getName

    override def load(path: String): RandomHyperplanesLSHModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath).select("randPlanes").head()
      //val randPlanes = data.getSeq[Int](0).grouped(2)
      //  .map(tuple => (tuple(0), tuple(1))).toArray
      val randPlanes = data.getSeq[Vector](0).toArray
      val model = new RandomHyperplanesLSHModel(metadata.uid, randPlanes)

      metadata.getAndSetParams(model)
      model
    }
  }
}
