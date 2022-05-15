package org.apache.spark.ml.made

import breeze.linalg.functions.euclideanDistance
import breeze.linalg.{DenseVector, sum}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{VectorUDT, Vector => MlVector, Vectors => MlVectors}
import org.apache.spark.ml.param.shared.{HasMaxIter, HasTol}
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter,
  Identifiable, MLReadable, MLReader, MLWritable, MLWriter, SchemaUtils}
import org.apache.spark.ml.{Estimator, Model, PredictorParams}
import org.apache.spark.mllib.linalg.{Vectors => MllibVectors}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}

trait LinearRegressionParams extends PredictorParams with HasMaxIter with HasTol {

  val learningRate: Param[Double] = new DoubleParam(this, "learningRate", "learning rate")

  def getLearningRate: Double = $(learningRate)

  def setLearningRate(value: Double): this.type = set(learningRate, value)

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setLabelCol(value: String): this.type = set(labelCol, value)

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setTol(value: Double): this.type = set(tol, value)

  setDefault(learningRate -> 0.05, maxIter -> 1e5.toInt, tol -> 1e-7)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT())

    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, $(predictionCol), DoubleType)
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema($(featuresCol)).copy(name = $(predictionCol)))
    }
  }
}

class LinearRegression(override val uid: String)
  extends Estimator[LinearRegressionModel] with LinearRegressionParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {

    implicit val encoder: Encoder[MlVector] = ExpressionEncoder()

    val vectors = {
      val vectorColumn = "result column"
      val assembler = new VectorAssembler().setInputCols(Array($(featuresCol), $(labelCol))).setOutputCol(vectorColumn)
      assembler.transform(dataset).select(vectorColumn).as[MlVector]
    }

    val weightSize = vectors.first().size - 1
    var previousWeights = DenseVector.fill(weightSize, Double.PositiveInfinity)
    val currentWeights = DenseVector.fill(weightSize, 0.0)

    // scala no breakable for loops crutch
    var iteration = 0
    while (iteration < $(maxIter) && euclideanDistance(previousWeights, currentWeights) > $(tol)) {

      val summary = vectors.rdd
        .mapPartitions((data: Iterator[MlVector]) => {
          val summarizer = new MultivariateOnlineSummarizer()

          data.foreach(row => {
            val x = row.asBreeze(0 until weightSize).toDenseVector
            val weightsDelta = x * (x.dot(currentWeights) - row.asBreeze(-1))
            summarizer.add(MllibVectors.fromBreeze(weightsDelta))
          })

          Iterator(summarizer)
        })
        .reduce(_ merge _)

      previousWeights = currentWeights.copy
      currentWeights -= $(learningRate) * summary.mean.asBreeze
      iteration += 1
    }

    copyValues(new LinearRegressionModel(currentWeights).setParent(this))
  }

  override def copy(extra: ParamMap): LinearRegression = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel(override val uid: String, val weights: DenseVector[Double])
  extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {

  def this(weights: DenseVector[Double]) = this(Identifiable.randomUID("linearRegressionModel"), weights)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = {
      dataset.sqlContext.udf.register(
        uid + "_transform",
        (x: MlVector) => { sum(x.asBreeze.toDenseVector * weights(0 until weights.length)) },
      )
    }
    dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
  }

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(weights), extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      val vectors = Tuple1(MlVectors.fromBreeze(weights))
      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {

  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)
      val vectors = sqlContext.read.parquet(path + "/vectors")

      implicit val encoder: Encoder[MlVector] = ExpressionEncoder()

      val weights = vectors.select(vectors("_1").as[MlVector]).first().asBreeze.toDenseVector
      val model = new LinearRegressionModel(weights)
      metadata.getAndSetParams(model)
      model
    }
  }
}
