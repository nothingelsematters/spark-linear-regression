package org.apache.spark.ml.made

import com.google.common.io.Files
import org.scalatest.flatspec._
import org.scalatest.matchers._
import breeze.linalg.{*, DenseMatrix, DenseVector}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, Dataset}

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta: Double = 1e-3

  lazy val yVector: DenseVector[Double] = LinearRegressionTest._yVector
  lazy val xyDataset: Dataset[_] = LinearRegressionTest._xyDataset
  lazy val expectedWeights: DenseVector[Double] = LinearRegressionTest._expectedWeights

  private def validateTransformation(transformed: DataFrame): Unit = {
    transformed.columns should be(Seq("features", "label", "prediction"))
    transformed.collect().length should be(xyDataset.collect().length)

    val predicted = transformed.select("prediction").collect()
    (0 until 10).foreach(i => { predicted.toVector(i).getDouble(0) should be(yVector(i) +- delta) })
  }

  "Linear Regression" should "predict data correctly" in {
    val weights = new LinearRegression().fit(xyDataset).weights
    (0 until weights.length).foreach(i => { weights(i) should be(expectedWeights(i) +- delta) })
  }

  "Linear Regression Model" should "make prediction out of weights" in {
    validateTransformation(new LinearRegressionModel(expectedWeights).transform(xyDataset))
  }

  "Linear Regression Model" should "function after write & read" in {
    //noinspection UnstableApiUsage
    val temporaryFolderPath = Files.createTempDir().getAbsolutePath

    new Pipeline()
      .setStages(Array(new LinearRegression()))
      .fit(xyDataset)
      .write
      .overwrite()
      .save(temporaryFolderPath)

    val transformed = PipelineModel.load(temporaryFolderPath).transform(xyDataset)

    validateTransformation(transformed)
  }
}

object LinearRegressionTest extends WithSpark {

  import sqlContext.implicits._

  private lazy val _expectedWeights: DenseVector[Double] = DenseVector(1.5, 0.3, -0.7)

  private lazy val _xMatrix: DenseMatrix[Double] = DenseMatrix.rand[Double](1000, 3)
  private lazy val _yVector: DenseVector[Double] = _xMatrix * _expectedWeights

  private lazy val _xyDataset: Dataset[_] = {
    val _xyMatrix = DenseMatrix.horzcat(_xMatrix, _yVector.asDenseMatrix.t)

    new VectorAssembler()
      .setInputCols(Array("x1", "x2", "x3"))
      .setOutputCol("features")
      .transform(
        _xyMatrix(*, ::).iterator
          .map(row => Tuple4(row(0), row(1), row(2), row(3)))
          .toSeq
          .toDF("x1", "x2", "x3", "label")
      )
      .select("features", "label")
  }
}
