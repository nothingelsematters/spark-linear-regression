package org.apache.spark.ml.made

import org.scalatest.flatspec._
import org.scalatest.matchers._

class StartSparkTest extends AnyFlatSpec with should.Matchers with WithSpark {

    "Spark" should "start context" in {
        val _ = sparkSession
        Thread.sleep(60000)
    }
}
