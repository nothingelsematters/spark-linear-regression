name := "spark-linear-regression"

version := "0.1"

scalaVersion := "2.13.8"

val sparkVersion = "3.2.1"
val breezeVersion = "1.2"

libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-sql" % sparkVersion withSources(),
    "org.apache.spark" %% "spark-mllib" % sparkVersion withSources(),
    "org.apache.spark" %% "spark-core" % sparkVersion,
    "org.apache.spark" %% "spark-sql" % sparkVersion,
    "org.apache.spark" %% "spark-mllib" % sparkVersion,
    "org.scalanlp" %% "breeze" % breezeVersion,
    "org.scalanlp" %% "breeze-viz" % breezeVersion,

    "org.scalatest" %% "scalatest" % "3.2.12" % "test" withSources(),
)
