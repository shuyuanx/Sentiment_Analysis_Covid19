from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when

conf = SparkConf().setMaster('local[4]').setAppName('random forest model')
sc = SparkContext(conf = conf)
sqlc = SQLContext(sc) 

df = sqlc.read.format('csv').options(header='true', inferSchema='true').load('processed_df.csv')
df = df.withColumnRenamed("label", "label_orig")
col_names = df.schema.names
features = col_names[:-1]
label = col_names[-1]
stages = []

df = df.withColumn("label_orig", when(col("label_orig")=='positive', 2).when(col("label_orig")=='neutral', 1).otherwise(0))

label_stringIdx = StringIndexer(inputCol = "label_orig", outputCol = 'label')
stages += [label_stringIdx]

assembler = VectorAssembler(inputCols=features, outputCol="features")
stages += [assembler]

pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)
selectedCols = ['label', 'features'] + col_names
df = df.select(selectedCols)

# train test split
train, test = df.randomSplit([0.8, 0.2], seed = 0)

# model training
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', maxDepth=30, numTrees=200)
rfModel = rf.fit(train)
predictions = rfModel.transform(test)

# model evaluation
results = predictions.select(['prediction', 'label'])
predictionAndLabels=results.rdd
metrics = MulticlassMetrics(predictionAndLabels)

"""
evaluator = MulticlassClassificationEvaluator()
print("Accuracy: " + str(evaluator.evaluate(results, {evaluator.metricName: "accuracy"})))
"""

# Overall statistics
precision = metrics.precision()
recall = metrics.recall()
f1Score = metrics.fMeasure()
print("Summary Stats")
print("Precision = %s" % precision)
print("Recall = %s" % recall)
print("F1 Score = %s" % f1Score)

cm=metrics.confusionMatrix().toArray()
accuracy=(cm[0][0]+cm[1][1]+cm[2][2])/cm.sum()

print("Overall accuracy score = %s" % accuracy)
#print(cm)

