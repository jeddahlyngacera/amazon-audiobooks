
# Data gathering
**MSDS2020**<br>
Chua and Gacera

Cluster contained 5 `m5.xlarge` instances with which 1 is the master instance and 4 are the core nodes (slave workers).

The <a href="https://registry.opendata.aws/amazon-reviews/">Amazon Customer Reviews Dataset</a> from the Registry of Open Data on AWS, specifically the `Books` product category in the s3 bucket <br>
`s3://amazon-reviews-pds/parquet/product_category=Books/`, is loaded using `spark.read.parquet` for processing:


```pyspark
reviews_raw = spark.read.parquet('s3://amazon-reviews-pds/parquet/product_category=Books/')
```


    VBox()


    Starting Spark application
    


<table>
<tr><th>ID</th><th>YARN Application ID</th><th>Kind</th><th>State</th><th>Spark UI</th><th>Driver log</th><th>Current session?</th></tr><tr><td>14</td><td>application_1577505212894_0015</td><td>pyspark</td><td>idle</td><td></td><td></td><td>✔</td></tr></table>



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    SparkSession available as 'spark'.
    


    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


### Schema/columns of the reviews dataset:


```pyspark
reviews_raw.printSchema()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    root
     |-- marketplace: string (nullable = true)
     |-- customer_id: string (nullable = true)
     |-- review_id: string (nullable = true)
     |-- product_id: string (nullable = true)
     |-- product_parent: string (nullable = true)
     |-- product_title: string (nullable = true)
     |-- star_rating: integer (nullable = true)
     |-- helpful_votes: integer (nullable = true)
     |-- total_votes: integer (nullable = true)
     |-- vine: string (nullable = true)
     |-- verified_purchase: string (nullable = true)
     |-- review_headline: string (nullable = true)
     |-- review_body: string (nullable = true)
     |-- review_date: date (nullable = true)
     |-- year: integer (nullable = true)

### Number of total reviews:


```pyspark
reviews_raw.count()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    20726160

### Filter reviews created in the year `2015`:


```pyspark
reviews = reviews_raw[reviews_raw.year==2015].dropna()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


### Number of reviews in 2015:


```pyspark
reviews.count()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    2993642

### Write filtered reviews to `parquet` files under `s3` bucket:

```python
reviews.write.parquet('s3://bdcc-jgacera-2020/amazon-reviews/2015/')
```

# Modeling

Machine learning classification models will be implemented to predict whether a book is a good candidate to be converted to audible or not using words extracted from `review_body`.


```pyspark
reviews = spark.read.parquet('s3://bdcc-jgacera-2020/amazon-reviews/2015/')
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


### Sample data for training


```pyspark
reviews = reviews.sample(False, 0.1, 42)
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


### Create `target` column with value `1` if considered good candidate for conversion to audible, `0` otherwise

Specifically, high-rated books (4 or 5 star rating) will be tagged as 1, while low-rated books (1-3 rating) as 0.


```pyspark
from pyspark.sql.functions import udf

convert = udf(lambda x: 1 if x>=4 else 0)

reviews = reviews.withColumn('target', convert('star_rating').cast('int'))
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
df_reviews = reviews.select('review_body', 'target')
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
sc.install_pypi_package('pandas')
import pandas as pd
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    Collecting pandas
      Using cached https://files.pythonhosted.org/packages/52/3f/f6a428599e0d4497e1595030965b5ba455fd8ade6e977e3c819973c4b41d/pandas-0.25.3-cp36-cp36m-manylinux1_x86_64.whl
    Requirement already satisfied: numpy>=1.13.3 in /usr/local/lib64/python3.6/site-packages (from pandas) (1.14.5)
    Collecting python-dateutil>=2.6.1
      Using cached https://files.pythonhosted.org/packages/d4/70/d60450c3dd48ef87586924207ae8907090de0b306af2bce5d134d78615cb/python_dateutil-2.8.1-py2.py3-none-any.whl
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.6/site-packages (from pandas) (2019.3)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.6/site-packages (from python-dateutil>=2.6.1->pandas) (1.12.0)
    Installing collected packages: python-dateutil, pandas
    Successfully installed pandas-0.25.3 python-dateutil-2.8.1


```pyspark
df_reviews.limit(3).toPandas()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


                                             review_body  target
    0  The book was very good! To be honest it was a ...       1
    1  Fantastic book.  The organization and presenta...       1
    2  Practical reading and straight to the point. I...       0

### Preprocess `review_body` text data for feature extraction


```pyspark
from pyspark.ml.feature import Tokenizer, RegexTokenizer

regexTokenizer = RegexTokenizer(inputCol='review_body', outputCol='words', pattern='\\W')
words = regexTokenizer.transform(df_reviews)
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
from pyspark.ml.feature import StopWordsRemover

remover = StopWordsRemover(inputCol='words', outputCol='cleaned')
cleaned = remover.transform(words)
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
cleaned.limit(3).toPandas()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


                                             review_body  ...                                            cleaned
    0  The book was very good! To be honest it was a ...  ...  [book, good, honest, little, slow, beginning, ...
    1  Fantastic book.  The organization and presenta...  ...  [fantastic, book, organization, presentation, ...
    2  Practical reading and straight to the point. I...  ...  [practical, reading, straight, point, interest...
    
    [3 rows x 4 columns]

### Create `features` column containing the vector representing the words in `review_body`


```pyspark
from pyspark.ml.feature import Word2Vec

word2Vec = Word2Vec(inputCol='cleaned', outputCol='features')
model = word2Vec.fit(cleaned)
result = model.transform(cleaned)
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


### Split data into train and test (75-25)


```pyspark
train_data, test_data = result.randomSplit([0.75, 0.25])
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


### Define model evaluator


```pyspark
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='target')
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


### RandomForestClassifier


```pyspark
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(featuresCol='features', labelCol='target', maxBins=100, maxDepth=10)
rf_trained = rf.fit(train_data)
df_predict = rf_trained.transform(test_data)
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
print('Area under ROC curve =', evaluator.evaluate(df_predict.select('prediction', 'target')))
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    Area under ROC curve = 0.5773991695143971


```pyspark
print('Area under precision-recall curve', 
      evaluator.evaluate(df_predict.select('prediction', 'target'), {evaluator.metricName: 'areaUnderPR'}))
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    Area under precision-recall curve 0.8892068563041179


```pyspark
rf_trained.save('s3://bdcc-jgacera-2020/amazon-reviews/rf_model')
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


### GradientBoostedTreeClassifier


```pyspark
from pyspark.ml.classification import GBTClassifier

gb = GBTClassifier(featuresCol='features', labelCol='target', maxBins=100, maxDepth=10)
gb_trained = gb.fit(train_data)
df_predict2 = gb_trained.transform(test_data)
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
print('Area under ROC curve =', evaluator.evaluate(df_predict2.select('prediction', 'target')))
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    Area under ROC curve = 0.6709837617256397


```pyspark
print('Area under precision-recall curve', 
      evaluator.evaluate(df_predict2.select('prediction', 'target'), {evaluator.metricName: 'areaUnderPR'}))
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    Area under precision-recall curve 0.9171162186154799


```pyspark
gb_trained.save('s3://bdcc-jgacera-2020/amazon-reviews/gb_model')
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


### DecisionTreeClassifier


```pyspark
from pyspark.ml.classification import DecisionTreeClassifier

dt = DecisionTreeClassifier(featuresCol='features', labelCol='target', maxBins=100, maxDepth=10)
dt_trained = dt.fit(train_data)
df_predict3 = dt_trained.transform(test_data)
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
print('Area under ROC curve =', evaluator.evaluate(df_predict3.select('prediction', 'target')))
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    Area under ROC curve = 0.6408895349448369


```pyspark
print('Area under precision-recall curve', 
      evaluator.evaluate(df_predict3.select('prediction', 'target'), {evaluator.metricName: 'areaUnderPR'}))
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    Area under precision-recall curve 0.9064882907347057


```pyspark
dt_trained.save('s3://bdcc-jgacera-2020/amazon-reviews/dt_model')
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


### LogisticRegression


```pyspark
from pyspark.ml.classification import LogisticRegression

lg = LogisticRegression(featuresCol='features', labelCol='target')
lg_trained = lg.fit(train_data)
df_predict4 = lg_trained.transform(test_data)
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
print('Area under ROC curve =', evaluator.evaluate(df_predict4.select('prediction', 'target')))
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    Area under ROC curve = 0.6281785326547763


```pyspark
print('Area under precision-recall curve', 
      evaluator.evaluate(df_predict4.select('prediction', 'target'), {evaluator.metricName: 'areaUnderPR'}))
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    Area under precision-recall curve 0.9031120829736311


```pyspark
lg_trained.save('s3://bdcc-jgacera-2020/amazon-reviews/lg_model')
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


### Best model: GradientBoostedTreeClassifier


```pyspark
from pyspark.ml.classification import GBTClassificationModel

best_model = GBTClassificationModel.load('s3://bdcc-jgacera-2020/amazon-reviews/gb_model')
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
gb_trained.treeWeights == best_model.treeWeights
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    True


```pyspark

```
