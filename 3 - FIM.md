
# Frequent Itemset Mining
**MSDS2020**<br>
Chua and Gacera


```pyspark
reviews = spark.read.parquet('s3://bdcc-jgacera-2020/amazon-reviews/2015/')
```


    VBox()


    Starting Spark application
    


<table>
<tr><th>ID</th><th>YARN Application ID</th><th>Kind</th><th>State</th><th>Spark UI</th><th>Driver log</th><th>Current session?</th></tr><tr><td>15</td><td>application_1577505212894_0016</td><td>pyspark</td><td>idle</td><td><a target="_blank" href="http://ip-172-31-37-101.ap-northeast-1.compute.internal:20888/proxy/application_1577505212894_0016/">Link</a></td><td><a target="_blank" href="http://ip-172-31-34-220.ap-northeast-1.compute.internal:8042/node/containerlogs/container_1577505212894_0016_01_000001/livy">Link</a></td><td>✔</td></tr></table>



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    SparkSession available as 'spark'.
    


    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
reviews.printSchema()
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

### Number of reviews in 2015:


```pyspark
reviews.count()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    2993642

### Filter to only high-rated books (4-5 `star_rating`)


```pyspark
reviews = reviews[reviews.star_rating>=4]
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


### Number of candidate books:


```pyspark
reviews.count()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    2607876

### Create transaction database for FIM


```pyspark
from pyspark.sql.functions import collect_set

df_trans = (reviews.groupby('customer_id')
                   .agg(collect_set('product_id').alias('products'))
                   .cache())
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


### Number of transactions:


```pyspark
df_trans.count()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    1383784


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
df_trans.limit(3).toPandas()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


      customer_id      products
    0    10001922  [3869930691]
    1    10019218  [1465420207]
    2    10020963  [0590431978]

### Define `FPGrowth` model


```pyspark
from pyspark.ml.fpm import FPGrowth

fpgrowth = FPGrowth(itemsCol='products', minSupport=0.0001, minConfidence=0)
fpgrowth_trained = fpgrowth.fit(df_trans)

freq_items = fpgrowth_trained.freqItemsets
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


### Resulting frequent itemsets:


```pyspark
freq_items.count()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    558


```pyspark
freq_items.take(3)
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    [Row(items=['0525427651'], freq=238), Row(items=['0692346856'], freq=164), Row(items=['0316376469'], freq=1869)]

### Filter to itemsets with more than 1 item


```pyspark
from pyspark.sql.functions import udf

no_items = udf(lambda x: len(x))

freq_items = freq_items.withColumn('no_items', no_items(freq_items['items']).cast('int'))
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
freq_items_2 = freq_items[freq_items.no_items>1]
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


### Number of frequent itemsets with more than 1 item:


```pyspark
freq_items_2.count()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    6

### Retrieve details of frequent itemsets


```pyspark
df_freq = freq_items_2.toPandas()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
all_items = [item for sublist in df_freq['items'].tolist() for item in sublist]
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
set(all_items)
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    {'1476755884', '1476755868', '1780674880', '1780671067', '1476789878', '0345803507', '0345803493'}


```pyspark
def retrieve_title(pid):
    return reviews[reviews.product_id==pid].limit(1).toPandas().loc[0, 'product_title']
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
df_freq['titles'] = df_freq['items'].apply(lambda x: [retrieve_title(i) for i in x])
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
df_freq[['titles', 'freq']]
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


                                                  titles  freq
    0  [Secret Garden: An Inky Treasure Hunt and Colo...   209
    1  [Rush Revere and the Brave Pilgrims: Time-Trav...   176
    2  [Rush Revere and the First Patriots: Time-Trav...   254
    3  [Rush Revere and the First Patriots: Time-Trav...   230
    4  [Rush Revere and the First Patriots: Time-Trav...   145
    5  [Fifty Shades Freed: Book Three of the Fifty S...   171


```pyspark
for i in range(len(df_freq)):
    print('freq:', df_freq.loc[i, 'freq'])
    print(df_freq.loc[i, 'titles'], '\n')
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    freq: 209
    ['Secret Garden: An Inky Treasure Hunt and Coloring Book', 'Enchanted Forest: An Inky Quest & Coloring Book [US Import]'] 
    
    freq: 176
    ['Rush Revere and the Brave Pilgrims: Time-Travel Adventures with Exceptional Americans', 'Rush Revere and the American Revolution: Time-Travel Adventures With Exceptional Americans'] 
    
    freq: 254
    ['Rush Revere and the First Patriots: Time-Travel Adventures With Exceptional Americans', 'Rush Revere and the American Revolution: Time-Travel Adventures With Exceptional Americans'] 
    
    freq: 230
    ['Rush Revere and the First Patriots: Time-Travel Adventures With Exceptional Americans', 'Rush Revere and the Brave Pilgrims: Time-Travel Adventures with Exceptional Americans'] 
    
    freq: 145
    ['Rush Revere and the First Patriots: Time-Travel Adventures With Exceptional Americans', 'Rush Revere and the Brave Pilgrims: Time-Travel Adventures with Exceptional Americans', 'Rush Revere and the American Revolution: Time-Travel Adventures With Exceptional Americans'] 
    
    freq: 171
    ['Fifty Shades Freed: Book Three of the Fifty Shades Trilogy (Fifty Shades of Grey Series) (English Edition)', 'Fifty Shades Darker']


```pyspark

```
