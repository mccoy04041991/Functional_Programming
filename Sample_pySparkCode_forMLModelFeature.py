from pyspark.sql.functions import *
from pyspark.ml.feature import StopWordsRemover, RegexTokenizer
from pyspark.ml import Pipeline

#Step-1: Break partsCatalog_Assemblyname into assembly_name & models
pcDF0 = pcRawDF.withColumn('new_col', split(regexp_replace('pc_Ori_Assemblyname', r'^(.*?)\s+-\s+(\S+)(.*)$', '$1$3\0$2'),'\0')) \
                .selectExpr(
                                'Itemno'
                              , 'fits_model_number'
#                              , 'partsCatalog_Description'
                              , 'fits_assembly_id'
                              , 'fits_assembly_name'
                              , "coalesce(new_col[0], fits_assembly_name) as pc_Assemblyname_Withoutmodelno"
                              , "coalesce(new_col[1], '') as modelno"
                            )

# display(pcDF0)

#Step-2: convert string into array of strings

pcDF1 = pcDF0.withColumn('temp1', split('fits_assembly_name', r'(?:(?![/_])\p{Punct}|\s)+')) \
                .withColumn('temp1', expr("filter(temp1, x -> x <> '')")) \
                .withColumn('temp2', split('pc_Assemblyname_Withoutmodelno', r'(?:(?![/_])\p{Punct}|\s)+')) \
                .withColumn('temp2', expr("filter(temp2, x -> x <> '')"))
       

# display(pcDF1)

#Step-3: split the array of array further
pcDF2 = pcDF1.withColumn('temp1', expr("transform(temp1, x -> split(x, '/'))")) \
             .withColumn('temp1', expr("transform(temp1, x -> transform(x, y -> reverse(split(y, ''))) )")) \
             .withColumn('temp2', expr("transform(temp2, x -> split(x, '/'))")) \
             .withColumn('temp2', expr("transform(temp2, x -> transform(x, y -> reverse(split(y, ''))) )"))

# display(pcDF2)

#Step - 4: use transform() to reset part ids.

pcDF3 = pcDF2.withColumn('temp1', expr("""

   flatten(
     transform(temp1, x ->
       transform(x, y ->
         concat_ws('', 
           reverse(
             transform(sequence(0, size(x[0])-1), i -> IF(y[i] is NULL or y[i] == '_', x[0][i], y[i]))
           )
         )
       )
     ) 
   ) 

""")) \
      .withColumn('temp2', expr("""

   flatten(
     transform(temp2, x ->
       transform(x, y ->
         concat_ws('', 
           reverse(
             transform(sequence(0, size(x[0])-1), i -> IF(y[i] is NULL or y[i] == '_', x[0][i], y[i]))
           )
         )
       )
     ) 
   ) 

"""))

# display(pcDF3)

#Step-5: merge and drop duplicates

pcDF4 = pcDF3.groupby('partsCatalog_Itemno').agg(
     expr("concat_ws(' ', collect_set(partsCatalog_Modelnumber)) AS partsCatalog_Modelnumber"),
     expr("concat_ws(' ', collect_set(partsCatalog_Description)) AS partsCatalog_Description"),
     expr("concat_ws(' ', collect_set(cast(pc_Ori_Assemblyid as int))) AS pc_Ori_Assemblyid"),
     expr("concat_ws(' ', array_distinct(flatten(collect_list(temp1)))) AS pc_Ori_Assemblyname"),
     expr("concat_ws(' ', array_distinct(flatten(collect_list(temp2)))) AS pc_Assemblyname_Withoutmodelno")
  )

#lower case model number
pcDF5 =pcDF4.withColumn('partsCatalog_Modelnumber', lower(col('partsCatalog_Modelnumber')))

#Step-6: Remove puntuations, turn data into small cases and remove stop words

for col in ['partsCatalog_Description', 'pc_Ori_Assemblyname', 'pc_Assemblyname_Withoutmodelno']:
    (temp1, temp2) = (col+'_tmp1', col+'_tmp2')        
    tk = RegexTokenizer(pattern=r'(?:\p{Punct}|\s)+', inputCol=col, outputCol=temp1)         
    sw = StopWordsRemover(inputCol=temp1, outputCol=temp2)         
    pipeline = Pipeline(stages=[tk, sw])         
    pcDF5 = pipeline.fit(pcDF5).transform(pcDF5) \
        .withColumn(col, expr('concat_ws(" ", array_distinct({}))'.format(temp2))) \
        .drop(temp1, temp2)
    

# display(pcDF5)