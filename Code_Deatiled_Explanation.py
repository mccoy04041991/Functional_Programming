This code is written to design a pipeline to process raw data into a final feature for a machine learning model based on Random Forest Algorithm.
Whenever data was recorded by point of sales person in the system, they enter following inputs:

Given Data:

Itemno: 7 digit code to represent item. For example, bolts are respresnted by 0450056
Assembly_id: This field represent the assembly id to which itemno belong to. Remember, one itemno can belong to mutiple asemblies.
Assembly_name: Represents the assembly name corresponding to assembly_id column.

Sample input dataframe:

Itemno   Assembly_id    Assembly_name
0450056   44011         OIL PUMP ASSEMBLY - A01EA09CA (4999202399920239A06)
0450056   135502        OIL PUMP ASSEMBLY - A02EA09CA/CB/CC (4999202399920239A06)
0450056   37884         OIL PUMP ASSEMBLY - A01EA05CA (4999202399920239A06)

Output:
We need all the assembly_id's and model_numbers that corresponds to single distinct item in a single row.

Sample output dataframe

|Itemno   |fits_assembly_id                                        |fits_assembly_name                                                                 |assembly_name                              |Models 

|0450056  |13039 135502 141114 4147 138865 2021 9164               |OIL PUMP ASSEMBLY A01EA09CA 4999202399920239A06  A02EA09CB A02EA09CC               |OIL PUMP ASSEMBLY 999202399920239A06       |A02EA09CA A02EA09CB A02EA09CC 

Challenges:
    -   Since fits_model_number column data isn't accurate, we need to extract model number information from fits_assembly_name column. 
        -   For instance,if we look at first row of input dataframe, we see that while fits_model_number only reports 1 model i.e. Z19VFK99LK belongs to Itemno 0450056 while acually it is two which is mentioned in   fits_assembly_name column as "Z19VFK99LK/LE" i.e. Itemno 0450056 goes to Z19VFK99LK and Z19VFK99LE.


How-to get it done?

I've broken this task to 5 steps which are as following:

#Step-1: Break fits_assembly_name column into assembly_name & models
After observing the pattern, I've figured out following
    -   Model numbers are mentioned after hyphen(-) in fits_assembly_name column, and
    -   They are condensed to fit the space. E.g. Three different model_numbers V08AB26, V08GB26 and V08LB26 are written as V08AB26/GB26/LB26.
        -   POS has used "/" and mentioned only the part that is different from earlier model_numbers.

To pre-process the column Assembly_name, use regexp_replace + split to separate modelsnumbers into a new column and remove it from the original column Assembly_name:
    -   I've used "regexp_replace" and "selectExpr" funtions available in pyspark.sql.functions.
        -   I've broken the fits_assembly_name column string by " - " and created two new columns i.e. pc_Assemblyname_Withoutmodelno and Models column as following:        
        -   regexp_replace: Replace all substrings of the specified string value that match regexp with rep
            - Usage: regexp_replace(x, pattern, replacement)
        -   selectExpr: Projects a set of SQL expressions and returns a new DataFrame. (Source: https://spark.apache.org/docs/1.5.2/api/python/pyspark.sql.html)
```
from pyspark.sql.functions import regexp_replace, split

df0 = df.withColumn('new_col', split(regexp_replace('Assembly_name', r'^(.*)-\s*(\S+)(.*)$', '$1$3\0$2'),'\0')) \
    .selectExpr(
        'Itemno'
      , 'Assembly_id'
      , "coalesce(new_col[0], Assembly_name) as Assembly_name"
      , "coalesce(new_col[1], '') as models"
)

df0.show(truncate=False)
+-------+-----------+---------------------------------------------------------------+--------------------+
|Itemno |Assembly_id|Assembly_name                                                  |models              |
+-------+-----------+---------------------------------------------------------------+--------------------+
|0450056|44011      |OIL PUMP ASSEMBLY  (4999202399920239A06)                       |A01EA09CA           |
|0450056|135502     |OIL PUMP ASSEMBLY  (4999202399920239A06)                       |A02EA09CA/CB/CC     |
|0450056|37884      |OIL PUMP ASSEMBLY  (4999202399920239A06)                       |A01EA05CA           |
|0450067|12345      |DRIVE TRAIN, TRANSMISSION (6 SPEED)  ALL OPTIONS (49VICTRANS08)|V08AB26/GB26/LB26   |
|0450068|1000       |SUSPENSION (7043244)  (49SNOWSHOCKFRONT7043244SB)              |S09PR6HSL/PS6HSL/HEL|
|0450066|12345      |DRIVE TRAIN, CLUTCH, PRIMARY  (49SNOWDRIVECLUTCH09600TRG)      |S09PR6HSL/PS_HSL/H_L|
|0450069|12346      |DRIVE TRAIN, CLUTCH, PRIMARY (49SNOWDRIVECLUTCH09600TRG)       |                    |
+-------+-----------+---------------------------------------------------------------+--------------------+
```

#Step-2: convert string into array of arrays
-   split the string by the pattern (?:(?!/)\p{Punct}|\s)+')) which is consecutive punctuation(except /) or spaces, then filter out the items which are EMPTY (leading/trailing). A temporary column temp1 is used to save the      intermediate columns
-  split the array of array further into array of arrays of arrays, the inner-most array has split string into chars. reverse the innermost array so it's easy for comparison.

df2 = df1.withColumn('temp1', expr("transform(temp1, x -> split(x, '/'))")) \
         .withColumn('temp1', expr("transform(temp1, x -> transform(x, y -> reverse(split(y, ''))) )"))

df1.select('temp1').show(truncate=False)
+-------------------------------------------------------------------------------------+
|temp1                                                                                |
+-------------------------------------------------------------------------------------+
|[OIL, PUMP, ASSEMBLY, A01EA09CA, 4999202399920239A06]                                |
|[OIL, PUMP, ASSEMBLY, A02EA09CA/CB/CC, 4999202399920239A06]                          |
|[OIL, PUMP, ASSEMBLY, A01EA05CA, 4999202399920239A06]                                |
|[DRIVE, TRAIN, TRANSMISSION, 6, SPEED, V08AB26/GB26/LB26, ALL, OPTIONS, 49VICTRANS08]|
|[SUSPENSION, 7043244, S09PR6HSL/PS6HSL/HEL, 49SNOWSHOCKFRONT7043244SB]               |
+-------------------------------------------------------------------------------------+
```

#Step-3: convert temp1 to array of arrays
split the array items again using /, so that all part-id on their own array item

```
df2 = df1.withColumn('temp1', expr("transform(temp1, x -> split(x, '/'))"))
df2.select('temp1').show(truncate=False)
+----------------------------------------------------------------------------------------------------------+
|temp1                                                                                                     |
+----------------------------------------------------------------------------------------------------------+
|[[OIL], [PUMP], [ASSEMBLY], [A01EA09CA], [4999202399920239A06]]                                           |
|[[OIL], [PUMP], [ASSEMBLY], [A02EA09CA, CB, CC], [4999202399920239A06]]                                   |
|[[OIL], [PUMP], [ASSEMBLY], [A01EA05CA], [4999202399920239A06]]                                           |
|[[DRIVE], [TRAIN], [TRANSMISSION], [6], [SPEED], [V08AB26, GB26, LB26], [ALL], [OPTIONS], [49VICTRANS08]] |
|[[SUSPENSION], [7043244], [S09PR6HSL, PS6HSL, HEL], [49SNOWSHOCKFRONT7043244SB]]                          |
+------------------------------------------------------------------------------------------
```

#Step-4: use transform to reset part-ids

Use transform() to reset part-ids. we check y[i] (the item of the innermost array) if it is NULL or is an underscore, then replace it with the corresponding item from x[0][i]. then we reverse the array and using concat_ws(''..) to convert it back into string.

```
df3 = df2.withColumn('temp1', expr("""

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

"""))
```

Below is the result from the above

df3.select('temp1').show(truncate=False)                                                                           
+---------------------------------------------------------------------------------------------+
|temp1                                                                                        |
+---------------------------------------------------------------------------------------------+
|[OIL, PUMP, ASSEMBLY, A01EA09CA, 4999202399920239A06]                                        |
|[OIL, PUMP, ASSEMBLY, A02EA09CA, A02EA09CB, A02EA09CC, 4999202399920239A06]                  |
|[OIL, PUMP, ASSEMBLY, A01EA05CA, 4999202399920239A06]                                        |
|[DRIVE, TRAIN, TRANSMISSION, 6, SPEED, V08AB26, V08GB26, V08LB26, ALL, OPTIONS, 49VICTRANS08]|
|[SUSPENSION, 7043244, S09PR6HSL, S09PS6HSL, S09PR6HEL, 49SNOWSHOCKFRONT7043244SB]            |
|[DRIVE, TRAIN, CLUTCH, PRIMARY, S09PR6HSL, S09PS6HSL, S09PR6HSL, 49SNOWDRIVECLUTCH09600TRG]  |
|[DRIVE, TRAIN, CLUTCH, PRIMARY, S09PR6HSL, S09PS6HSL, S09PR6HSL, 49SNOWDRIVECLUTCH09600TRG]  |
+---------------------------------------------------------------------------------------------+

Where:

    -   transform(temp1, x -> func1(x)) : iterate through each item in the array temp1 to run func1(x), x is the inner array (array of strings)
        func1(x) mentioned above is another transform function which iterates through the sequence(1, size(x)) and run func2(i) on each i:
        transform(sequence(1, size(x)), i -> func2(i))
    
    -   func2(i) mentioned above is an aggregate function, which iterates through the sequence(1,i), with initial value of x[0] and accumulate/reduce using the function:
        (acc,j) -> concat(substr(acc, 1, length(acc)-length(x[j-1])), x[j-1])
    
        Note: substr() position is 1-based and array-indexing is 0-based, thus we need x[j-1] to refer to the current array item in the above reduce/aggregate function

    -   finally, run flatten() to merge the inner arrays

        -   This step is doing something like the following pysudo-loop:

            for x in temp1:
              for i in range(1, size(x)+1):
                acc = x[0]
                for j in range(1,i+1):
                  acc = concat(substr(acc, 1, length(acc)-length(x[j-1])), x[j-1])

#Step-5: merge and drop duplicates
```
df4 = df3.groupby('Itemno').agg(
      expr("concat_ws(' ', array_distinct(flatten(collect_list(temp1)))) AS Assembly_names")
    , expr("concat_ws(' ', collect_set(Assembly_id)) as Assembly_ids")
  )
```

Where:
    -   use collect_list() to get an array of arrays(temp1 which is array of strings)
    -   use flatten() to convert the above into array of strings
    -   use array_distinct() to drop duplicates
    -   use concat_ws() to convert above array into a string

        df4.select('Assembly_names').show(truncate=False)
        +---------------------------------------------------------------------------------------+
        |Assembly_names                                                                         |
        +---------------------------------------------------------------------------------------+
        |OIL PUMP ASSEMBLY A01EA09CA 4999202399920239A06 A02EA09CA A02EA09CB A02EA09CC A01EA05CA|
        |SUSPENSION 7043244 S09PR6HSL S09PS6HSL S09PS6HEL 49SNOWSHOCKFRONT7043244SB             |
        |DRIVE TRAIN TRANSMISSION 6 SPEED V08AB26 V08GB26 V08LB26 ALL OPTIONS 49VICTRANS08      |
        +---------------------------------------------------------------------------------------+

#Step-6: Remove puntuations, turn data into small cases and remove stop words
    -   We've used a list comprehension to handle multiple columns, and 
    -   We've used pyspark.ml.Pipeline to skip the intermediate dataframes

```

from pyspark.ml.feature import StopWordsRemover, RegexTokenizer
from pyspark.ml import Pipeline
for col in ['fits_assembly_name', 'assembly_name']:
    tk = RegexTokenizer(pattern=r'(?:\p{Punct}|\s)+', inputCol=col, outputCol='temp1')
    sw = StopWordsRemover(inputCol='temp1', outputCol='temp2')
    pipeline = Pipeline(stages=[tk, sw])
    df4 = pipeline.fit(df4).transform(df4) \
        .withColumn(col, expr('concat_ws(" ", array_distinct(temp2))')) \
        .drop('temp1', 'temp2')
```