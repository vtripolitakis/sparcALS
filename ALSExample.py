#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys,json

from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS

"""
ALSExample.py - 25/9/2015 - Evangelos Tripolitakis
vtripolitakis@gmail.com

$ cd $SPARK_HOME
$ ./bin/spark-submit --master local[*]  \
> /path_to/ALSExample.py /path_to/ratings_file /path_to/output_files 

"""


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("""
        first argument is ratings file and second is output file
        """)
        exit(-1)

    txtFile = sys.argv[1]
    outputFile = sys.argv[2]

    sc = SparkContext(appName="ALSExample")
    ratings = sc.textFile(txtFile)
    processedRatings = ratings.map(lambda line: (int(line.split(",")[0]),int(line.split(",")[1]),float(line.split(",")[2])))
    users = ratings.map(lambda rating: int(rating.split(",")[0])).distinct().collect()

    #train model
    model = ALS.trainImplicit(processedRatings, 1,seed=10)

    outArray=[]
    f=open(outputFile,'w')
    
    for user in users:        
        outArray.append(model.recommendProducts(user,20))

    f.write(json.dumps(outArray))

    sc.stop()
    f.close()