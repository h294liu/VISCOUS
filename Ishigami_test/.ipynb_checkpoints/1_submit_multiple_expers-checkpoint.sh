#!/bin/bash

startNSample=10500
endNSample=15000
sampleInterval=500

startExperId=1 
endExperId=50

functionsPath=/home/h294liu/github/viscous/functions
outputBasepath=/home/h294liu/project/proj/6_viscous/Ishigami_test/experiments  # root path where parameter estimation will be stored.
current_path=$(pwd)
   
# 1. link functions to outputPath.
ln -s $functionsPath $outputBasepath/functions

# 2. create experiments
for nSample in $(seq ${startNSample} ${sampleInterval} ${endNSample}); do
    for experId in $(seq ${startExperId} 1 ${endExperId}); do
        outputPath=${outputBasepath}/sample${nSample}_exper${experId}
        if [ -d ${outputPath} ]; then rm -r ${outputPath}; fi
        mkdir -p $outputPath
        echo $outputPath

        # 1. copy Ishigami_test.py to outputPath.
        file1=Ishigami_test.py
        ln -s $current_path/$file1 $outputPath/$file1
        
        # 2. generate a new submit_Ishigami_test.sh.
        file2=submit_Ishigami_test.sh
        cp $file2 $outputPath/$file2
        sed -i "s/xxxxxx/s${nSample}_e${experId}/g" $outputPath/$file2
        sed -i "s/NSAMPLE/${nSample}/g" $outputPath/$file2
        chmod 744 $outputPath/$file2

        # 3. submit a job
        cd $outputPath
        sbatch $file2

        # 4. return
        cd $current_path
    done
done
