total=$(find "$SCRATCH/LeBenchmark/" -type f | wc -l)                                                                                              
count=0                                  

find "$SCRATCH/LeBenchmark/" -type f | while read -r file; do                     
    touch "$file"                        
    count=$((count + 1))                 
    printf "\rProcessed %d/%d files" "$count" "$total"                            
done                                     

echo -e "\nDone!"   