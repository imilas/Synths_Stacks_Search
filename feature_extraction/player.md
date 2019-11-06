Extract sounds into a CSV file:

````
rm 3_stack.csv ; python3 extract_features.py 3_stack/sounds/*.wav | tee 3_stack.csv
````

Play the sounds:

````
python3 tsneit.py
````
