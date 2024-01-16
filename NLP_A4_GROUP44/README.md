# NLP_A4_GROUP44 2023
NLP_AS4_GROUP44

# Change the following lines in parser_utils.py with the absolute directory. This was necessary in Windows OS.
## LINE 36
data_path = 'C:/Users/USER_NAME/Documents/GitHub/NLP_A4_GROUP_44/NLP_A4_GROUP44/A4/data' 
# Change with local directory for the data folder 
LINE 40
embedding_file = 'C:/Users/USER_NAME/Documents/GitHub/NLP_A4_GROUP_44/NLP_A4_GROUP44/A4/data/en-cw.txt' 
# Change with local directory for 'en-cw.txt' 

# Output of the program in run.py
# LINE 186
output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.weights"