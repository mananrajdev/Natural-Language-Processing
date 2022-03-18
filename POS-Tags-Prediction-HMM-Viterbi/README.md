In the folder, I have attached a python file and a Jupyter notebook. The python file can be run directly using the command -
Python3 hmm_viterbi.py
<br>
The data folder should be present in the same directory.
<b> Data Preprocessing </b>
1. Convert all numeric values to a token: < num >. Numeric values can be like (1,23,456 or even 1.234). So, a regex is used to find numbers, comma, and decimal points and also to check if numbers follow the specific convention
2. All words which occur only once are converted to a special token: < unk >
<br><br>
<b>Task 1</b><br>
Vocabulary file is created after the above preprocessing is done. It is stored in vocab.txt file.<br>
<br>Questions –<br>
What is the selected threshold for unknown words replacement?<br>
Ans. 2<br>
What is the total size of your vocabulary?<br>
Ans. 38917<br>
What is the total occurrences of the special token '< unk >' after replacement?<br>
Ans. 17347<br>
What is the final size of your vocabulary?<br>
Ans. 21571<br>
<br>
<b>Task 2</b><br>
Emission dictionary is formed by first counting all the occurrence of tags into a dictionary<br>
Then counting the tag->word occurrence but counting the words not in vocab as unknown token.<br>
Emission Parameters: 28681<br>
For transmission dictionary, we find out the tags and its succeeding tags and their count and divide that with the count of individual tag count.<br>
Transition Parameters: 1378<br>
<br>
<b>Task 3</b><br>
A word tag dictionary is created to store all the tags a word has in the training file<br>
In the greedy algorithm, we first check if the word is in the vocabulary or not. If not, we convert it to unknown token.<br>
For a word in the dev/test file, we iterate over all the tags in the word tag dictionary and calculate its emission and transmission value. For the transmission value, if the word is the very first word, we calculate transmission of “.” And that tag else we calculate transmission of tag in previous state and current state. We multiple the emission and transmission values and store it. The tag for which this value is maximum is allotted to that word.<br>
<b>Accuracy on Dev Set: 93.74%</b><br>
<br>
<b>Task 4</b><br>
I maintained a result dictionary.<br>
In the Viterbi algorithm, I first checked if the word is the first word of the sentence and added the tag and its probability for that word. The backpointer to that word would be “start” as it is the first word. And for calculating probability I multiplied emission value with transmission value of “.” With that word.
For all other words, I calculated the probability of all possible tags of that word, I iterated over all the tags present in the previous stage and stored the maximum probability amongst those previous tag and stored that tag as the backpointer.<br>
If it is the last word of the sentence, I just add an “end” tag and the value calculated as its backpointer.<br>
Then I backpropogate through the dictionary and get the tags using the backpointers.<br>
<b>Accuracy on Dev Set: 95%</b><br>
