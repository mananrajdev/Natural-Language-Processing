<h1><b>Description</b></h1>

This assignment gives me hands-on experience on building deep learning models on Named Entity Recognition (NER). We will use the CoNLL-2003 corpus to build a neural network for NER. <br>
The files train and dev have sentences with human-annotated NER tags. In the file of test, there are only the raw sentences. The data format is that, each line contains three items separated by a white space symbol. The first item is the index of the word in the sentence. The second item is the word type and the third item is the corresponding NER tag. There will be a blank line at the end of one sentence.

<b>Task 1 - Simple Bidirectional LSTM model</b> <br>
Implementing the bidirectional LSTM network with PyTorch. The architecture of the network is: <br>
Embedding → BLSTM → Linear → ELU → classifier <br>
The hyper-parameters of the network are listed in the following table: <br>
<table style="width: 100%;">
    <tbody>
        <tr>
            <td style="width: 50.0000%;">
                <p>embedding dim</p>
            </td>
            <td style="width: 50.0000%;">100<br></td>
        </tr>
        <tr>
            <td style="width: 50.0000%;">number of LSTM layers<br></td>
            <td style="width: 50.0000%;">1</td>
        </tr>
        <tr>
            <td style="width: 50.0000%;">LSTM hidden dim<br>LSTM Dropout<br></td>
            <td style="width: 50.0000%;">256<br>0.33</td>
        </tr>
        <tr>
            <td style="width: 50.0000%;">Linear output dim<br></td>
            <td style="width: 50.0000%;">128</td>
        </tr>
    </tbody>
</table>

<br>
This simple BLSTM model is trained with the training data on NER with SGD as the optimizer and the parameters are tuned. 
<br>


<b>Task 2 - Using GloVe word embeddings</b> <br>
The second task is to use the GloVe word embeddings to improve the BLSTM in Task 1. The way we use the GloVe word embeddings is straight forward: we initialize the embeddings in our neural network with the corresponding vectors in GloVe.
<br>

<b>Task 3 - LSTM-CNN model</b> <br>
The task is to equip the BLSTM model in Task 2 with a CNN module to capture character-level information. The character embedding dimension is set to 30. I tuned other hyper-parameters of CNN module, such as the number of CNN layers, the kernel size and output dimension of each CNN layer.
