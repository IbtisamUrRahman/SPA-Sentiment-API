from flask import Flask
import pickle
from joblib import load
from nltk.stem.porter import PorterStemmer
import os
import re
from flask import request
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
app = Flask(__name__)


import torch
import torch.nn as nn
import torch.nn.functional as F
import unicodedata


questionList = ["How can you help me?", "What is opinion mining?", "What is text mining?", "How you perform analysis?","Which database you are using?",
                "Does you have mobile app?",
                "Which algorithms you are using?",
                "How SPA works?",
                "What are the features of SPA?",
                "What is SPA?"
                ]
answerList = ["I am here to perform sentiment analysis on your tweets and their replies. Just login to our application and enjoy services.", "Sentiment analysis, also referred to as opinion mining, is an approach to natural language processing (NLP) that identifies the emotional tone behind a body of text. It involves the use of data mining, machine learning (ML) and artificial intelligence.",
                "Text mining, also referred to as text data mining, roughly equivalent to text analytics, is the process of deriving high-quality information from text. High-quality information is typically derived through the devising of patterns and trends through means such as statistical pattern learning",
                "In order to perform analysis, awe have trained algorithms on the some dataset. After proper training and testing, SPA become able to predict the human behavior from the text.",
                "In order to handle large amount of dataset, we are using Mango DB.",
                "Yes we have Android application of SPA which perform sentiment analysis on the tweets, replies, user mentions and hashtags.",
                "We are using three different algorithms i.e. recurrent neural network, logistic regression and naïve Bayes.  We have got high accuracy from these algorisms.",
                "SPA categorizes the text into five categories i.e. appreciated, abusive, suggestive, serious concern, disappointed after performing sentiment analysis. For this purpose, three algorithms are used i.e. recurrent neural network, logistic regression and naïve Bayes.",
                "Perform sentiment analysis, mobile application, country wise and gender wise analysis, categorizes the text into five different human behaviors i.e. appreciated, abusive, suggestive, serious concern, disappointed. ",
                "SPA stands for Social Pocket Analyzer. It perform analysis on the tweets and replies of tweets. It also perform sentiment analysis on user mentions and hashtag tweets. It categorizes the text into five categories i.e. appreciated, abusive, suggestive, serious concern, disappointed."
                ]


device = torch.device("cpu")
MAX_LENGTH = 10  # Maximum sentence length

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))
        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens
        for word in keep_words:
            self.addWord(word)

def normalizeString(s):
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden

class GreedySearchDecoder(torch.jit.ScriptModule):
    def __init__(self, encoder, decoder, decoder_n_layers):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._device = device
        self._SOS_token = SOS_token
        self._decoder_n_layers = decoder_n_layers

    __constants__ = ['_device', '_SOS_token', '_decoder_n_layers']

    @torch.jit.script_method
    def forward(self, input_seq : torch.Tensor, input_length : torch.Tensor, max_length : int):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self._decoder_n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=self._device, dtype=torch.long) * self._SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self._device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self._device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


def evaluateExample(sentence, encoder, decoder, searcher, voc):
    try:
        print("> " + sentence)
        # Normalize sentence
        input_sentence = normalizeString(sentence)
        # Evaluate sentence
        output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        print('Bot:', ' '.join(output_words))
        return ' '.join(output_words)
		
    except:
        return 'Sorry, i\'m not supppose to tell you this'
        print("Error: Encountered unknown word.")	




def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]


@app.route('/logreg', methods=['POST'])
def index():
    text = request.form['text']
    result = LogisticReg.predict(
        [text])
    print(result)
    if result[0] == 1:
        return ('Appreciated')
    elif result[0] == 2:
        return ('Abusive')
    elif result[0] == 3:
        return ('Disappointed')
    elif result[0] == 4:
        return ('Serious Concern')
    elif result[0] == 5:
        return ('Suggestion')

@app.route('/naivebayes', methods=['POST'])
def index1():
    text = request.form['text']
    result = NBModel.predict(
        [text])
    print(result)
    if result[0] == 1:
        return ('Appreciated')
    elif result[0] == 2:
        return ('Abusive')
    elif result[0] == 3:
        return ('Disappointed')
    elif result[0] == 4:
        return ('Serious Concern')
    elif result[0] == 5:
        return ('Suggestion')
@app.route('/chatbot', methods=['POST'])
def chatbotRoute():
    text = request.form['text']
    probabilityDict = {}
    for question in questionList:
        matchedCount = 0
        for word in question.split(' '):
            if word in text:
                matchedCount += 1
        probabilityDict[questionList.index(question)] = matchedCount
    item = max(probabilityDict.keys(), key=(lambda key: probabilityDict[key]))
    if probabilityDict[item] >= round(len(text.split(' ')) - 0.3 * len(text.split(' '))):
        return answerList[item]
    else:
        botResponse = evaluateExample(text, traced_encoder, traced_decoder, scripted_searcher, voc)
        return botResponse

    

@app.route('/rnn', methods=['POST'])
def index2():
    
    # text = request.form['text']
    txt = [request.form['text']]
    # print(txt)
    sequences = []
    for word in txt[0].split():
        if word in wordDict:
            sequences.append(wordDict[word])
        else:
            sequences.append('1')


    if(len(sequences) > 130):
        print('greater than 130')
        sequences = sequences[:130]
    else:
        print('Less or equal than 130')
        sequences = ([0] * (130 - len(sequences))+ sequences)
    padded = np.array([sequences])
    # print(padded)
    pred = model.predict(padded)
    labels = ['','Appreciated', 'Abusive', 'Disappointed', 'Serious Concern', 'Suggestion']
    return labels[np.argmax(pred)]


if __name__ == '__main__':
    LogisticReg = pickle.load(open('models/LogisticRegressionModel.sav', 'rb'))
    NBModel = load('models/NaiveBayesModel.joblib')
    model = load_model('models/RNNmodel.h5')
    model._make_predict_function()
    wordDict = np.load('embedding/allWords.npy').item()

    save_dir = os.path.join("data", "save")
    corpus_name = "cornell movie-dialogs corpus"

    # Configure models
    model_name = 'cb_model'
    attn_model = 'dot'
    #attn_model = 'general'
    #attn_model = 'concat'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    # If you're loading your own model
    # Set checkpoint to load from
    checkpoint_iter = 4000
    # loadFilename = os.path.join(save_dir, model_name, corpus_name,
    #                             '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
    #                             '{}_checkpoint.tar'.format(checkpoint_iter))

    # If you're loading the hosted model
    loadFilename = 'data/4000_checkpoint.tar'

    # Load model
    # Force CPU device options (to match tensors in this tutorial)
    checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc = Voc(corpus_name)
    voc.__dict__ = checkpoint['voc_dict']


    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    # Load trained model params
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()
    print('Models built and ready to go!')




    ### Convert encoder model
    # Create artificial inputs
    test_seq = torch.LongTensor(MAX_LENGTH, 1).random_(0, voc.num_words)
    test_seq_length = torch.LongTensor([test_seq.size()[0]])
    # Trace the model
    traced_encoder = torch.jit.trace(encoder, (test_seq, test_seq_length))

    ### Convert decoder model
    # Create and generate artificial inputs
    test_encoder_outputs, test_encoder_hidden = traced_encoder(test_seq, test_seq_length)
    test_decoder_hidden = test_encoder_hidden[:decoder.n_layers]
    test_decoder_input = torch.LongTensor(1, 1).random_(0, voc.num_words)
    # Trace the model
    traced_decoder = torch.jit.trace(decoder, (test_decoder_input, test_decoder_hidden, test_encoder_outputs))

    ### Initialize searcher module
    scripted_searcher = GreedySearchDecoder(traced_encoder, traced_decoder, decoder.n_layers)


    print('scripted_searcher graph:\n', scripted_searcher.graph)

    scripted_searcher.save("scripted_chatbot.pth")


    app.run(debug=False)