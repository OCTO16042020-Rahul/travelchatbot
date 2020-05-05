#==========================================================================
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
from flask import Flask, render_template, request

with open("qasystem_updated.json", encoding="utf8") as file:
    data = json.load(file)

m=0
userinfo=''
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            # print(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model11 = tflearn.DNN(net)

try:
    model11.load("model.tflearn")
except:
    model11.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model11.save("model.tflearn")

app = Flask(__name__)


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)
model11.load("model.tflearn")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    global m,userinfo
    userText = request.args.get('msg')
    userinfo=userinfo+" "+userText
    #m=5

    if m==0:
        outdatagot="Hi, welcome to Dora, <br>where you want to travel"
        m+=1
        return str(outdatagot)
    if m==1:
        outdatagot="what is your budget"
        m+=1
        return str(outdatagot)
    if m==2:
        outdatagot="how many number of peoples"
        m+=1
        return str(outdatagot)
    if m==3:
        outdatagot="is this family trip"
        m+=1
        return str(outdatagot)
    if m == 4:
        outdatagot = "what type of hotels do you want <br> AC <br> Luxury <br> Deluxe"
        m += 1
        return str(outdatagot)
    if m == 5:
        outdatagot = "Which type of places you wan't to  travel<br> Fort<br> Historical Places<br> Hill Station <br> Beach >"
        m += 1
        return str(outdatagot)
    if m == 6:
        outdatagot = "How many days you wan't to stay >"
        m += 1
        return str(outdatagot)
    if m == 7:
        print(userinfo)
        results = model11.predict([bag_of_words(userinfo, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        outdatagot = ""
        for tg in data["intents"]:
             if tg['tag'] == tag:
                 responses = tg['responses']
                 outdatagot = random.choice(responses)
                 m=0
                 print(outdatagot)

    return str(outdatagot)


if __name__ == "__main__":
    app.run("0.0.0.0")
#==========================================================================