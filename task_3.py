#!pip install datasets==3.6.0  #Installed locally
#!pip install sklearn-crfsuite

from datasets import load_dataset
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from collections import defaultdict

dataset = load_dataset("conll2003")

def word_features(sentence, i):
    word = sentence[i]

    features = {
        'bias': 1.0,
        'word.lower': word.lower(),
        'word.isupper': word.isupper(),
        'word.istitle': word.istitle(), #Nouns
        'word.isdigit': word.isdigit(), #Anything number

        'prefix_1': word[:1],
        'prefix_2': word[:2],
        'prefix_3': word[:3],

        'suffix_1': word[-1:],
        'suffix_2': word[-2:],
        'suffix_3': word[-3:],
    }

    if i > 0:    #Prev word
        prev_word = sentence[i-1]
        features.update({
            'prev_word.lower': prev_word.lower(),
            'prev_word.istitle': prev_word.istitle(),
            'prev_word.isupper': prev_word.isupper(),
        })
    else:
        features['BOS'] = True #Start of sentence

    if i < len(sentence) - 1:    #Next word
        next_word = sentence[i+1]
        features.update({
            'next_word.lower': next_word.lower(),
            'next_word.istitle': next_word.istitle(),
            'next_word.isupper': next_word.isupper(),
        })
    else:
        features['EOS'] = True #End of sentence

    return features

label_list = dataset["train"].features["ner_tags"].feature.names

def sentence_to_labels(x): #Convert tags for training
    return [label_list[tag] for tag in x["ner_tags"]]

def sentence_to_features(x):
    return [word_features(x['tokens'], i) #for each word in sentence
            for i in range(len(x['tokens']))]

X_train = [sentence_to_features(x) for x in dataset["train"]]
y_train = [sentence_to_labels(x) for x in dataset["train"]]

X_test = [sentence_to_features(x) for x in dataset["test"]]
y_test = [sentence_to_labels(x) for x in dataset["test"]]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(X_train, y_train)

y_pred = crf.predict(X_test)

info = crf.state_features_

def top_state_features(state_features, top_n=20):
    features_by_label = defaultdict(list)

    for (feature, label), weight in state_features.items(): #Regrouping weights
        features_by_label[label].append((weight, feature))

    for label in features_by_label: #Sorting and printing
        print(f"\nTop positive features for {label}:")
        for weight, feature in sorted(features_by_label[label], reverse=True)[:top_n]:
            print(f"{weight:6.3f}  {feature}")

print(top_state_features(crf.state_features_))


##Evaluation for Q4 comparison

eval_labels = list(crf.classes_)
eval_labels.remove('O') #Remove inflated accuracy

acc = metrics.flat_accuracy_score(y_test, y_pred)

macro_f1 = metrics.flat_f1_score(y_test, y_pred, average='macro', labels=eval_labels)

print(f"Overall Accuracy: {acc:.4f}")
print(f"Macro-F1 Score:   {macro_f1:.4f}")


