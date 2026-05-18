# Ran on kaggle so dataset was present there 
import torch
import time
import json
import os
import numpy as np 
import pandas as pd 
import pickle
import matplotlib.pyplot as plt
from utils.py import getTopKNeighbors
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

file = "/kaggle/input/cc-news-subset/updated_vocab_document_dict.json"

# Loading using json as cannot load directly using pandas (getting all arrays must be of same length issue)
import json
with open(file, "r", encoding="utf-8") as f:
    data = json.load(f)

vocab = list(data.keys())

docs = []

for k in data:
    for i in data[k]:
        text = i[1]
        docs.append(text)

len(docs)

# Converthing to pandas DataFrame to work furthur
df = pd.DataFrame({"text": docs})
df.head()


# we are building the corpus from the raw data and then converting them to dataframe (initialy we loaded using json)
# Its easier to manipulate with panas DF
docs = []
for key in data:
    for item in data[key]:
        
        text = item[1]
        docs.append(text)

df = pd.DataFrame({"text": docs})

print("Total documents:", len(df))

# maps so we don't need to iterate all the context for a center word, we can directly update the contexts in window
N = len(vocab)
word2id = {}
id2word = {}

for i, word in enumerate(vocab):
    word2id[word] = i
    id2word[i] = word


# just removing the newlien char with normal space, as numbers and stopwords were there in vocab so doesn't apply those also not all tokens were lowercase
def text_preprocess(text):
    text = text.replace("\n", " ")
    tokens = text.split()
    return tokens

df["tokens"] = df["text"].apply(text_preprocess)
df.head()


# Function defined for making the corr matrix on fly useful when running with hyperparameter i.e 
def makeCorrMatrix(winSize):
    print("Currently building the Corr-Matrix with Window Size:-", winSize)
    corrMat = {}
    
    for doc in df["tokens"]:
        tokenIds = []

        for word in doc:
            if word in word2id:
                tokenIds.append(word2id[word])

        for center in range(len(tokenIds)):

            l = max(0, center - winSize) # these min and max are to ensure that we don't go out of bound 
            r = min(len(tokenIds), center + winSize + 1)

            for context in range(l, r):

                if center == context:
                    continue

                i = tokenIds[center]
                j = tokenIds[context]

                pair = (i, j)

                if pair in corrMat:
                    corrMat[pair] += 1
                else:
                    corrMat[pair] = 1

    print("No. of pairs which co-occur are (non zero):", len(corrMat))

    return corrMat


#function to train so its easy to call while looping over dimensions or other hyper-params

def trainGloVe(expId, pairs, cnts, N, d, lr, epochs, xmax, alpha):

    start = time.time()
#check-pointing is added as after crashes our training can be resume from there as each experiments are taking a lot of time.
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/{expId}.pkl"

    if os.path.exists(ckpt_path):

        print("Resuming experiment:", expId)

        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)

        W = ckpt["W"]
        Wt = ckpt["Wt"]
        b = ckpt["b"]
        bt = ckpt["bt"]
        startEpoch = ckpt["epoch"]
        losses = ckpt["losses"]

    else:

        print("Starting new experiment:", expId)

        W = np.random.randn(N, d) * 0.01
        Wt = np.random.randn(N, d) * 0.01
        b = np.zeros(N)
        bt = np.zeros(N)

        startEpoch = 0
        losses = []

    for epoch in range(startEpoch, epochs):

        totalLoss = 0

        for idx in range(len(pairs)):

            i = pairs[idx, 0]
            j = pairs[idx, 1]

            wi = W[i].copy()
            wj = Wt[j].copy()

            bi = b[i]
            bj = bt[j]

            logX = np.log(cnts[idx])
            weight_val = (cnts[idx] / xmax) ** alpha if cnts[idx] < xmax else 1.0

            totalLoss += weight_val * (
                np.dot(wi, wj) + bi + bj - logX
            ) ** 2

            W[i] = W[i] - lr * weight_val * (
                np.dot(wi, wj) + bi + bj - logX
            ) * wj

            Wt[j] = Wt[j] - lr * weight_val * (
                np.dot(wi, wj) + bi + bj - logX
            ) * wi

            b[i] = b[i] - lr * weight_val * (
                np.dot(wi, wj) + bi + bj - logX
            )

            bt[j] = bt[j] - lr * weight_val * (
                np.dot(wi, wj) + bi + bj - logX
            )

        losses.append(totalLoss)

        print("Epoch:", epoch, "Loss:", totalLoss)
        
        with open(ckpt_path, "wb") as f:  #we are also saving the trained W for each d so we can use that in task 4
            pickle.dump({
                "W": W,
                "Wt": Wt,
                "b": b,
                "bt": bt,
                "epoch": epoch + 1,
                "losses": losses
            }, f)

    latency = time.time() - start

    return {
        "losses": losses,
        "latency": latency
    }, W

# arrays defined for hyper-param tuning
windows = [3, 4, 5]
lrs = [0.005, 0.001, 0.0001]
epochs = 10

alpha = 0.75
xmax = 100
d = 200

# traning for all the values so to check which one performs better..!
results = []

for w in windows:

    co = makeCorrMatrix(w)

    print("Converting co-occurrence matrix to numpy arrays")
    pairs = np.array(list(co.keys()), dtype=np.int64)
    counts = np.array(list(co.values()), dtype=np.float32)
    print("Total training pairs:", len(pairs))

    for lr in lrs:

        expId = f"w{w}_lr{lr}_d{d}"

        res, W = trainGloVe(
            expId,
            pairs,
            counts,
            N,
            d,
            lr,
            epochs,
            xmax,
            alpha
        )

        results.append({
            "id": expId,
            "window": w,
            "lr": lr,
            "epochs": epochs,
            "latency": res["latency"],
            "final_loss": res["losses"][-1]
        })

        np.save(f"glove_{expId}.npy", W)

        print("Finished:", expId)
        print("Final Loss:", res["losses"][-1])
        print("-" * 40)

# for r in results:
#     print(r)


#running for the final best values foundfrom above with all dimensions

dimensions = [50, 100, 200, 300]

finalResult = {}

co = makeCorrMatrix(3)

print("Converting co-occurrence matrix to numpy arrays")
pairs = np.array(list(co.keys()), dtype=np.int64)
counts = np.array(list(co.values()), dtype=np.float32)
print("Total training pairs:", len(pairs))

for d in dimensions:

    expId = f"final_d{d}"

    res, W = trainGloVe(expId, pairs, counts, N, d, 0.005, 15, xmax, alpha)
    print(res)

    finalResult[d] = res
    np.save(f"glove_W_{d}.npy", W)

#plotting the losses for visualization using matplotlib
for d in dimensions:

    losses = finalResult[d]["losses"]

    plt.plot(losses, label=f"d={d}")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curves for Different Dimensions")
plt.legend()
plt.show()

#Using the utility function getTopKNeighbors from utiltiy
for d in dimensions:
    W = np.load(f"/kaggle/input/glove-saved-w/glove_W_{d}.npy")
    print(f"\nDimension {d}")
    print(getTopKNeighbors("King", W, word2id, id2word, k=5))

for d in dimensions:
    W = np.load(f"/kaggle/input/glove-saved-w/glove_W_{d}.npy")
    print(f"\nDimension {d}")
    print(getTopKNeighbors("Cricket", W, word2id, id2word, k=5))


for d in dimensions:
    W = np.load(f"/kaggle/input/glove-saved-w/glove_W_{d}.npy")
    print(f"\nDimension {d}")
    print(getTopKNeighbors("College", W, word2id, id2word, k=5))

for d in dimensions:
    W = np.load(f"/kaggle/input/glove-saved-w/glove_W_{d}.npy")
    print(f"\nDimension {d}")
    print(getTopKNeighbors("Apple", W, word2id, id2word, k=5))