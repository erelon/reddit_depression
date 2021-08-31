import _pickle as cpickle
import numpy as np
import torch
import tqdm
import json

from model import BertForMultiLabelClassification
from multilabel_pipeline import MultiLabelPipeline
from transformers import AutoTokenizer
from torch import tensor


def safe_split(p):
    if len(tokenizer.encode(p)) < 500:
        return [p]
    sen = p.split(".")
    p = [""]
    i = 0
    for s in sen:
        if len(tokenizer.encode(p[i])) + len(tokenizer.encode(s)) + 1 < 500:
            p[i] += s + "."
        else:
            i += 1
            p.append("")
            p[i] += s + "."
    return p


if __name__ == '__main__':
    # tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-ekman")
    # model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-ekman")

    tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
    model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")
    goemotions = MultiLabelPipeline(
        model=model,
        tokenizer=tokenizer,
        threshold=0.,
        device=-1 if torch.cuda.is_available() == False else 7
    )

    counter = torch.zeros(model.classifier.out_features, dtype=torch.int64)

    f = open("RSDD/training", "r")
    for _ in tqdm.tqdm(range(39309)):
        org_line = f.readline()
        if not org_line:
            break
        line = json.loads(org_line)
        all_emov = []
        all_posts = []
        all_timestamps = []
        for post in line[0]["posts"]:
            timestamp, p = post
            P = safe_split(p)
            for p in P:
                if len(tokenizer.encode(p)) > 512:
                    continue
                all_posts.append(p)
                all_timestamps.append(timestamp)

        emo_v = []
        for k in [all_posts[i:i + 50] for i in range(0, len(all_posts), 50)]:
            emo_v.extend(goemotions(k))

        for e, timestamp in zip(emo_v, all_timestamps):
            new_l = [timestamp, e["scores"]]
            counter[np.argmax(e["scores"])] += 1
            all_emov.append(new_l)

        n_post = {
            "posts": all_emov,
            "id": line[0]["id"],
            "label": line[0]["label"]
        }

        s = open("training_original", "ab")
        p = cpickle.Pickler(s)
        for_saving = p.dump(n_post)
        s.close()

    f.close()

    print(counter)

    # c = pickle.dumps(counter)
    f = open("training_original_counter.txt", "w")
    f.write(str(counter))
    f.write(str(emo_v["labels"]))
    f.close()
