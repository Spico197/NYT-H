# NYT-H

Datasets and codes for our COLING2020 paper: 

Towards Accurate and Consistent Evaluation: A Dataset for Distantly-Supervised Relation Extraction [[pdf]](https://www.aclweb.org/anthology/2020.coling-main.566.pdf)


## Dependencies
- python == 3.7.7
  - torch == 1.5.1
  - numpy == 1.18.5
  - sklearn == 0.21.3
  - tqdm == 4.36.1
  - matplotlib == 3.2.1

## Data
### Download

Download Links:
- Google Drive: [download](https://drive.google.com/file/d/1my4W7O-ioCYWRiP6VCbgGfXTzgdrDep0/view?usp=sharing)
- Baidu Netdisk: [download](https://pan.baidu.com/s/1X6bkyBgM9jKb2vAJ3JspGA) Code: `nyth`

Download the data from one of the links above, and paste the `nyt-h.tar.bz2` file in the main folder. Then extract data files from the tarball file.

```bash
$ tar jxvf nyt-h.tar.bz2
```

You will see a `data` folder has been extracted with a structure as below.

To reproduce the results in the paper, you can download the GloVe embedding from [here](http://nlp.stanford.edu/data/glove.6B.zip), and unzip it anywhere you want, and specify the filepath when running the program. `glove.6B.50d.txt` file is used as the default choice in this paper.

### Data Example
```python
{
    "instance_id": "NONNADEV#193662",
    "bag_id": "NONNADEV#91512",
    "relation": "/people/person/place_lived",
    "bag_label": "unk", # `unk` means the bag is not annotated, otherwise `yes` or `no`.
    "sentence": "Instead , he 's the kind of writer who can stare at the wall of his house in New Rochelle -LRB- as he did with '' Ragtime '' -RRB- , think about the year the house was built (1906) , follow his thoughts to the local tracks that once brought trolleys from New Rochelle to New York City and wind up with a book featuring Theodore Roosevelt , Scott Joplin , Emma Goldman , Stanford White and Harry Houdini . ''",
    "head": {
        "guid": "/guid/9202a8c04000641f8000000000176dc3",
        "word": "Stanford White",
        "type": "/influence/influence_node,/people/deceased_person" # type for entities, split by comma if one entity has many types
    },
    "tail": {
        "guid": "/guid/9202a8c04000641f80000000002f8906",
        "word": "New York City",
        "type": "/architecture/architectural_structure_owner,/location/citytown"
    }
}
```

### File Structure and Data Preparation
```
data
├── bag_label2id.json : bag annotation labels to numeric identifiers. `unk` label means the bag is not annotated, otherwise `yes` or `no`
├── rel2id.json : relation labels to numeric identifiers
├── na_train.json : NA instances for training to reproduce results in our paper
├── na_rest.json : rest of the NA instances
├── train_nonna.json : Non-NA instances for training (NO ANNOTATIONS)
├── dev.json : Non-NA instances for model selection during training (NO ANNOTATIONS)
└── test.json : Non-NA instances for final evaluation, including `bag_label` annotations
```

To get the full NA set:
```bash
$ cd data && cat na_rest.json na_train.json > na.json
```

To reproduce the results in our paper, combine the sampled NA instances(`na_train.json`) and `train_nonna.json` to get the train set:
```bash
$ cd data && cat train_nonna.json na_train.json > train.json
```

## Train & Evaluation for Reproducing the Results
We set 5 random seeds and take the average scores as the final results.
Please be aware that although we have set the random seed, the results may variants in different hardware architectures.

### Sent2Sent
```bash
python -u run.py \
    --description="NYT-H training with pcnn at sent2sent track" \
    --task_name="sent" \
    --model_name="sent_pcnn" \
    --data_dir="data/train" \
    --eval_model="best" \
    --overwrite_output_dir \
    --output_dir="outputs/sent_pcnn_422" \
    --embedding_path="</path/to/your/embeddings>/glove.6B.50d.txt" \
    --embedding_dim=50 \
    --seed=422 \
    --main_device="cuda:0" \
    --device_ids="[0]" \
    --epoch=50 \
    --batch_size=64 \
    --dropout_rate=0.5 \
    --lr=1e-3 \
    --max_len=150 \
    --pos_dis=75 \
    --pos_dim=5 \
    --save_best_model \
    --do_eval \
    --do_train \
    --select_score='non_na_macro_f1' \
    --tb_logging_step=1000
```

### Bag2Bag
```bash
python -u run.py \
    --description="NYT-H training with pcnn+att at bag level" \
    --task_name="bag" \
    --model_name="bag_pcnn_att" \
    --data_dir="data/train" \
    --eval_model="best" \
    --overwrite_output_dir \
    --output_dir="outputs/bag_pcnn_att_422" \
    --embedding_path="</path/to/your/embeddings>/glove.6B.50d.txt" \
    --embedding_dim=50 \
    --seed=422 \
    --main_device="cuda:0" \
    --device_ids="[0]" \
    --epoch=50 \
    --batch_size=64 \
    --dropout_rate=0.5 \
    --lr=1e-3 \
    --max_len=150 \
    --pos_dis=75 \
    --pos_dim=5 \
    --save_best_model \
    --do_eval \
    --do_train \
    --select_score='non_na_macro_f1' \
    --tb_logging_step=1000
```

### Bag2Sent
NOTICE: This track is for bag-model evaluation, so models must be trained in Bag2Bag track in advance.

```bash
python -u run.py \
    --description="NYT-H training with pcnn+att at bag level, eval at bag2sent level" \
    --task_name="bag2sent" \
    --model_name="bag_pcnn_att" \
    --data_dir="data/train" \
    --eval_model="best" \
    --overwrite_output_dir \
    --output_dir="outputs/bag_pcnn_att_422" \
    --embedding_path="</path/to/your/embeddings>/glove.6B.50d.txt" \
    --embedding_dim=50 \
    --seed=422 \
    --main_device="cuda:0" \
    --device_ids="[0]" \
    --batch_size=64 \
    --lr=1e-3 \
    --max_len=150 \
    --pos_dis=75 \
    --pos_dim=5 \
    --do_eval
```

## DSGT and MAGT Evaluation
If you train the model by using our codes and set `--do_eval` flag, then you can find all the evaluation results in the `log.log` file in the output directory.

If you want to check the DSGT and MAGT P/R/F1 scores directly, you should generate the result file with specific formats and type the command below.

```bash
python -u evaluate.py --track bag2sent \ # or bag2bag, sent2sent
                      --rel2id /path/to/your/rel2id/file \
                      --test /path/to/the/test.json/file \
                      --pred /path/to/your/prediction/result/file
```

Please check the `evaluate.py` file for more information on other metrics (AUC/Precision@K).

### Formats of Result Files
`pred` means the predicted relation of the bag/instance.

#### Sent2Sent or Bag2Sent
```python
{"instance_id": "TEST#123", "pred": "/business/company/founders"}
{"instance_id": "TEST#124", "pred": "/business/person/company"}
{"instance_id": "TEST#125", "pred": "/people/person/place_lived"}
...
```

#### Bag2Bag
```python
{"bag_id": "TEST#123", "pred": "/business/company/founders"}
{"bag_id": "TEST#124", "pred": "/business/person/company"}
{"bag_id": "TEST#125", "pred": "/people/person/place_lived"}
...
```

## Metrics
### AUC
Because of different AUC calculation strategies, you can see there are two kinds of AUC values if you check the log files.

For relation classification problem, assume the number of non-NA relation is `N_r`,
and the number of instances is `N`.

- `auc`: only the max probabilities are taken into consideration, so there are only `N` points on the Precision Recall Curve (PRC)
- `non_na_auc`: all the probabilities of non-NA relations are taken into consideration, so there are `N_r * N` points on the PRC

Here, we follow [OpenNRE](https://github.com/thunlp/OpenNRE) and report the `non_na_auc` as DSGT AUC. You can also get the MAGT AUC by checking `b_non_na_auc` in the logs.

### Non-NA Macro F1
You can get `non_na_macro_f1` and `dsgt_f1` if you run the program and check the logs. These two metrics are exactly the same. MAGT F1 scores are `magt_f1` in the logs.

All the models are selected via `non_na_macro_f1` in the paper.

### Precision@K
Precision@K (P@K) values are calculated in the Bag2Bag track.
Only bags with `yes` labels are taken into account, so the maximum `K` value is determined by the number of `yes` bags.

You can check the `b_P@50`, `b_P@100`, `b_P@300`, `b_P@500`, `b_P@1000` and `b_P@2000` values in the logs.

### PRC Plot
We use 5 random seeds and wish to get objective averaged results.
We cannot take the mean scores points on the PRCs directly, so all the line graphs are ploted by linear interpolation.

You can refer to `plot_prc.py` file for more details.

### Relation Coverage
All the relation coverage results can be found in `outputs/<output_dir>/eval/k_covered_rel.json` file.
Figure 4a in the paper is ploted via interpolation, while Figure 4b is ploted by average operation.

You can refer to `plot_prc.py` file for more details.

## Cite
```bibtex
@inproceedings{zhu-etal-2020-nyth,
    title = "Towards Accurate and Consistent Evaluation: A Dataset for Distantly-Supervised Relation Extraction",
    author = "Zhu, Tong  and
      Wang, Haitao  and
      Yu, Junjie  and
      Zhou, Xiabing  and
      Chen, Wenliang  and
      Zhang, Wei  and
      Zhang, Min",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.566",
    pages = "6436--6447",
}
```

## License
MIT
