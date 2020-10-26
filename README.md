# NYT-H

Source codes for the COLING2020 paper: 

Towards Accurate and Consistent Evaluation: A Dataset for Distantly-Supervised Relation Extraction

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
- Google Drive: [download](https://drive.google.com/file/d/1R7xGcbn8QKE_a1esBsam0zRCrZTFLIS9/view?usp=sharing)
- Baidu Netdisk: [download](https://pan.baidu.com/s/1413fhz_77zQjUMKFRtF_yw) Code: `nyth`

Download the data from one of the links above, and paste the `nyt-h.tar.bz2` file in the mail folder. Then extract data files from the tarball file.

```bash
$ tar jxvf nyt-h.tar.bz2
```

You will see a `data` folder has been extracted with a structure as below.

To reproduce the results in the paper, you can download the GloVe embedding from [here](http://nlp.stanford.edu/data/glove.6B.zip), and unzip it anywhere you want, and specify the filepath when running the program. `glove.6B.50d.txt` file is used as the default choice in this paper.

### File Structure
```
.
└── data
    ├── bag_label2id.json : bag annotation label to numeric identifiler
    ├── rel2id.json : relation label to numeric identifiler
    ├── release : all the NYT-H data includes NA set, train set and test set
    │   ├── na.json
    │   ├── test.json
    │   └── train.json
    └── train : dataset for reproducing the results in the paper. dev set is split from `release/train.json`, and half of the train set are NA instances. `bag_id` and `instance_id` are re-ordered
        ├── bag_label2id.json
        ├── dev.json
        ├── rel2id.json
        ├── test.json
        └── train.json
```

## Train
We set 5 random seeds and take the average scores as the final results. Please be aware that although we have set the random seed, the results may variants in different architectures.

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
```
@inproceedings{zhu-2020-nyth,
    title = "Towards Accurate and Consistent Evaluation: A Dataset for Distantly-Supervised Relation Extraction",
    author = "Zhu, Tong and
        Wang, Haitao and
        Yu, Junjie and
        Zhou, Xiabing and
        Chen, Wenliang and
        Zhang, Wei and
        Zhang, Min",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    year = "2020",
    publisher = "Association for Computational Linguistics",
}
```

## License
MIT
