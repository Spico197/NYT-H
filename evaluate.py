#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import os
import pdb
import json
import warnings
from collections import OrderedDict
from functools import total_ordering
from itertools import combinations

import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm

warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)


@total_ordering
class Threshold(object):
    """For Precision-Recall Curve"""
    def __init__(self, threshold, label, true_label):
        self.th = threshold
        self.label = label
        self.true_label = true_label
        self.flag = int(self.label == self.true_label)
    
    def __eq__(self, obj):
        return self.th == obj.th

    def __lt__(self, obj):
        return self.th < obj.th


def compute_metrics_nyth(labels, preds, ids, target_names):
    r"""calculate the metrics of NYT-H dataset"""
    results = OrderedDict()
    results['acc'] = (preds == labels).mean()
    results['macro-f1'] = metrics.f1_score(labels, preds, average='macro')
    results['macro-recall'] = metrics.recall_score(labels, preds, average='macro')
    results['macro-precision'] = metrics.precision_score(labels, preds, average='macro')
    results['micro-f1'] = metrics.f1_score(labels, preds, average='micro')
    results['micro-recall'] = metrics.recall_score(labels, preds, average='micro')
    results['micro-precision'] = metrics.precision_score(labels, preds, average='micro')
    report = metrics.classification_report(labels, preds, 
                                            digits=4, labels=ids, 
                                            target_names=target_names, output_dict=True)
    rels = set(target_names)
    f1s = list()
    ps = list()
    rs = list()
    for key, val in report.items():
        if key in rels and key != 'NA':
            ps.append(val['precision'])
            rs.append(val['recall'])
            f1s.append(val['f1-score'])
    non_na_macro_precision = sum(ps)/len(ps)
    non_na_macro_recall = sum(rs)/len(rs)
    non_na_macro_f1 = sum(f1s)/len(f1s)
    results['non_na_macro_precision'] = non_na_macro_precision
    results['non_na_macro_recall'] = non_na_macro_recall
    results['non_na_macro_f1'] = non_na_macro_f1
    results['report'] = report
    return results


def evaluate_nyth(model, criterion, logger, processor, config, dataset_name, prefix=""):
    r"""evaluate the """
    eval_output_dir = os.path.join(config.output_dir, "eval")
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    number_of_total_examples = {
        'train': processor.num_train_examples,
        'dev': processor.num_dev_examples,
        'test': processor.num_test_examples,
    }
    logger.info(f"***** Running evaluation {prefix} *****")
    logger.info(f"  Num examples = {number_of_total_examples[dataset_name]}")
    results = dict()
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = list()
    labels = list()
    outs = list()
    tokens = list()
    instance_ids = list()
    bag_ids = list()

    data_loaders = {
        "train": processor.train_loader,
        "dev": processor.dev_loader,
        "test": processor.test_loader
    }
    data_loader = data_loaders[dataset_name]

    with torch.no_grad():
        model.eval()
        r"""opennre"""
        pred_result = list()
        r"""end of opennre"""

        for raw_batch in tqdm(data_loader, desc="Evaluating", ncols=60):
            if config.task_name == 'sent':
                batch = tuple(t.to(config.device) for t in raw_batch[:-2])
                rel_labels = batch[4]
                bag_labels = batch[5]
                instance_id = raw_batch[6]
                bag_id = raw_batch[7]
                inputs = {
                    "token2ids": batch[0],
                    "pos1s": batch[1],
                    "pos2s": batch[2],
                    "mask": batch[3],
                }
            elif config.task_name == 'bag':
                batch = tuple(t.to(config.device) for t in raw_batch[:-3])
                rel_labels = batch[4]
                bag_labels = batch[5]
                instance_id = raw_batch[6]
                bag_id = raw_batch[7]
                inputs = {
                    "token2ids": batch[0],
                    "pos1s": batch[1],
                    "pos2s": batch[2],
                    "mask": batch[3],
                    "scopes": raw_batch[8],
                    "is_training": False,
                    "rel_labels": rel_labels,
                }
            else:
                raise NotImplementedError

            instance_ids.extend(instance_id)
            bag_ids.extend(bag_id)

            out = model(**inputs)        
            loss = criterion(out, rel_labels)
            eval_loss += loss.item()
            nb_eval_steps += 1

            _, pred = torch.max(out, dim=1)  # replace softmax with max function, same results
            pred = pred.cpu().numpy().reshape((-1, 1))
            rel_labels = rel_labels.cpu().numpy().reshape((-1, 1))
            bag_labels = bag_labels.cpu().numpy().reshape((-1, 1))
            for x in batch[0].cpu().numpy():
                tokens.append(" ".join([processor.id2word[y] for y in x.tolist()]))
            if config.task_name == 'sent' or (config.task_name == 'bag' \
                and '_one' in config.model_name):
                softmax_out = torch.softmax(out.cpu().detach(), dim=-1)
            else:
                softmax_out = out.cpu().detach() # reference from opennre
            outs.append(softmax_out.numpy())
            preds.append(pred)
            labels.append(rel_labels)

            r"""opennre"""
            for i in range(softmax_out.size(0)):
                for relid in range(processor.class_num):
                    if processor.id2rel[relid] != 'NA':
                        pred_ins = {
                            'label': int(rel_labels[i].item() == relid), 
                            'score': softmax_out[i][relid].item(),
                            'pred_label': relid
                        }
                        if rel_labels[i].item() == relid:
                            if bag_labels[i] == 1:
                                pred_ins.update({"b_label": 1})
                            elif bag_labels[i] == 0:
                                pred_ins.update({"b_label": 0})
                        pred_result.append(pred_ins)
            r"""end of opennre"""
        
        eval_loss = eval_loss / nb_eval_steps

        outs = np.concatenate(outs, axis=0).astype(np.float32)
        preds = np.concatenate(preds, axis=0).reshape(-1).astype(np.int64)
        labels = np.concatenate(labels, axis=0).reshape(-1).astype(np.int64)

        id2rel = processor.id2rel
        ids = list(range(len(id2rel)))
        target_names = [id2rel[ind] for ind in ids]
        results = compute_metrics_nyth(labels, preds, ids, target_names)
        
        """Precision-Recall Curve (Ours)"""
        probs = torch.tensor(outs)
        # just take the probs in the max position
        thresholds, indices = probs[:,1:].max(dim=1)
        indices += 1

        ppp, rrr, _ = metrics.precision_recall_curve(labels==indices.cpu().detach().numpy(), thresholds)
        with open(os.path.join(eval_output_dir, 'prc_skprc_mine.json'), 'wt', encoding='utf-8') as fout:
            json.dump({'precision': ppp.tolist(), 'recall': rrr.tolist()}, fout, ensure_ascii=False)

        thresholds = thresholds.numpy()
        indices = indices.numpy()
        th_objs = list()
        for th, lb, truth in zip(thresholds, indices, labels):
            th_objs.append(Threshold(th, lb, truth))
        th_list_sorted = sorted(th_objs, reverse=True)
        tot_len = len(thresholds)

        correct = 0
        ps = list()
        rs = list()
        ths = list()
        for ind, th in enumerate(th_list_sorted):
            correct += th.flag
            ps.append(float(correct)/(ind + 1))
            rs.append(float(correct)/tot_len)
            ths.append(float(th.th))
        with open(os.path.join(eval_output_dir, "prc.json"), 'wt', encoding='utf-8') as fout:
            json.dump({
                "precision": ps,
                "recall": rs,
                "threshold": ths,
            }, fout, ensure_ascii=False)
        results['auc'] = metrics.auc(rs, ps)

        r"""opennre"""
        sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
        prec = []
        rec = []
        correct = 0
        # import ipdb; ipdb.set_trace()
        tot_count_flags = labels.copy()
        tot_count_flags[tot_count_flags > 0] = 1
        tot_count = int(tot_count_flags.sum())
        # take `all` non-na probs
        correct_k_with_rel = {"k": list(), "covered_rel": list()}
        correct_covered_rel = set()
        all_k_with_rel = {"k": list(), "covered_rel": list()}
        all_covered_rel = set()
        for i, item in enumerate(sorted_pred_result):
            correct += item['label']
            prec.append(float(correct) / float(i + 1))
            rec.append(float(correct) / float(tot_count))
            if item['label'] > 0:
                correct_covered_rel.add(item['pred_label'])
                correct_k_with_rel['k'].append(i + 1)
                correct_k_with_rel['covered_rel'].append(len(correct_covered_rel))
            all_covered_rel.add(item['pred_label'])
            all_k_with_rel['k'].append(i + 1)
            all_k_with_rel['covered_rel'].append(len(all_covered_rel))

        non_na_auc = metrics.auc(x=rec, y=prec)
        np_prec = np.array(prec)
        np_rec = np.array(rec)

        with open(os.path.join(eval_output_dir, "prc_opennre.json"), 'wt', encoding='utf-8') as fout:
            json.dump({
                "precision": prec,
                "recall": rec,
            }, fout, ensure_ascii=False)

        with open(os.path.join(eval_output_dir, "k_covered_rel.json"), 'wt', encoding='utf-8') as fout:
            json.dump({
                "correct": correct_k_with_rel,
                "all": all_k_with_rel,
            }, fout, ensure_ascii=False)

        max_f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
        mean_prec = np_prec.mean()
        results['non_na_auc'] = non_na_auc
        results['max_f1'] = max_f1
        results['mean_prec'] = mean_prec
        r"""end of opennre"""
        # -------------------------------------------------------------------------------------------------
        if dataset_name == 'test' and config.task_name == "bag":
            """opennre for bag_labels"""
            b_pred_result = list(filter(lambda x: "b_label" in x, pred_result))
            b_sorted_pred_result = sorted(b_pred_result, key=lambda x: x['score'], reverse=True)
            b_prec = []
            b_rec = []
            b_correct = 0
            b_tot_count = sum([x['b_label'] for x in b_sorted_pred_result])
            # take `all` non-na probs
            for i, item in enumerate(b_sorted_pred_result):
                b_correct += item['b_label']
                b_prec.append(float(b_correct) / float(i + 1))
                b_rec.append(float(b_correct) / float(b_tot_count))
                if i + 1 in [50, 100, 200, 300, 400, 500, 1000, 2000]:
                    results[f'b_P@{i + 1}'] = float(b_correct) / float(i + 1)
            b_non_na_auc = metrics.auc(x=b_rec, y=b_prec)
            np_b_prec = np.array(b_prec)
            np_b_rec = np.array(b_rec)
            with open(os.path.join(eval_output_dir, "b_prc_opennre.json"), 'wt', encoding='utf-8') as fout:
                json.dump({
                    "precision": b_prec,
                    "recall": b_rec,
                }, fout, ensure_ascii=False)

            b_max_f1 = (2 * np_b_prec * np_b_rec / (np_b_prec + np_b_rec + 1e-20)).max()
            b_mean_prec = np_b_prec.mean()
            results['b_non_na_auc'] = b_non_na_auc
            results['b_max_f1'] = b_max_f1
            results['b_mean_prec'] = b_mean_prec
            """end of opennre for bag_labels"""
        # -------------------------------------------------------------------------------------------------

        with open(os.path.join(eval_output_dir, 'eval_mc.txt'), 'wt', encoding='utf-8') as fin:
            for ins_id, bag_id, l, t, p in zip(instance_ids, bag_ids, labels, tokens, preds):
                rel2results = OrderedDict()
                for rel in processor.rel2id:
                    if rel != 'NA':
                        if rel == processor.id2rel[p]:
                            rel2results[rel] = True
                        else:
                            rel2results[rel] = False

                l = processor.id2rel[l]
                p = processor.id2rel[p]

                result = OrderedDict()
                result["instance_id"] = ins_id
                result["bag_id"] = bag_id
                result["result"] = str(l==p)
                result["label"] = l
                result["pred"] = p
                result["tokens"] = t
                result["rel2result"] = rel2results

                fin.write('{}\n'.format(json.dumps(result)))
        
        ds_p, ds_r, ds_f1 = compute_dsgt(labels, preds, processor.rel2id, verbose=False)
        results.update({"dsgt_p": ds_p, "dsgt_r": ds_r, "dsgt_f1": ds_f1})
        if dataset_name == 'test':
            idname = "bag_id" if config.task_name == "bag" else "instance_id"
            id2results = dict()
            with open(os.path.join(eval_output_dir, 'eval_mc.txt'), 'r', encoding='utf-8') as fin:
                for line in fin:
                    ins = json.loads(line)
                    id2results[ins[idname]] = ins
            ma_p, ma_r, ma_f1 = compute_magt(labels, preds, config.task_name, 
                processor.rel2id, processor.test_dataset.data, id2results, verbose=False)
            results.update({"magt_p": ma_p, "magt_r": ma_r, "magt_f1": ma_f1})

        logger.info("***** {} Eval results {} *****".format(dataset_name, prefix))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        
    return results, eval_loss, preds, labels, outs


def evaluate_crcnn(model, criterion, logger, processor, config, dataset_name, prefix=""):
    r"""evaluate the """
    eval_output_dir = os.path.join(config.output_dir, "eval")
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    number_of_total_examples = {
        'train': processor.num_train_examples,
        'dev': processor.num_dev_examples,
        'test': processor.num_test_examples,
    }
    logger.info(f"***** Running evaluation {prefix} *****")
    logger.info(f"  Num examples = {number_of_total_examples[dataset_name]}")
    results = dict()
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = list()
    labels = list()
    outs = list()
    tokens = list()
    instance_ids = list()
    bag_ids = list()

    data_loaders = {
        "train": processor.train_loader,
        "dev": processor.dev_loader,
        "test": processor.test_loader
    }
    data_loader = data_loaders[dataset_name]

    with torch.no_grad():
        model.eval()
        for raw_batch in tqdm(data_loader, desc="Evaluating", ncols=60):
            batch = tuple(t.to(config.device) for t in raw_batch[:-2])
            inputs = {
                "token2ids": batch[0],
                "pos1s": batch[1],
                "pos2s": batch[2],
                "mask": batch[3],
            }
            rel_labels = batch[4]
            bag_labels = batch[5]
            instance_id = raw_batch[6]
            bag_id = raw_batch[7]
            instance_ids.extend(instance_id)
            bag_ids.extend(bag_id)

            out = model(**inputs)
            loss = criterion(out, rel_labels)
            eval_loss += loss.item()
            nb_eval_steps += 1

            scores, pred = torch.max(out[:, 1:], dim=1)
            pred = pred + 1
            scores = scores.cpu().numpy().reshape((-1, 1))
            pred = pred.cpu().numpy().reshape((-1, 1))
            for i in range(pred.shape[0]):
                if scores[i][0] < 0:
                    pred[i][0] = 0
            
            rel_labels = rel_labels.cpu().numpy().reshape((-1, 1))
            for x in batch[0].cpu().numpy():
                tokens.append(" ".join([processor.id2word[y] for y in x.tolist()]))
            outs.append(out.detach().cpu().numpy())
            preds.append(pred)
            labels.append(rel_labels)
        
        eval_loss = eval_loss / nb_eval_steps

        outs = np.concatenate(outs, axis=0).astype(np.float32)
        preds = np.concatenate(preds, axis=0).reshape(-1).astype(np.int64)
        labels = np.concatenate(labels, axis=0).reshape(-1).astype(np.int64)
        
        id2rel = processor.id2rel
        ids = list(range(len(id2rel)))
        target_names = [id2rel[ind] for ind in ids]
        results = compute_metrics_nyth(labels, preds, ids, target_names)

        with open(os.path.join(eval_output_dir, 'eval_mc.txt'), 'wt', encoding='utf-8') as fin:
            for ins_id, bag_id, l, t, p in zip(instance_ids, bag_ids, labels, tokens, preds):
                rel2results = OrderedDict()
                for rel in processor.rel2id:
                    if rel != 'NA':
                        if rel == processor.id2rel[p]:
                            rel2results[rel] = True
                        else:
                            rel2results[rel] = False

                l = processor.id2rel[l]
                p = processor.id2rel[p]

                result = OrderedDict()
                result["instance_id"] = ins_id
                result["bag_id"] = bag_id
                result["result"] = str(l==p)
                result["label"] = l
                result["pred"] = p
                result["tokens"] = t
                result["rel2result"] = rel2results

                fin.write('{}\n'.format(json.dumps(result)))

        ds_p, ds_r, ds_f1 = compute_dsgt(labels, preds, processor.rel2id, verbose=False)
        results.update({"dsgt_p": ds_p, "dsgt_r": ds_r, "dsgt_f1": ds_f1})
        if dataset_name == 'test':
            id2results = dict()
            with open(os.path.join(eval_output_dir, 'eval_mc.txt'), 'r', encoding='utf-8') as fin:
                for line in fin:
                    ins = json.loads(line)
                    id2results[ins['instance_id']] = ins
            ma_p, ma_r, ma_f1 = compute_magt(labels, preds, config.task_name, 
                processor.rel2id, processor.test_dataset.data, id2results, verbose=False)
            results.update({"magt_p": ma_p, "magt_r": ma_r, "magt_f1": ma_f1})
        
        logger.info("***** {} Eval results {} *****".format(dataset_name, prefix))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        
    return results, eval_loss, preds, labels, outs


def evaluate_bag2sent(model, criterion, logger, processor, config, dataset_name, prefix=""):
    r"""evaluate the """
    eval_output_dir = os.path.join(config.output_dir, "eval")
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    logger.info(f"***** Running evaluation {prefix} *****")
    logger.info(f"  Num examples = {processor.num_test_examples}")
    results = dict()
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = list()
    labels = list()
    outs = list()
    tokens = list()
    instance_ids = list()
    bag_ids = list()

    data_loaders = {
        "test": processor.test_loader
    }
    data_loader = data_loaders[dataset_name]

    with torch.no_grad():
        r"""opennre"""
        pred_result = list()
        r"""end of opennre"""

        for raw_batch in tqdm(data_loader, desc="Evaluating", ncols=60):
            model.eval()
            if config.task_name == 'bag2sent':
                batch = tuple(t.to(config.device) for t in raw_batch[:-3])
                rel_labels = batch[4]
                bag_labels = batch[5]
                instance_id = raw_batch[6]
                bag_id = raw_batch[7]
                inputs = {
                    "token2ids": batch[0],
                    "pos1s": batch[1],
                    "pos2s": batch[2],
                    "mask": batch[3],
                    "scopes": raw_batch[8],
                    "is_training": False,
                    "rel_labels": rel_labels,
                }
                instance_ids.extend(instance_id)
                bag_ids.extend(bag_id)
            else:
                raise NotImplementedError

            out = model(**inputs)
            loss = criterion(out, rel_labels)
            eval_loss += loss.item()
            nb_eval_steps += 1

            _, pred = torch.max(out, dim=1)  # replace softmax with max function, same results
            pred = pred.cpu().numpy().reshape((-1, 1))
            rel_labels = rel_labels.cpu().numpy().reshape((-1, 1))
            for x in batch[0].cpu().numpy():
                tokens.append(" ".join([processor.id2word[y] for y in x.tolist()]))
            outs.append(out.detach().cpu().numpy())
            preds.append(pred)
            labels.append(rel_labels)

            r"""opennre"""
            for i in range(out.size(0)):
                for relid in range(processor.class_num):
                    if processor.id2rel[relid] != 'NA':
                        pred_result.append({
                            'label': int(rel_labels[i].item() == relid), 
                            'score': out[i][relid].item()
                        })
            r"""end of opennre"""
        
        eval_loss = eval_loss / nb_eval_steps

        outs = np.concatenate(outs, axis=0).astype(np.float32)
        preds = np.concatenate(preds, axis=0).reshape(-1).astype(np.int64)
        labels = np.concatenate(labels, axis=0).reshape(-1).astype(np.int64)
        
        id2rel = processor.id2rel
        ids = list(range(len(id2rel)))
        target_names = [id2rel[ind] for ind in ids]
        results = compute_metrics_nyth(labels, preds, ids, target_names)

        """Precision-Recall Curve"""
        probs = torch.tensor(outs)
        thresholds, indices = probs[:,1:].max(dim=1)
        indices += 1
        thresholds, indices = probs.max(dim=1)
        thresholds = thresholds.numpy()
        indices = indices.numpy()
        th_objs = list()
        for th, lb, truth in zip(thresholds, indices, labels):
            th_objs.append(Threshold(th, lb, truth))
        th_list_sorted = sorted(th_objs, reverse=True)
        tot_len = len(thresholds)

        correct = 0
        ps = list()
        rs = list()
        ths = list()
        for ind, th in enumerate(th_list_sorted):
            correct += th.flag
            ps.append(float(correct)/(ind + 1))
            rs.append(float(correct)/tot_len)
            ths.append(float(th.th))
        with open(os.path.join(eval_output_dir, "prc_bag2sent.json"), 'wt', encoding='utf-8') as fout:
            json.dump({
                "precision": ps,
                "recall": rs,
                "threshold": ths,
            }, fout, ensure_ascii=False)
        results['auc'] = metrics.auc(rs, ps)

        r"""opennre"""
        sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
        prec = []
        rec = []
        correct = 0
        total = processor.num_test_examples
        for i, item in enumerate(sorted_pred_result):
            correct += item['label']
            prec.append(float(correct) / float(i + 1))
            rec.append(float(correct) / float(total))
        non_na_auc = metrics.auc(x=rec, y=prec)
        np_prec = np.array(prec)
        np_rec = np.array(rec) 
        max_f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
        mean_prec = np_prec.mean()
        results['non_na_auc'] = non_na_auc
        results['max_f1'] = max_f1
        results['mean_prec'] = mean_prec
        r"""end of opennre"""

        with open(os.path.join(eval_output_dir, 'eval_mc_bag2sent.txt'), 'wt', encoding='utf-8') as fin:
            for ins_id, bag_id, l, t, p in zip(instance_ids, bag_ids, labels, tokens, preds):
                rel2results = OrderedDict()
                for rel in processor.rel2id:
                    if rel != 'NA':
                        if rel == processor.id2rel[p]:
                            rel2results[rel] = True
                        else:
                            rel2results[rel] = False

                l = processor.id2rel[l]
                p = processor.id2rel[p]

                result = OrderedDict()
                result["instance_id"] = ins_id
                result["bag_id"] = bag_id
                result["result"] = str(l==p)
                result["label"] = l
                result["pred"] = p
                result["tokens"] = t
                result["rel2result"] = rel2results

                fin.write('{}\n'.format(json.dumps(result)))
        
        ds_p, ds_r, ds_f1 = compute_dsgt(labels, preds, processor.rel2id, verbose=False)
        results.update({"dsgt_p": ds_p, "dsgt_r": ds_r, "dsgt_f1": ds_f1})
        if dataset_name == "test":
            id2results = dict()
            with open(os.path.join(eval_output_dir, 'eval_mc_bag2sent.txt'), 'r', encoding='utf-8') as fin:
                for line in fin:
                    ins = json.loads(line)
                    id2results[ins['instance_id']] = ins
            ma_p, ma_r, ma_f1 = compute_magt(labels, preds, "bag2sent", 
                processor.rel2id, processor.test_dataset.data, 
                id2results, verbose=False)
            results.update({"magt_p": ma_p, "magt_r": ma_r, "magt_f1": ma_f1})
        
        logger.info("***** {} Eval results {} *****".format(dataset_name, prefix))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

    return results, eval_loss, preds, labels, outs


def compute_dsgt(labels, preds, rel2id, verbose=True):
    ids = list(range(0, len(rel2id)))
    id2rel = {val: key for key, val in rel2id.items()}
    target_names = [id2rel[x] for x in ids]
    report = metrics.classification_report(labels, preds, 
                                            digits=4, labels=ids, 
                                            target_names=target_names, output_dict=True)
    rels = set(target_names)
    f1s = list()
    ps = list()
    rs = list()
    for key, val in report.items():
        if key in rels and key != 'NA':
            ps.append(val['precision'])
            rs.append(val['recall'])
            f1s.append(val['f1-score'])
    non_na_macro_precision = sum(ps)/len(ps)
    non_na_macro_recall = sum(rs)/len(rs)
    non_na_macro_f1 = sum(f1s)/len(f1s)
    if verbose:
        logger.info("DS Non-NA Macro Precision: {:.3f}".format(non_na_macro_precision*100))
        logger.info("DS Non-NA Macro Recall: {:.3f}".format(non_na_macro_recall*100))
        logger.info("DS Non-NA Macro F1: {:.3f}".format(non_na_macro_f1*100))
    return non_na_macro_precision, non_na_macro_recall, non_na_macro_f1


def compute_magt(labels, preds, track, rel2id, test, id2results, verbose=True):
    id_names = {
        "sent": "instance_id",
        "bag2sent": "instance_id",
        "bag": "bag_id",
    }
    id_name = id_names[track]
    # initialization
    id2ins = dict()
    rel2ids = dict()
    rel2yes_ids = dict()
    for rel in rel2id:
        if rel != 'NA':
            rel2ids[rel] = list()
            rel2yes_ids[rel] = list()
            
    for ins in test:
        # create index
        id2ins[ins[id_name]] = ins
        rel2ids[ins['relation']].append(ins[id_name])
        if ins['bag_label'] == 'yes':
            rel2yes_ids[ins['relation']].append(ins[id_name])
    
    # without yes instances from other relations
    stat_results = dict()
    for rel in rel2id:
        if rel != 'NA':
            stat_results[rel] = dict(preds=list(), labels=list())

    for ins in test:
        if ins['bag_label'] == 'yes':
            stat_results[ins['relation']]['labels'].append(1)
        else:
            stat_results[ins['relation']]['labels'].append(0)
        stat_results[ins['relation']]['preds'].append(int(id2results[ins[id_name]]['rel2result'][ins['relation']]))
    
    for rel in stat_results:
        assert len(stat_results[rel]['preds']) == len(stat_results[rel]['labels'])
    
    rel2f1s = dict()
    for rel in stat_results:
        rel2f1s[rel] = dict(precision=0.0, recall=0.0, f1_score=0.0)

    for rel in stat_results:
        rel2f1s[rel]['f1_score'] = metrics.f1_score(stat_results[rel]['labels'], stat_results[rel]['preds'])
        rel2f1s[rel]['precision'] = metrics.precision_score(stat_results[rel]['labels'], stat_results[rel]['preds'])
        rel2f1s[rel]['recall'] = metrics.recall_score(stat_results[rel]['labels'], stat_results[rel]['preds'])

    precisions = [x['precision'] for x in rel2f1s.values()]
    macro_precision = sum(precisions)/len(precisions)
    recalls = [x['recall'] for x in rel2f1s.values()]
    macro_recall = sum(recalls)/len(recalls)
    f1s = [x['f1_score'] for x in rel2f1s.values()]
    macro_f1 = sum(f1s) / len(f1s)
    if verbose:
        logger.info("Macro Precision: {:.3f}".format(macro_precision*100))
        logger.info("Macro Recall: {:.3f}".format(macro_recall*100))
        logger.info("Macro F1: {:.3f}".format(macro_f1*100))
    return macro_precision, macro_recall, macro_f1


if __name__ == "__main__":
    """setup logging module"""
    import logging
    import sys
    logger = logging.getLogger('new_lib_logger')
    logger.setLevel(logging.INFO)
    # combination_type = sys.argv[1]
    # log_path = f"combinations{combination_type}.txt"
    log_path = "log.log"
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fmt = "[%(asctime)-15s]-%(levelname)s-%(filename)s-%(lineno)d-%(process)d: %(message)s"
    datefmt = "%a %d %b %Y %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    with open('datav5/rel2id.json', 'r', encoding='utf-8') as fin:
        rel2id = json.load(fin)

    with open('datav5/fix_test_bag_v5.json', 'r', encoding='utf-8') as fin:
        test = json.load(fin)
    
    # eval_dir_base = '/data4/tzhu/sent_cnn/output_dir_rs_{}/bag_bigru_att/eval/'
    eval_dir_base = '/data4/tzhu/nyth_codes/data_debug/bag_pcnn_att_{}/eval/'
    # eval_dir = '/data4/tzhu/bert_re_torch/nyth_bert_marker_auc/eval/'
    eval_level = 'bag2sent'
    seeds = [422, 4227, 42270, 422701, 4227019]
    dsgts_p = list(); dsgts_r = list(); dsgts_f1 = list()
    magts_p = list(); magts_r = list(); magts_f1 = list()


    for seed in seeds:
        eval_dir = eval_dir_base.format(seed)
        logger.info(f"{eval_level} - {eval_dir.split('/')[-3]}")
        if eval_level == 'sent':
            id2results = dict()
            labels = list()
            preds = list()
            with open(os.path.join(eval_dir, 'eval_mc.txt'), 'r', encoding='utf-8') as fin:
                for line in fin:
                    ins = json.loads(line)
                    id2results[ins['instance_id']] = ins
                    labels.append(rel2id[ins['label']])
                    preds.append(rel2id[ins['pred']])
            
            # original non-NA f1 after distant supervision training
            print(' -------------------- DS Eval --------------------')
            ds_p, ds_r, ds_f1 = compute_dsgt(labels, preds, rel2id, verbose=True)
            print(' -------------------- MC Eval --------------------')
            ma_p, ma_r, ma_f1 = compute_magt(labels, preds, "sent", rel2id, test, id2results, verbose=True)
        elif eval_level == 'bag':
            id2results = dict()
            labels = list()
            preds = list()
            with open(os.path.join(eval_dir, 'eval_mc.txt'), 'r', encoding='utf-8') as fin:
                for line in fin:
                    ins = json.loads(line)
                    id2results[ins['bag_id']] = ins
                    labels.append(rel2id[ins['label']])
                    preds.append(rel2id[ins['pred']])
            print(' -------------------- DS Eval --------------------')
            ds_p, ds_r, ds_f1 = compute_dsgt(labels, preds, rel2id, verbose=True)
            print(' -------------------- MC Eval --------------------')
            ma_p, ma_r, ma_f1 = compute_magt(labels, preds, "bag", rel2id, test, id2results, verbose=True)
        elif eval_level == 'bag2sent':
            id2results = dict()
            labels = list()
            preds = list()
            with open(os.path.join(eval_dir, 'eval_mc_bag2sent.txt'), 'r', encoding='utf-8') as fin:
                for line in fin:
                    ins = json.loads(line)
                    id2results[ins['instance_id']] = ins
                    labels.append(rel2id[ins['label']])
                    preds.append(rel2id[ins['pred']])
            print(' -------------------- DS Eval --------------------')
            ds_p, ds_r, ds_f1 = compute_dsgt(labels, preds, rel2id, verbose=True)
            print(' -------------------- MC Eval --------------------')
            ma_p, ma_r, ma_f1 = compute_magt(labels, preds, "bag2sent", rel2id, test, id2results, verbose=True)
        else:
            raise ValueError

        dsgts_p.append(ds_p)
        dsgts_r.append(ds_r)
        dsgts_f1.append(ds_f1)
        magts_p.append(ma_p)
        magts_r.append(ma_r)
        magts_f1.append(ma_f1)

    print(' -------------------- AVERAGE --------------------')
    logger.info("MEAN DSGT Macro Precision: {:.3f}".format(sum(dsgts_p)/len(dsgts_p)*100))
    logger.info("MEAN DSGT Macro Recall: {:.3f}".format(sum(dsgts_r)/len(dsgts_r)*100))
    logger.info("MEAN DSGT Macro F1: {:.3f}".format(sum(dsgts_f1)/len(dsgts_f1)*100))

    logger.info("MEAN MAGT Macro Precision: {:.3f}".format(sum(magts_p)/len(magts_p)*100))
    logger.info("MEAN MAGT Macro Recall: {:.3f}".format(sum(magts_r)/len(magts_r)*100))
    logger.info("MEAN MAGT Macro F1: {:.3f}".format(sum(magts_f1)/len(magts_f1)*100))
