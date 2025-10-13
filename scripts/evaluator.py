import re
import evaluate
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def is_equal_json(json1, json2):
    if type(json1) != type(json2):
        return False
    if isinstance(json1, dict):
        if set(json1.keys()) != set(json2.keys()):
            return False
        for key in json1.keys():
            # ignore extra whitespaces
            val1 = json1[key]
            val2 = json2[key]
            if type(val1)==str and type(val2)==str:
              val1 =  re.sub(r'\s+', ' ', val1.strip())
              val2 =  re.sub(r'\s+', ' ', val2.strip())
            if not is_equal_json(val1, val2):
                return False
        return True
    elif isinstance(json1, list):
        if len(json1) != len(json2):
            return False
        for item1, item2 in zip(json1, json2):
            if not is_equal_json(item1, item2):
                return False
        return True
    else:
        return json1 == json2


class Evaluator:
    def __init__(self, task_type, id2label=None):
        self.task_type = task_type
        self.id2label = id2label

        if self.task_type == "classification":
            self.eval = self.classification_eval
        elif self.task_type == "docvqa" or self.task_type == "ie":
            self.eval = self.exact_match_eval
    
    def cal_acc(self, prediction, label):
        return float(is_equal_json(prediction, label))

    def classification_eval(self, predictions, labels):
        # accuracy
        ems = [self.cal_acc(p, l) for p, l in zip(predictions, labels)]
        em = np.mean(ems)
        results = {"accuracy": em.item()}

        # precision + confusion matrix
        label_names = list(self.id2label.values())
        label2id = {label_name: int(idx) for idx, label_name in enumerate(label_names)}
        
        true_labels, model_preds = [], []
        for p, l in zip(predictions, labels):
            p_class = p['class']
            l_class = l['class']
            # undertrained model may output `<s_class> scientific report</s_class>` instead of `<s_class>scientific_report</s_class>`
            p["class"] = "_".join(p_class.split()) 
    
            if p_class in label2id and l_class in label2id:
                true_labels.append(label2id[l_class])
                model_preds.append(label2id[p_class])
            else:
                raise ValueError(f"Unknown class in prediction or label: pred {p_class}, label {l_class}")
        
        # confusion matrix
        confusion_metric = evaluate.load("confusion_matrix")
        result = confusion_metric.compute(references=true_labels, predictions=model_preds)
        conf_matrix = result['confusion_matrix']
        results['confusion_matrix'] = conf_matrix.tolist()
        # Create a DataFrame
        conf_matrix_df = pd.DataFrame(conf_matrix, index=label_names, columns=label_names)

        # Create a heatmap of the confusion matrix
        plt.figure(figsize=(8, 5))
        sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')

        precision_metric = evaluate.load("precision")
        result2 = precision_metric.compute(references=true_labels, predictions=model_preds, average=None)
        results["precision"] = { label_names[i]:p.item() for i, p in  enumerate(result2['precision'])}
        return results

    def exact_match_eval(self, predictions, labels):
        ems = [self.cal_acc(p, l) for p, l in zip(predictions, labels)]
        em = np.mean(ems)
        return {"accuracy": em.item()}
