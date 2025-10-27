import json
import numpy as np

def evaluate(true_data, predicted_data):
    fields = ['event_type', 'when', 'where']
    mae_fields = ['confidence', 'priority']
    em_count = 0
    total_em = 0
    exact_match = True
    results = {}


    for field in fields:
        correct = 0
        total = 0
        for true, predicted in zip(true_data, predicted_data):
            true_output = true["output"]
            predicted_output = predicted["output"]

            if true_output and predicted_output:
                true_value = str(true_output.get(field)).strip().lower()
                predicted_value = str(predicted_output.get(field)).strip().lower()

                if true_value == predicted_value:
                    correct += 1
                total += 1

        f1 = correct / total if total else 0

        results[field] = {
            "f1": round(f1,2),
            "correct": correct,
            "total": total,
        }
        # Since the f1 is working same as above so we have comment it out , and its an nlp generation prediction so here fp and fn will be same
        # false positive means model predicted positive but actually its negative , so if model predicted today but in reality its d+2 , what it wil be false postive or false negative ???
          #         if true_value == predicted_value:
        #             tp += 1
        #         else :
        #             fp += 1
        #             fn += 1

        # precision = tp / (tp + fp)
        # recall = tp / (tp + fn)
        # f1 = 2 * (precision * recall) / (precision + recall)

        # results[field] = {
        #     "f1": f1,
        # }

#   MAE
    for field in mae_fields:
        true_vals = []
        pred_vals = []

        for true, predicted in zip(true_data, predicted_data):
            t_out = true["output"]
            p_out = predicted.get("output")

            if t_out and p_out:
                t_val = t_out.get(field)
                p_val = p_out.get(field)
                true_vals.append(float(t_val) if float(t_val) else 0) 
                pred_vals.append(float(p_val) if float(p_val) else 0)


        mae = np.mean(np.abs(np.array(true_vals) - np.array(pred_vals)))
        results[field] = {"mae": float(mae), "count": len(true_vals)}

  #  Exact Match
    for true, predicted in zip(true_data, predicted_data):
        true_output = true["output"]
        predicted_output = predicted["output"]

        if true_output and predicted_output:
          exact_match = True

          for k in true_output.keys():
            t_val = str(true_output.get(k)).strip()
            p_val = str(predicted_output.get(k)).strip()

            if isinstance(t_val, list):
                if set(t_val) != set(p_val):
                    exact_match = False
                    break
            else:
                if str(t_val).strip() != str(p_val).strip():
                    exact_match = False
                    break
          if exact_match:
            results['exact_match'] = True

    results['exact_match'] = exact_match


    return results

with open("true_data.jsonl", "r", encoding="utf-8") as f:
    true_data = [json.loads(line) for line in f if line.strip()]

with open("predictions.json", "r") as f:
    predicted_data = json.load(f)

result = evaluate(true_data, predicted_data)
print(json.dumps(result, indent=2))