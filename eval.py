import json
import numpy as np

def evaluate(true_data, predicted_data):
    fields = ['event_type', 'when', 'where']
    mae_fields = ['confidence', 'priority']
    exact_match = True
    results = {}
    results['F1'] = {}


    for field in fields:
        tp = {}
        fp = {}
        fn = {}
        field_results = {}

        for true, predicted in zip(true_data, predicted_data):
            true_output = true["output"]
            predicted_output = predicted["output"]

            if true_output and predicted_output:
                true_value = str(true_output.get(field)).strip().lower()
                predicted_value = str(predicted_output.get(field)).strip().lower()

                if true_value == predicted_value:
                    if true_value not in tp:
                        tp[true_value] = 0
                    tp[true_value] += 1
                else:
                    if true_value not in fn:
                        fn[true_value] = 0
                    fn[true_value] += 1
                    if predicted_value not in fp:
                        fp[predicted_value] = 0
                    fp[predicted_value] += 1

        all_labels = set(tp.keys()) | set(fp.keys()) | set(fn.keys())
        f1_list = []

        for label in all_labels:
            t = tp.get(label, 0)
            f = fp.get(label, 0)
            n = fn.get(label, 0)

            precision = t / (t + f) if (t + f) > 0 else 0.0
            recall = t / (t + n) if (t + n) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            f1_list.append(float(f1))
            field_results[label] = {
                "f1": round(f1, 2),
                "tp": t,
                "fp": f,
                "fn": n
            }

        results[field] = field_results

        avg_f1 = np.mean(f1_list) if f1_list else 0.0
        results['F1'][field] = float(round(avg_f1, 2))

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