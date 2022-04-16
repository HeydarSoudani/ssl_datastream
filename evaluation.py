import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


def final_step_evaluation(results, base_labels, known_labels, k=(1, 5,), eps=1e-8):
  results = np.array(results, dtype=[
    ('true_label', np.int32),
    ('predicted_label', np.int32),
    ('real_novelty', np.bool),
    ('detected_novelty', np.bool)
  ])
  print('Final step, base labels: {}'.format(base_labels))
  print('Final step, known labels: {}'.format(known_labels))

  ## == Close World Classification Accuracy, CwCA ===
  # === or ==========================================
  ## == Base classes Classification Accuracy, BcCA ==
  base_results = results[np.isin(results['true_label'], list(base_labels))]
  CwCA = accuracy_score(
    base_results['true_label'],
    base_results['predicted_label']
  )

  ## == Novel classes Classification Accuracy, NcCA ==
  novel_results = results[np.isin(results['true_label'], list(base_labels), invert=True)]
  NcCA = accuracy_score(
    novel_results['true_label'],
    novel_results['predicted_label']
  )

  ## == All classes Classification Accuracy, ACA =====
  AcCA = accuracy_score(
    results['true_label'],
    results['predicted_label']
  )

  ## == Open World Classification Accuracy, OwCA =====
  temp1 = results[np.isin(results['predicted_label'], list(base_labels), invert=True)]
  temp2 = results[np.isin(results['predicted_label'], list(base_labels))]
  temp1['predicted_label'] = -1
  ow_results = np.concatenate((temp1, temp2))

  temp1 = ow_results[np.isin(ow_results['true_label'], list(base_labels), invert=True)]
  temp2 = ow_results[np.isin(ow_results['true_label'], list(base_labels))]
  temp1['true_label'] = -1
  ow_results = np.concatenate((temp1, temp2))

  OwCA = accuracy_score(
    ow_results['true_label'],
    ow_results['predicted_label']
  )

  ## == Unknown (Novel) Detection Accuracy (UDA) =====
  temp1 = results[np.isin(results['predicted_label'], list(base_labels), invert=True)]
  temp2 = results[np.isin(results['predicted_label'], list(base_labels))]
  temp1['detected_novelty'] = True
  nov_results = np.concatenate((temp1, temp2))
  temp1 = nov_results[np.isin(nov_results['true_label'], list(base_labels), invert=True)]
  temp2 = nov_results[np.isin(nov_results['true_label'], list(base_labels))]
  temp1['real_novelty'] = True
  nov_results = np.concatenate((temp1, temp2))

  real_novelties = nov_results[nov_results['real_novelty']]
  detected_novelties = nov_results[nov_results['detected_novelty']]
  detected_real_novelties = nov_results[nov_results['detected_novelty'] & nov_results['real_novelty']]
  
  tp = len(detected_real_novelties)
  fp = len(detected_novelties) - len(detected_real_novelties)
  fn = len(real_novelties) - len(detected_real_novelties)
  tn = len(results) - tp - fp - fn
  M_new = fn / (tp + fn + eps)
  F_new = fp / (fp + tn + eps)

  return CwCA, NcCA, AcCA, OwCA, M_new, F_new
