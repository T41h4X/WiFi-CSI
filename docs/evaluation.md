# Evaluation Metrics & Performance Criteria

## 1. Metrics Defined

Because Human Activity Recognition often exhibits class imbalance, raw accuracy alone can be deceptive. Multiple complementary metrics are evaluated:

1. **Overall Accuracy**:
   $$\text{Accuracy} = \frac{\sum_{i=1}^C \text{TP}_i}{N_{\text{total}}}$$
2. **Macro-Averaged Precision**:
   $$\text{Precision}_{\text{macro}} = \frac{1}{C} \sum_{i=1}^C \frac{\text{TP}_i}{\text{TP}_i + \text{FP}_i}$$
3. **Macro-Averaged Recall**:
   $$\text{Recall}_{\text{macro}} = \frac{1}{C} \sum_{i=1}^C \frac{\text{TP}_i}{\text{TP}_i + \text{FN}_i}$$
4. **Macro-Averaged F1-Score**:
   $$\text{F1}_{\text{macro}} = \frac{1}{C} \sum_{i=1}^C \frac{2 \cdot \text{Precision}_i \cdot \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}$$
5. **Confusion Matrix**:
   $$C_{i,j} = \text{Count of samples belonging to ground-truth class } i \text{ predicted as class } j$$

---

## 2. Evaluation Protocol
* Models are evaluated strictly on the unseen **Test Split** ($N=284$), which is isolated from both the training and validation optimization loops.
* Best model checkpoints are selected based on the peak **Validation Macro-F1**, ensuring that minority classes are not sacrificed to inflate validation accuracy.
