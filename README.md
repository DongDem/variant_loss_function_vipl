# variant_loss_function_vipl
Variant_loss_function_vipl

## Tensorflow version (old, maybe conflict)

- Comparison softmax loss, center loss and proposed loss in alex_net model
- The difference in the function `get_center_loss`
### Training
- Prepare the image path and label path (train set)
- Run file `MMI_with_proposed_loss_alex_net.py`

### Testing
- Get the accuracy and confusion matrix in the test set
- Run file `test_MMI_with_proposed_loss_alex_net.py`

## Pytorch version (Recommend)
- The difference in the function `VariantCenterLoss` in `variantcenterloss.py`
- Only in mnist dataset (example)

### Training
- Comment `test` phase and run file `pytorch_test.py` 

### Testing
- Comment `train` phase and run file `pytorch_test.py`



