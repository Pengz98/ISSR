#1 Make sure that all of the dependencies including the Open3D-ML are successfully installed.  
#2 Download the datasets (S3DIS or ScanNet) and check the 'dataset_path' in the main python file (InterPCSeg.py).
#3 Run the 'InterPCSeg.py' to start interactive semantic segmentation.
#3.1 The input scene, ground truth mask, error map and current mask would appear one by one.
#3.2 Users could put clicks on the error regions of current mask based on the error map, where the correct labels will automatically assigned to the clicked points.
#3.3 When the current mask has achieved expectance, users could turn to the next scene through double clicking on any point in the current mask.
