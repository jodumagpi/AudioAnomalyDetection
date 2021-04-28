This is our project repository for the development of Detecting Anomalies in Audio Data.

Please refer to the following subfolders for better organization.

**sound_processing**
  * Python modules related to sound processing
  
**models**
  * Python modules related to AI/ML models for detecting audio anomalies

**data**
  * Train and Test dataset
  
-----
  
**HOW TO GENERATE YOUR OWN DATA**

```python
from sound_processing.sound_loader import SoundLoader
from sound_processing.feature_extractor import FeatureExtractor

# select the features you want to use (in this case, we just pick two features)
F = FeatureExtractor()
extractors = [F.spectral_centroid, F.rmse] # F.features if you want to use all instead

sl = SoundLoader('./sample/', 'labels.csv', 'reduced_noise.wav', extractors=extractors, seed=555)
dataset = sl.data_maker() # gives a dictionary of the inputs (numpy arrays) and corresponding labels

# if you want to get the dataloaders directly
loaders = sl.data_loader(dataset) # gives dataloaders ready for training and testing
```
