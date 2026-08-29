#Load modules
import joblib
from tensorflow import keras

#Shared code - paths, test images
from src.config import OUTPUT_DIR, MODEL_DIR
from src.data import load_test_images

#Fills in the gender and usage columns of styles_prediction.csv.
#articleType and season belong to the other tasks and are left as they are.

X_test, sub = load_test_images()
print('Test images:', X_test.shape)

#Load the encoders that were fitted on the training labels. Refitting here would
#reorder the classes and turn every prediction into a different word.
le_gender = joblib.load(MODEL_DIR / 'label_encoder_gender.joblib')
le_usage = joblib.load(MODEL_DIR / 'label_encoder_usage.joblib')

#The plain full-vocabulary CNNs - the merged-class models predict Other,
#which is not a label the submission accepts
cnn_gender = keras.models.load_model(MODEL_DIR / 'cnn_gender.keras')
cnn_usage = keras.models.load_model(MODEL_DIR / 'cnn_usage.keras')

sub['gender'] = le_gender.inverse_transform(cnn_gender.predict(X_test).argmax(axis=1))
sub['usage'] = le_usage.inverse_transform(cnn_usage.predict(X_test).argmax(axis=1))

#Checks on the deliverable itself - these are the mistakes that pass silently
assert len(sub) == 5829, 'expected 5829 rows, got %d' % len(sub)
assert sub['gender'].notna().all()
assert sub['usage'].notna().all()
assert sub['gender'].isin(le_gender.classes_).all()
assert sub['usage'].isin(le_usage.classes_).all()

print('\nPredicted gender counts:')
print(sub['gender'].value_counts())
print('\nPredicted usage counts:')
print(sub['usage'].value_counts())

path = OUTPUT_DIR / 'styles_prediction_task3.csv'
sub.to_csv(path, index=False)
print('\nsaved', path)
print(sub.head())
