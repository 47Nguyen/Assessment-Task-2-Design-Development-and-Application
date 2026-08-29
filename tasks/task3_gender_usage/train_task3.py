#Load modules
import re
import time
import joblib
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#Modelling - LR, RF, tuning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from tensorflow import keras
from tensorflow.keras import layers

#Shared code - paths, data, hand-made features, model, metrics
from src.config import SEED, IMG_SHAPE, OUTPUT_DIR, CACHE_DIR, MODEL_DIR
from src.data import get_split, load_metadata, load_images, normalise
from src.features import hog_features, colour_hist
from src.models import build_cnn, default_callbacks
from src.evaluate import evaluate_model, per_class_report, plot_confusion, class_weights

epochs = 30
batch_size = 128


#Classical baselines - HOG + colour histogram, LogisticRegression and RandomForest
#The CNN learns its own filters. These two models get told what to look at
#instead - HOG for shape and edges, an HSV histogram for colour - so the
#comparison is "learned features vs hand-made features", not just two scores.
def run_classical_baselines():
    grid = {'n_estimators': [100, 300],
            'max_depth': [None, 20],
            'min_samples_leaf': [1, 5]}

    for target in ['gender', 'usage']:
        print('\nTarget:', target)

        #Raw pixels, not the CNN's standardised ones - HOG and the histogram
        #expect the original 0-255 image
        X_train, X_val, y_train, y_val, le = get_split(target, normalised=False)

        #HOG over 38k images takes a few minutes, so keep it next to the image cache
        cache_path = CACHE_DIR / ('features_%s.npz' % target)

        if cache_path.exists():
            cached = np.load(cache_path)
            features_train, features_val = cached['train'], cached['val']
            print('Loaded features from', cache_path.name)
        else:
            print('Building features for', len(X_train), 'train and', len(X_val), 'val images')
            t = time.time()
            features_train = np.hstack([hog_features(X_train), colour_hist(X_train)])
            features_val = np.hstack([hog_features(X_val), colour_hist(X_val)])
            print('feature time (s):', round(time.time() - t, 1))
            np.savez(cache_path, train=features_train, val=features_val)

        print('Feature vector length:', features_train.shape[1])

        #Logistic regression
        lr = LogisticRegression(max_iter=1000)

        t = time.time()
        lr.fit(features_train, y_train)
        lr_time = time.time() - t
        print('\nlogreg fit time (s):', round(lr_time, 1))

        evaluate_model(y_val, lr.predict(features_val), target, 'logreg_hog_colour',
                       notes='HOG + HSV histogram, max_iter 1000, fit %.1f s' % lr_time)

        #Random forest, tuned on macro-F1 because accuracy is dominated by the big classes
        #cv=3 rather than 5, and the trees parallelised instead of the search, because
        #copying a 30890 x 2000 feature matrix into 8 worker processes runs out of memory
        rf = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                          grid, scoring='f1_macro', cv=3)

        t = time.time()
        rf.fit(features_train, y_train)
        rf_time = time.time() - t
        print('\nrf fit time (s):', round(rf_time, 1))
        print('Best params:', rf.best_params_)
        print('Best cv macro-F1: %.4f' % rf.best_score_)

        evaluate_model(y_val, rf.predict(features_val), target, 'rf_hog_colour_tuned',
                       notes='HOG + HSV histogram, grid search picked %s, fit %.1f s'
                             % (rf.best_params_, rf_time))

    print('\nDone. Results are in outputs/results.csv.')


#Gender - plain CNN, class-weighted CNN, simplified 3-class (Men / Women / Other)
def run_gender_models():
    target = 'gender'

    #Load the data
    X_train, X_val, y_train, y_val, le = get_split(target)

    print('\nTarget:', target)
    print('Train images:', X_train.shape)
    print('Val images:', X_val.shape)
    print('Classes:', list(le.classes_))

    #Baseline - always predict the biggest class
    #Taken from train, not val, because at prediction time we can't see the val labels
    most_common = np.bincount(y_train).argmax()
    print('\nMost common class in train:', le.classes_[most_common])

    y_pred_baseline = np.full(len(y_val), most_common)

    evaluate_model(y_val, y_pred_baseline, target, 'baseline_majority',
                   notes='always predicts the biggest class')

    #Plain CNN
    print('\nTraining the plain CNN')

    cnn = build_cnn(n_classes=len(le.classes_))
    cnn.summary()

    cnn.fit(X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=default_callbacks(target))   #saves models/cnn_gender.keras

    #Probabilities -> class id
    y_pred_cnn = cnn.predict(X_val).argmax(axis=1)

    evaluate_model(y_val, y_pred_cnn, target, 'cnn_baseline',
                   notes='build_cnn defaults, lr 1e-3, batch %d' % batch_size)

    #Class-weighted CNN
    #Makes the rare classes cost more, so accuracy should drop and macro-F1 should rise
    print('\nTraining the class-weighted CNN')

    weights = class_weights(y_train)
    print('\nClass weights:')
    for class_id, weight in weights.items():
        print(' ', le.classes_[class_id], round(weight, 2))

    cnn_weighted = build_cnn(n_classes=len(le.classes_))

    cnn_weighted.fit(X_train, y_train,
                     validation_data=(X_val, y_val),
                     epochs=epochs,
                     batch_size=batch_size,
                     class_weight=weights,
                     callbacks=default_callbacks(target + '_weighted'))

    y_pred_weighted = cnn_weighted.predict(X_val).argmax(axis=1)

    evaluate_model(y_val, y_pred_weighted, target, 'cnn_weighted',
                   notes='same CNN, balanced class weights')

    #Simplified 3-class - Men / Women / Other
    #Unisex is about intent, and Boys/Girls differ by size, which a 60x80 crop can't show
    print('\nTraining the simplified 3-class model (Men / Women / Other)')

    simple_map = {'Men': 'Men',
                  'Women': 'Women',
                  'Unisex': 'Other',
                  'Boys': 'Other',
                  'Girls': 'Other'}

    #5-class numbers -> 3-class words
    def to_simple(y_numbers):
        words = le.inverse_transform(y_numbers)
        return np.array([simple_map[w] for w in words])

    le_simple = LabelEncoder().fit(['Men', 'Other', 'Women'])

    y_train_simple = le_simple.transform(to_simple(y_train))
    y_val_simple = le_simple.transform(to_simple(y_val))

    print('Simplified classes:', list(le_simple.classes_))

    cnn_simple = build_cnn(n_classes=len(le_simple.classes_))

    cnn_simple.fit(X_train, y_train_simple,
                   validation_data=(X_val, y_val_simple),
                   epochs=epochs,
                   batch_size=batch_size,
                   callbacks=default_callbacks(target + '_simple'))

    y_pred_simple = cnn_simple.predict(X_val).argmax(axis=1)

    #Fair comparison, both scored as 3 classes
    #3-class and 5-class macro-F1 aren't comparable because the denominators differ,
    #so fold the 5-class model's answers down to 3 and score both in the same space
    print('\nFair comparison, both scored as 3 classes')

    y_pred_cnn_folded = le_simple.transform(to_simple(y_pred_cnn))

    evaluate_model(y_val_simple, y_pred_cnn_folded, 'gender_3class', 'cnn_5class_folded',
                   notes='trained on 5 classes, answers merged down to 3 afterwards')

    evaluate_model(y_val_simple, y_pred_simple, 'gender_3class', 'cnn_trained_as_3',
                   notes='trained directly on Men/Women/Other')

    #Per-class tables, worst recall first
    print('\nPer-class results (worst recall first)')

    #Drop the summary rows so they don't look like real classes
    def real_classes_only(table):
        return table.drop(index=[i for i in ('micro avg', 'samples avg') if i in table.index])

    for name, preds in [('cnn_baseline', y_pred_cnn), ('cnn_weighted', y_pred_weighted)]:
        table = real_classes_only(per_class_report(y_val, preds, le))
        print('\n---', name, '---')
        print(table.round(3))
        table.to_csv(OUTPUT_DIR / ('per_class_%s_%s.csv' % (target, name)))

    table_simple = real_classes_only(per_class_report(y_val_simple, y_pred_simple, le_simple))
    print('\n--- cnn_trained_as_3 ---')
    print(table_simple.round(3))
    table_simple.to_csv(OUTPUT_DIR / ('per_class_%s_simple.csv' % target))

    #Confusion matrices
    #Watch whether Boys falls into Men and Girls into Women
    plot_confusion(y_val, y_pred_cnn, le, target)
    plot_confusion(y_val, y_pred_weighted, le, target + '_weighted')
    plot_confusion(y_val_simple, y_pred_simple, le_simple, target + '_3class')

    print('\nDone. Results are in outputs/results.csv, figures in outputs/.')


#Usage - plain CNN, sqrt-damped-capped weighted CNN, simplified 5-class
#Casual is 77% of the data, so always guessing Casual gives 0.769 accuracy while
#the macro-F1 ceiling is only about 0.50. Always report both.
def run_usage_models():
    target = 'usage'

    #Load the data
    #get_split() drops the 72 rows with no usage label
    X_train, X_val, y_train, y_val, le = get_split(target)

    print('\nTarget:', target)
    print('Train images:', X_train.shape)
    print('Val images:', X_val.shape)
    print('Classes:', list(le.classes_))

    #Home has 1 image in total, so it can't be scored - worth showing
    print('\nHow many of each class are in each split:')
    for class_id, class_name in enumerate(le.classes_):
        n_train = int((y_train == class_id).sum())
        n_val = int((y_val == class_id).sum())
        flag = '   <- not in val, cannot be scored' if n_val == 0 else ''
        print(' ', class_name, 'train', n_train, ' val', n_val, flag)

    #Baseline - always predict the biggest class
    most_common = np.bincount(y_train).argmax()
    print('\nMost common class in train:', le.classes_[most_common])

    y_pred_baseline = np.full(len(y_val), most_common)

    evaluate_model(y_val, y_pred_baseline, target, 'baseline_majority',
                   notes='always predicts Casual - high accuracy, useless model')

    #Plain CNN
    print('\nTraining the plain CNN')

    cnn = build_cnn(n_classes=len(le.classes_))
    cnn.summary()

    cnn.fit(X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=default_callbacks(target))   #saves models/cnn_usage.keras

    y_pred_cnn = cnn.predict(X_val).argmax(axis=1)

    evaluate_model(y_val, y_pred_cnn, target, 'cnn_baseline',
                   notes='build_cnn defaults, lr 1e-3, batch %d' % batch_size)

    #Class-weighted CNN, sqrt-damped and capped
    #Full "balanced" weighting gives Home (1 image) a weight near 4800, which
    #collapses training below the majority baseline - confirmed, see results.csv
    #row usage/cnn_weighted (accuracy 0.199). Damp with sqrt and cap at 10 instead.
    print('\nTraining the class-weighted CNN')

    raw_weights = class_weights(y_train)
    weights = {k: min(np.sqrt(v), 10.0) for k, v in raw_weights.items()}
    print('\nClass weights (sqrt-damped, capped at 10):')
    for class_id, weight in weights.items():
        print(' ', le.classes_[class_id], round(weight, 2))

    cnn_weighted = build_cnn(n_classes=len(le.classes_))

    cnn_weighted.fit(X_train, y_train,
                     validation_data=(X_val, y_val),
                     epochs=epochs,
                     batch_size=batch_size,
                     class_weight=weights,
                     callbacks=default_callbacks(target + '_weighted'))

    y_pred_weighted = cnn_weighted.predict(X_val).argmax(axis=1)

    evaluate_model(y_val, y_pred_weighted, target, 'cnn_weighted_capped',
                   notes='same CNN, sqrt-damped class weights capped at 10 '
                         '(uncapped balanced weights fail, see cnn_weighted row above)')

    #Simplified 5-class
    #Smart Casual, Travel, Party and Home are under 0.3% of the data between them
    print('\nTraining the simplified 5-class model')

    mapping = {'Casual': 'Casual',
               'Sports': 'Sports',
               'Ethnic': 'Ethnic',
               'Formal': 'Formal',
               'Smart Casual': 'Other',
               'Travel': 'Other',
               'Party': 'Other',
               'Home': 'Other'}

    #8-class numbers -> 5-class words
    def to_simple(y_numbers):
        words = le.inverse_transform(y_numbers)
        return np.array([mapping[w] for w in words])

    le_simple = LabelEncoder().fit(['Casual', 'Ethnic', 'Formal', 'Other', 'Sports'])

    y_train_simple = le_simple.transform(to_simple(y_train))
    y_val_simple = le_simple.transform(to_simple(y_val))

    print('Simplified classes:', list(le_simple.classes_))

    cnn_simple = build_cnn(n_classes=len(le_simple.classes_))

    cnn_simple.fit(X_train, y_train_simple,
                   validation_data=(X_val, y_val_simple),
                   epochs=epochs,
                   batch_size=batch_size,
                   callbacks=default_callbacks(target + '_simple'))

    y_pred_simple = cnn_simple.predict(X_val).argmax(axis=1)

    #Fair comparison, both scored as 5 classes
    #8-class and 5-class macro-F1 aren't comparable, so fold the 8-class model's
    #answers down to 5 and score both in the same space
    print('\nFair comparison, both scored as 5 classes')

    y_pred_cnn_folded = le_simple.transform(to_simple(y_pred_cnn))

    evaluate_model(y_val_simple, y_pred_cnn_folded, 'usage_5class', 'cnn_8class_folded',
                   notes='trained on 8 classes, answers merged down to 5 afterwards')

    evaluate_model(y_val_simple, y_pred_simple, 'usage_5class', 'cnn_trained_as_5',
                   notes='trained directly on the 5 merged classes')

    #Per-class tables, worst recall first
    #The tiny classes show up here as a row of zeros - that's the case for merging
    print('\nPer-class results (worst recall first)')

    #Drop the summary rows so they don't look like real classes
    def real_classes_only(table):
        return table.drop(index=[i for i in ('micro avg', 'samples avg') if i in table.index])

    for name, preds in [('cnn_baseline', y_pred_cnn), ('cnn_weighted', y_pred_weighted)]:
        table = real_classes_only(per_class_report(y_val, preds, le))
        print('\n---', name, '---')
        print(table.round(3))
        table.to_csv(OUTPUT_DIR / ('per_class_%s_%s.csv' % (target, name)))

    table_simple = real_classes_only(per_class_report(y_val_simple, y_pred_simple, le_simple))
    print('\n--- cnn_trained_as_5 ---')
    print(table_simple.round(3))
    table_simple.to_csv(OUTPUT_DIR / ('per_class_%s_simple.csv' % target))

    #Confusion matrices
    #Expect a strong vertical stripe on Casual swallowing the other classes
    plot_confusion(y_val, y_pred_cnn, le, target)
    plot_confusion(y_val, y_pred_weighted, le, target + '_weighted')
    plot_confusion(y_val_simple, y_pred_simple, le_simple, target + '_5class')

    print('\nDone. Results are in outputs/results.csv, figures in outputs/.')


#Multitask - one shared CNN trunk with a gender head and a usage head
#Gender and usage are the same photo asked two questions, so one trunk can learn
#the clothing features once and two small heads can read off both answers. The
#question is whether sharing helps, hurts, or makes no difference.
def run_multitask():
    #Same frozen split as get_split(), but keep only rows that have both labels.
    #Usage is missing on 72 rows and gender on none, so this costs about 72 images
    #out of 38.6k - not enough to make the comparison unfair.
    meta = load_metadata(verbose=False)
    meta = meta[meta['gender'].notna() & meta['usage'].notna()].reset_index(drop=True)

    tr = meta[meta['_split'] == 'train']
    va = meta[meta['_split'] == 'val']

    print('Train rows:', len(tr))
    print('Val rows:', len(va))

    #Load the encoders the single-task scripts already fitted and saved,
    #so the class ids mean the same thing in every results.csv row
    le_gender = joblib.load(MODEL_DIR / 'label_encoder_gender.joblib')
    le_usage = joblib.load(MODEL_DIR / 'label_encoder_usage.joblib')

    X_train = load_images(tr['id'], 'train')
    X_val = load_images(va['id'], 'train')

    X_train, stats = normalise(X_train)
    X_val, _ = normalise(X_val, stats)

    gender_train = le_gender.transform(tr['gender'])
    gender_val = le_gender.transform(va['gender'])
    usage_train = le_usage.transform(tr['usage'])
    usage_val = le_usage.transform(va['usage'])

    #Same blocks as build_cnn(), which is single-output, split into two heads at the end
    keras.utils.set_random_seed(SEED)

    inputs = keras.Input(shape=IMG_SHAPE, name='image')

    x = inputs
    for f in (32, 64, 128):
        x = layers.Conv2D(f, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    gender_head = layers.Dense(len(le_gender.classes_), activation='softmax', name='gender')(x)
    usage_head = layers.Dense(len(le_usage.classes_), activation='softmax', name='usage')(x)

    multitask = keras.Model(inputs, [gender_head, usage_head], name='cnn_multitask')

    multitask.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss={'gender': 'sparse_categorical_crossentropy',
                            'usage': 'sparse_categorical_crossentropy'},
                      metrics={'gender': 'accuracy', 'usage': 'accuracy'})
    multitask.summary()

    stopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                            restore_best_weights=True, verbose=1)

    t = time.time()
    multitask.fit(X_train, {'gender': gender_train, 'usage': usage_train},
                  validation_data=(X_val, {'gender': gender_val, 'usage': usage_val}),
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=[stopper])
    fit_time = time.time() - t

    multitask.save(MODEL_DIR / 'cnn_multitask.keras')

    print('\nmultitask fit time (s):', round(fit_time, 1))
    print('The two single-task CNNs were not timed when they ran, so compare this by')
    print('re-running run_gender_models() and run_usage_models() with a timer if the report needs it.')

    pred_gender, pred_usage = multitask.predict(X_val)

    evaluate_model(gender_val, pred_gender.argmax(axis=1), 'gender', 'cnn_multitask',
                   notes='shared trunk with a gender and a usage head, fit %.1f s' % fit_time)

    evaluate_model(usage_val, pred_usage.argmax(axis=1), 'usage', 'cnn_multitask',
                   notes='shared trunk with a gender and a usage head, fit %.1f s' % fit_time)

    #Per-class, to see whether sharing rescued any of the rare classes
    print('\nPer-class results (worst recall first)')

    print('\n--- gender ---')
    print(per_class_report(gender_val, pred_gender.argmax(axis=1), le_gender).round(3))

    print('\n--- usage ---')
    print(per_class_report(usage_val, pred_usage.argmax(axis=1), le_usage).round(3))

    print('\nDone. Results are in outputs/results.csv.')


#Post-hoc analysis - calibration curves, error-image grids, cost table
#A macro-F1 number says which model won but not whether it can be trusted.
#Three things it doesn't say: whether the confidence it reports means anything,
#what the mistakes actually look like, and what the score cost to buy.
def run_analysis():
    results = pd.read_csv(OUTPUT_DIR / 'results.csv')

    #Where each model's weights ended up, and which output head to read.
    #The merged-class models (gender_3class, usage_5class) are analysis only - they
    #answer a different question and can't be submitted - so they never appear here.
    saved_models = {'cnn_baseline': ('cnn_%s.keras', None),
                    'cnn_weighted': ('cnn_%s_weighted.keras', None),
                    'cnn_weighted_capped': ('cnn_%s_weighted.keras', None),
                    'cnn_multitask': ('cnn_multitask.keras', 0)}

    head_index = {'gender': 0, 'usage': 1}

    cost_rows = []

    #Fit times were written into the notes column where they were measured
    def fit_time_from_notes(notes):
        found = re.search(r'fit ([0-9.]+) s', str(notes))
        return float(found.group(1)) if found else np.nan

    for target in ['gender', 'usage']:
        print('\nTarget:', target)

        rows = results[results['target'] == target]

        #Best model that we can actually reload and ask for probabilities
        reloadable = rows[rows['model'].isin(saved_models)]
        best = reloadable.sort_values('macro_f1', ascending=False).iloc[0]
        print('Best submittable model:', best['model'], 'macro-F1 %.4f' % best['macro_f1'])

        filename, multi = saved_models[best['model']]
        if '%s' in filename:
            filename = filename % target
        model = keras.models.load_model(MODEL_DIR / filename)

        X_train, X_val, y_train, y_val, le = get_split(target, verbose=False)
        probabilities = model.predict(X_val)
        if multi is not None:
            probabilities = probabilities[head_index[target]]

        y_pred = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)

        #Calibration - is a 0.9 prediction right 90% of the time?
        edges = np.linspace(0, 1, 11)
        mean_confidence, bin_accuracy, bin_size = [], [], []

        for i in range(10):
            in_bin = (confidence >= edges[i]) & (confidence < edges[i + 1] if i < 9 else confidence <= 1.0)
            if in_bin.sum() == 0:
                continue
            mean_confidence.append(confidence[in_bin].mean())
            bin_accuracy.append((y_pred[in_bin] == y_val[in_bin]).mean())
            bin_size.append(int(in_bin.sum()))

        print('\nReliability by confidence decile:')
        for c, a, n in zip(mean_confidence, bin_accuracy, bin_size):
            print('  confidence %.2f  accuracy %.2f  n %d' % (c, a, n))

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], '--', color='grey', label='perfectly calibrated')
        ax.plot(mean_confidence, bin_accuracy, 'o-', label=best['model'])
        ax.set_xlabel('mean predicted confidence')
        ax.set_ylabel('accuracy in bin')
        ax.set_title('Calibration - ' + target)
        ax.legend()
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / ('calibration_%s.png' % target), dpi=150)
        plt.close(fig)
        print('saved', OUTPUT_DIR / ('calibration_%s.png' % target))

        #Error analysis - what the weakest classes are actually being confused with
        #Raw pixels this time, the standardised ones are unreadable as a picture
        table = per_class_report(y_val, y_pred, le)
        worst = [c for c in table.index[:3] if c in list(le.classes_)]
        print('\nWorst recall classes:', worst)

        _, X_val_raw, _, _, _ = get_split(target, normalised=False, verbose=False)

        picks = []
        for class_name in worst:
            class_id = list(le.classes_).index(class_name)
            wrong = np.where((y_val == class_id) & (y_pred != class_id))[0]
            picks.extend(wrong[:4])

        if picks:
            columns = 4
            rowsn = int(np.ceil(len(picks) / columns))
            fig, axes = plt.subplots(rowsn, columns, figsize=(columns * 2, rowsn * 2.6))
            for ax, i in zip(np.ravel(axes), picks):
                ax.imshow(X_val_raw[i])
                ax.set_title('%s -> %s' % (le.classes_[y_val[i]], le.classes_[y_pred[i]]), fontsize=8)
                ax.axis('off')
            for ax in np.ravel(axes)[len(picks):]:
                ax.axis('off')
            plt.tight_layout()
            fig.savefig(OUTPUT_DIR / ('errors_%s.png' % target), dpi=150)
            plt.close(fig)
            print('saved', OUTPUT_DIR / ('errors_%s.png' % target))

        #Cost table - what each score cost in weights and in training time
        for _, row in rows.iterrows():
            name = row['model']
            params = 0
            if name in saved_models:
                path, _unused = saved_models[name]
                if '%s' in path:
                    path = path % target
                params = keras.models.load_model(MODEL_DIR / path).count_params()
            cost_rows.append({'target': target,
                              'model': name,
                              'macro_f1': row['macro_f1'],
                              'params': params,
                              'fit_time_s': fit_time_from_notes(row['notes'])})

    cost = pd.DataFrame(cost_rows).sort_values(['target', 'macro_f1'], ascending=[True, False])
    cost.to_csv(OUTPUT_DIR / 'cost_table_task3.csv', index=False)

    print('\nCost table')
    print(cost.round(4).to_string(index=False))
    print('\nClassical fit times are printed in outputs/log_classical.txt, they are not')
    print('in results.csv so they are left as NaN here rather than guessed.')
    print('\nDone. Figures and cost_table_task3.csv are in outputs/.')


run_classical_baselines()
run_gender_models()
run_usage_models()
run_multitask()
run_analysis()
