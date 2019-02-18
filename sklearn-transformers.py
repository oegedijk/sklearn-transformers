import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pandas import DataFrameMapper


def gen_features(columns, classes=None, input_df=True,
                 suffix = None, alias = None):
    """
    Return a list of feature transformers.

    - alias to rename the column s
    - suffix to add an suffic to column (e.g. `_enc`)

    """
    if classes is None:
        return [(column, None) for column in columns]
    else:
        classes = [cls for cls in classes if cls is not None]

    # placeholder for all the
    feature_defs = []

    for column in columns:
        feature_transformers = []

        classes = [cls for cls in classes if cls is not None]
        if not classes:
            feature_defs.append((column, None))
        else:

            # collect all the transformer classes for this column:
            for definition in classes:
                if isinstance(definition, dict):
                    params = definition.copy()
                    klass = params.pop('class')
                    feature_transformers.append(klass(**params))
                else:
                    feature_transformers.append(definition())

            if not feature_transformers:
                # if no transformer classes found, then return as is (None)
                feature_transformers = None

            if input_df:
                if alias:
                    feature_defs.append((column,
                                         feature_transformers,
                                         {'input_df' : True, 'alias' : alias}))
                elif suffix:
                    feature_defs.append((column,
                                        feature_transformers,
                                        {'input_df' : True,
                                         'alias' : str(column)+str(suffix)}))
                else:
                    feature_defs.append((column,
                                         feature_transformers,
                                         {'input_df' : True}))

            else:
                if alias:
                    feature_defs.append((column,
                                        feature_transformers,
                                        {'alias' : alias}))
                elif suffix:
                    feature_defs.append((column,
                                         feature_transformers,
                                         {'alias' : str(column)+str(suffix)}))
                else:
                    feature_defs.append((column, feature_transformers))

    return feature_defs


class DummyTransform(TransformerMixin):
    def __init__(self, name="", verbose = 0):
        """
        Dummy transformer to make sure column is retained in dataset
        even though it does not need to be transformerd.
        """
        self.verbose = verbose
        self.name = name

    def fit(self, X, y=None):
        if self.verbose:
            print(f'Fit: DummyTransform for: {self.name}...')
        return self

    def transform(self, X, y=None):
        if self.verbose:
            print(f'Transform: DummyTransform for: {self.name}...')
        return X


class ReluTransform(TransformerMixin):
    def __init__(self, name="", verbose = 0):
        """
        Sets negative values to 0 like a Rectified Linear Unit (ReLU)

        (for things like revenue which never should be zero anyway)
        """
        self.verbose = verbose
        self.name = name

    def fit(self, X, y=None):
        if self.verbose:
            print(f'Fit: ReluTransform for: {self.name}...')
        return self

    def transform(self, X, y=None):
        if self.verbose:
            print(f'Transform: ReluTransform for: {self.name}...')
        return np.maximum(X, 0)


class NumericFill(TransformerMixin):
    def __init__(self, fill='ExtremeValue', name="", verbose = 0):
        """
        Fill missing numerical values with either -999 or the mean
        of the column.
        """
        self.verbose = verbose
        self.name = name
        self.fill=fill
        self._mean = None

    def fit(self, X, y=None):
        # coerce column to either numeric, or to nan:
        self._mean = pd.to_numeric(X, errors='coerce').mean()
        if self.verbose and self.fill=='mean':
            print(f'Fit: Filling numerical NaN {self.name} with \
                    {self.fill}: {self._mean}...')
        if self.verbose and self.fill=='ExtremeValue':
            print(f'Fit: Filling numerical NaN {self.name} with \
                    {self.fill}: -999...')
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if self.fill == 'mean':
            if self.verbose:
                print(f'Transform: Filling numerical NaN {self.name} \
                        with {self.fill} : {self._mean} ...')
            X = pd.to_numeric(X, errors='coerce').fillna(self._mean)

        elif self.fill =='ExtremeValue':
            if self.verbose:
                print(f'Transform: Filling numerical NaN \
                        {self.name} with {self.fill} : -999 ...')
            X = pd.to_numeric(X, errors='coerce').fillna(-999)
        return X


class StandardScale(TransformerMixin):
    def __init__(self,  name="", verbose = 0):
        """
        Scale numerical features to mean=0, sd=1
        """
        self.verbose = verbose
        self.name = name
        self._mean = None
        self._sd = None

    def fit(self, X, y=None):
        if self.verbose:
            print(f'Fit: StandarScaling {self.name}: \
                    ({self._mean}, {self._sd}...')

        self._mean = pd.to_numeric(X, errors='coerce').mean()
        self._sd = pd.to_numeric(X, errors='coerce').std()
        return self

    def transform(self, X, y=None):
        if self.verbose:
            print(f'Transform: StandarScaling {self.name}: \
                    ({self._mean}, {self._sd}...')

        X = X.copy()
        X = pd.to_numeric(X, errors='coerce').fillna(self._mean)
        X -= self._mean
        if self._sd > 0:
            X /= self._sd
        return X.astype(np.float32)


class LabelEncode(TransformerMixin):
    def __init__(self, name="", verbose = 0):
        """
        Safe LabelEncoder. Deals with missing values and values
        not seen in original column (labels these 'Missing' before label
        encoding).

        (sklearn's LabelEncoder honestly kind of sucks for this)
        """
        self.verbose = verbose
        self.name = name
        self.le = None
        self.mapping = None

    def fit(self, X, y=None):
        if self.verbose:
            print(f'Fit: Label Encoding {self.name} ...')
        # get all the labels of the categorical variables and add a dummy
        # label 'Missing' as a category for both NaN's and new labels
        # not seen in the training set.
        labels = X.append(pd.Series(['Missing'])).value_counts().index.tolist()
        # create the mapping from the feature names to labels 0, 1, 2, etc
        self.mapping = dict(zip(labels, range(len(labels)) ))
        return self

    def transform(self, X, y=None):
        if self.verbose:
            print(f'Transform: Label Encoding {self.name} ...')
        X = X.copy()
        # missing_idx is the default value for the dictionary lookup:
        missing_idx = self.mapping["Missing"]
        return X.fillna("Missing").astype(str).apply(
                    lambda x: self.mapping.get(x, missing_idx)).astype(int)


class ExtremeValueFill(TransformerMixin):
    def __init__(self, name="", verbose = 0):
        """
        Fills missing numerical values with -999

        """
        self.verbose = verbose
        self.name = name

    def fit(self, X, y=None):
        if self.verbose:
            print(f'Fit: Filling numerical NaN {self.name}...')
        return self

    def transform(self, X, y=None):
        if self.verbose:
            print(f'Transform: Filling numerical NaN {self.name}...')
        return X.fillna(-999)


class BinaryTargetFill(TransformerMixin):
    def __init__(self, name="", verbose = 0):
        """
        Cleans up dependent variable for classification:
        changes prediction > 1 to 1, and NaN to 0
        """
        self.verbose = verbose
        self.name=name

    def fit(self, X, y=None):
        if self.verbose:
            print(f'Fit: Filling target NaN {self.name}...')

        return self

    def transform(self, X, y=None):
        if self.verbose:
            print(f'Transform: Filling target NaN {self.name}...')

        X = X.copy()
        X.loc[X>1] = 1
        return X.fillna(0)


class MissingFill(TransformerMixin):
    def __init__(self, name = "", verbose = 0 ):
        """
        Fills missing categorical values with the label "Missing"
        """
        self.verbose = verbose
        self.name = name

    def fit(self, X, y=None):
        if self.verbose:
            print(f'Fit: Filling categorical NaN \
                    {self.name} with \"Missing\" ...')
        return self

    def transform(self, X, y=None):
        if self.verbose:
            print(f'Transform: Filling categorical NaN \
                    {self.name} with \"Missing\"...')
        return X.fillna("Missing")


class FrequencyFill(TransformerMixin):
    def __init__(self, name = "", verbose = 0):
        """
        Replaces categorical variables by the frequency of the labels
        in the trainig set.

        Fills missing values with -999
        """
        self.verbose = verbose
        self.name=name

    def fit(self, X, y=None):
        if self.verbose:
            print(f'Fit: Replacing variable by their frequency {self.name}...')

        self.frequency_dict = X.value_counts().to_dict()
        self.fitted = True
        return self

    def transform(self, X, y=None):
        if self.verbose:
            print(f'Transform: Replacing variable by \
                    their frequency {self.name}...')

        assert self.fitted == True

        X_freq = X.replace(self.frequency_dict)
        X_freq = X_freq.replace([np.inf, -np.inf], np.nan)
        X_freq = X_freq.fillna(-999)
        return X_freq


class OneHot(TransformerMixin):
    def __init__(self, topx = None, name = "", verbose = 0):
        """
        One hot encodes column. Adds a column _na, and codes any label not
        seen in the training data as _na. Also makes sure all columns in the
        training data will get created in the transformed dataframe.

        If topx is given only encodes the topx most frequent labels,
        and labels everything else _na.
        """
        self.verbose = verbose
        self.topx = topx
        self.name = name

    def fit(self, X, y=None):
        if self.verbose:
            print(f'Fit: One-hot coding categorical variable {self.name}...')
        X = X.copy()
        if self.topx is None:
            # store the particular categories to be encoded:
            self.categories = X.unique()
            # Then do a simple pd.get_dummies to get the columns
            self.columns = pd.get_dummies(pd.DataFrame(X),
                                          prefix = "",
                                          prefix_sep = "",
                                          dummy_na=True).columns
        else:
            # only take the topx most frequent categories
            self.categories = [x for x in  X.value_counts()\
                                             .sort_values(ascending=False)\
                                             .head(self.topx).index]
            # set all the other categories to np.nan
            X.loc[~X.isin(self.categories)] = np.nan
            self.columns = pd.get_dummies(pd.DataFrame(X),
                                             prefix = "",
                                             prefix_sep = "",
                                             dummy_na=True).columns
        return self

    def transform(self, X, y=None):
        if self.verbose:
            print(f'Transform: One-hot coding categorical \
                    variable {self.name}...')

        X = X.copy()
        # set all categories not present during fit() to np.nan:
        X.loc[~X.isin(self.categories)] = np.nan

        # onehot encode using pd.get_dummies
        X_onehot = pd.get_dummies(pd.DataFrame(X), prefix = "",
                                  prefix_sep = "", dummy_na=True)

        # add in columns missing in transform() that were present during fit()
        missing_columns = set(self.columns) - set(X_onehot.columns)
        for col in missing_columns:
            X_onehot[col]=0
        # make sure columns are in the same order
        X_onehot = X_onehot[self.columns]
        assert set(X_onehot.columns) == set(self.columns)
        # save the column names so that they can be assigned by DataFrameMapper
        self._feature_names = X_onehot.columns
        return X_onehot

    def get_feature_names(self):
        # helper function for sklearn-pandas to assign right column names
        return self._feature_names


class LookupEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, lookup_table, fill='mean', smoothing = 0,
                 name = "", verbose=0):
        """
        Replaces every label in a categorical variable with the value given
        in the lookup_table.

        When this used to apply a mean encoding (a.k.a. target encoding),
        you can add smoothing to bias the labels with only few observations
        towards the mean.

        NaN can either be filled with 'mean' or 'ExtremeValue'
        """
        self.verbose = verbose
        self._dim = None
        self.lookup_table = lookup_table
        self.mapping = None
        self.fill = fill
        self._mean = None
        self.smoothing = smoothing
        self.name = name

    def fit(self, X, y=None, **kwargs):
        if self.verbose:
            print(f'Fit: Lookup table encoding {self.name}...')

        if len(self.lookup_table.columns)== 3:
            self.lookup_table.columns = ['label', 'encoding', 'count']
        elif len(self.lookup_table.columns)== 2:
            self.lookup_table.columns = ['label', 'encoding']

        if self.fill=='mean':
            assert len(self.lookup_table.columns)== 3
            self._mean = (
                        np.sum(self.lookup_table['encoding'] *
                               self.lookup_table['count']) /
                               np.sum(self.lookup_table['count'])
            )

        if self.smoothing>0:
            assert len(self.lookup_table.columns)== 3
            # smoothing is used for mean encoding and biases the variables with
            # small counts towards the average
            # mean encoding. This is to prevent overfitting for categories with
            # small sample size.
            self.lookup_table['smooth_encoding'] = (
                        (self.lookup_table['encoding'] * self.lookup_table['count']
                        + self.mean*self.smoothing)
                        / (self.lookup_table['count'] + self.smoothing)
            )

            self.mapping = pd.Series(
                self.lookup_table['smooth_encoding'].values.tolist(),
                index=self.lookup_table['label'].values.tolist()
            )
        else:
            self.mapping = pd.Series(
                self.lookup_table['encoding'].values.tolist(),
                index=self.lookup_table['label'].values.tolist()
            )

        self.fitted = True
        return self

    def transform(self, X, y=None):
        if self.verbose:
            print(f'Transform: Lookup table encoding {self.name}...')

        assert self.fitted == True

        mapped_column =  X.map(self.mapping)
        if self.fill == 'mean':
            mapped_column.fillna(self._mean, inplace=True)
        elif self.fill =='ExtremeValue':
            mapped_column.fillna(-999, inplace=True)
        return mapped_column


class DoubleLookupEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, lookup_table, fill='ExtremeValue', name="", verbose=0):
        """
        Replaces every label combination in two columns with the value given
        in the lookup_table.

        NaN can either be filled with 'mean' or 'ExtremeValue'.

        The second
        """
        self.lookup_table = lookup_tabl
        self.fill = fill
        self.name=name
        self.verbose = verbose
        self._mean = None
        self.mapping = None

    def fit(self, X, y, **kwargs):
        if self.verbose:
            print(f'Fit: Double Lookup table encoding {self.name}...')

        self.lookup_table.columns = ['label1', 'label2', 'encoding', 'count']
        if self.fill=='mean':
            self._mean = np.sum(
                        (self.lookup_table['count'] * self.lookup_table['encoding'])
                        / np.sum(self.lookup_table['count'])
            )
        self.fitted = True
        return self

    def transform(self, X, y =  None):
        if self.verbose:
            print(f'Transform: Double Lookup table encoding {self.name}...')

        assert self.fitted == True
        assert isinstance(X, pd.DataFrame)

        X = X.copy()
        X.columns = ['label1', 'label2']
        # make sure version is always int:

        mapped_column =  X.merge(self.lookup_table, how='left',
                                      on=['label1', 'label2'])['encoding']

        if self.fill == 'mean':
            mapped_column.fillna(self._mean, inplace=True)
        elif self.fill =='ExtremeValue':
            mapped_column.fillna(-999, inplace=True)
        return mapped_column


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=0, fill='ExtremeValue', name="", verbose=0,):
        """
        Replaces every label in a categorical column with the average value of
        the target (i.e. value that you are trying to predict).

        Add smoothing to bias values to the mean target value for labels
        with few observations.
        """
        self.smoothing = smoothing
        self.fill = fill
        self.name=name
        self.verbose = verbose
        self._mean = None
        self.mapping = None

    def fit(self, X, y, **kwargs):
        if self.verbose: print(f'Fit: Target Mean encoding {self.name}...')

        assert X.shape[0] == y.shape[0]
        assert isinstance(X, pd.Series)
        assert isinstance(y, pd.Series)

        combined = pd.concat([X, y], axis=1, ignore_index=True)
        combined.columns = ['label', 'target']

        self._mean = y.mean()

        self.mapping = combined.groupby('label').target.\
                apply(lambda x: ((x.mean() * x.count()) +
                                 self._mean*self.smoothing) /
                                    (x.count()+self.smoothing))
        self.fitted = True
        return self

    def transform(self, X, y=None):
        if self.verbose:
            print(f'Transform: Target Mean encoding {self.name}...')
        assert self.fitted == True
        assert (y is None or X.shape[0] == y.shape[0])
        assert isinstance(X, pd.Series)

        combined = pd.DataFrame(X)
        combined.columns = ['label']
        combined['label_target_enc'] = combined['feat'].map(self.mapping)

        if self.fill == 'mean':
            combined['feat_target_enc'].fillna(self._mean, inplace=True)
        elif self.fill =='ExtremeValue':
            combined['feat_target_enc'].fillna(-999, inplace=True)
        return combined['feat_target_enc']


def fit_transformers(X, target=None, target_enc_cols = [], onehot_cols = [],
                            onehot_top10_cols = [], lookup_cols = {},
                            double_lookup_cols = {},
                            fill="ExtremeValue",
                            verbose=0):
    """

    Returns fitted transformer object.

    Fills all numerical columns except target.

    One hot encodes all categorical columns, except the categorical columns
    that are target encoded, onehot-top10 encoded, lookup encoded or double
    lookup encoded.

    """
    num_columns = list(set(X.select_dtypes(include=np.number).columns)
                            - set([target]))

    obj_columns = X.select_dtypes(include=['object']).columns

    # all the obj columns not otherwise specified get onehot-encoded:
    onehot_columns = list(set(list(set(obj_columns)
                            - set(target_enc_cols)
                            - set(onehot_top10_cols)
                            - set(lookup_cols.keys())
                            - set([key[0] for key in double_lookup_cols.keys()]
                                + [key[1] for key in double_lookup_cols.keys()])
                        ) + onehot_cols))

    mappers = []

    # dummy transform to make sure target stays in the transformed dataframe:
    mappers = mappers + gen_features(
        columns=[target],
        classes = [{'class' : DummyTransform,
                    'name': target,
                    'verbose':1}],
        input_df = True
        )

    # Fill missing values in numerical columns:
    for num_col in num_columns:
        mappers = mappers + gen_features(
            columns=[num_col],
            classes = [{'class' : NumericFill,
                        'fill': fill,
                        'name': num_col,
                        'verbose':1}],
            input_df = True
        )

    for target_col in target_enc_cols:
        mappers = mappers + gen_features(
            columns=[target_col],
            classes = [{'class' : TargetEncoder,
                        'fill' : fill,
                        'name': target_col,
                        'smoothing': 10,
                        'verbose':1}],
            input_df = True,
            suffix = "_enc")

    for onehot_col in onehot_top10_cols:
        mappers = mappers + gen_features(
            columns=[onehot_col],
            classes=[{'class' : OneHot,
                      'topx': 10,
                      'name': onehot_col,
                      'verbose':1}],
            input_df = True
        )

    for onehot_col in onehot_cols:
        mappers = mappers + gen_features(
            columns=[onehot_col],
            classes=[{'class' : OneHot,
                      'name': onehot_col,
                      'verbose':1}],
            input_df = True
        )

    for col, lookup_table in lookup_cols.items():
        print(f'lookup table: {col}')
        mappers = mappers + gen_features(
            columns=[col],
            classes=[{'class' : LookupEncoder,
                      'lookup_table' : lookup_table,
                      'fill' : fill,
                      'name' : col,
                      'verbose':1}],
            input_df = True,
            suffix = "_enc"
        )

    for cols, lookup_table in double_lookup_cols.items():
        mappers = mappers + gen_features(
            columns=[[cols[0], cols[1]]],
            classes=[{'class' : DoubleLookupEncoder,
                      'lookup_table' : lookup_table,
                      'fill' : fill,
                      'name':cols[2],
                      'verbose':1}],
            input_df = True,
            alias = cols[2]
        )

    if verbose:
        print("Columns being transformed: ")
        print("numeric columns: ", num_columns)
        print("categorical columns: ", obj_columns)
        print("target_column: ", target)
        print("target encoded columns: ", target_enc_cols)
        print("top10 onehotencoded columns: ", onehot_top10_cols)
        print("onehotencoded columns: ", onehot_columns)
        print("lookup columns: ", lookup_cols.keys())
        print("double lookup columns: ", double_lookup_cols.keys())

    mapper = DataFrameMapper(mappers, df_out=True)


    if verbose: print("fitting transformer...")
    X = X.copy()
    mapper.fit(X, X[target])
    return mapper


def fit_embedding_transformers(X, target=None,
                               lookup_cols = {}, double_lookup_cols = {},
                               fill="mean", verbose=0):
    """
    Return a transformer object that fills numerical features with its mean
    and then StandardScales them.

    Categorical features get label encoded to prepare them to be entered into
    an Embedding layer.

    Returns the transformer object, a list of numeric features and a list of
    categorical features.
    """
    num_columns = X.drop([target], axis=1)\
                   .select_dtypes(include=np.number).columns.tolist()

    obj_columns = X.select_dtypes(include=['object']).columns.tolist()

    mappers = []

    mappers = mappers + gen_features(
        columns=[target],
        classes = [{'class' : DummyTransform,
                    'name': target,
                    'verbose':1}],
        input_df = True
        )

    for num_col in num_columns:
        mappers = mappers + gen_features(
            columns=[num_col],
            classes = [{'class' : NumericFill,
                        'fill': fill,
                        'name': num_col,
                        'verbose':1},
                       {'class' : StandardScale,
                        'name': num_col,
                        'verbose':1}],
            input_df = True
        )

    for obj_col in obj_columns:
        mappers = mappers + gen_features(
            columns=[obj_col],
            classes = [{'class' : LabelEncode,
                        'name': obj_col,
                        'verbose':1}],
            input_df = True
        )


    for col, lookup_table in lookup_cols.items():
        print(f'lookup table: {col}')
        mappers = mappers + gen_features(
            columns=[col],
            classes=[{'class' : LookupEncoder,
                      'lookup_table' : lookup_table,
                      'fill' : fill,
                      'name' : col,
                      'verbose':1}],
            input_df = True,
            suffix = "_enc"
        )

    for cols, lookup_table in double_lookup_cols.items():
        mappers = mappers + gen_features(
            columns=[[cols[0], cols[1]]],
            classes=[{'class' : DoubleLookupEncoder,
                      'lookup_table' : lookup_table,
                      'fill' : fill,
                      'name':cols[2],
                      'verbose':1}],
            input_df = True,
            alias = cols[2]
        )

    if verbose:
        print("Columns being transformed: ")
        print("numeric columns: ", num_columns)
        print("categorical columns: ", obj_columns)
        print("target_column: ", target)
        print("lookup columns: ", lookup_cols.keys())
        print("double lookup columns: ", double_lookup_cols.keys())

    transformers =  DataFrameMapper(mappers, df_out=True)

    if verbose: print("fitting transformer...")
    X = X.copy()
    transformers.fit(X, X[target])
    return transformers, num_columns, obj_columns


def apply_transformers(X, transformers, verbose=0):
    if transformers is not None:
        return transformers.transform(X)
    else:
        return None
