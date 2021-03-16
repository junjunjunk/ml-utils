import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.utils.validation import FLOAT_DTYPES, check_array, check_is_fitted


class BM25Transformer(TransformerMixin, BaseEstimator):
    """Transform a count matrix to a normalized tf or Okapi BM25 representation

    Parameters
    ----------
    norm : {'l1', 'l2'}, default='l2'
        Each output row will have unit norm, either:
        * 'l2': Sum of squares of vector elements is 1. The cosine
        similarity between two vectors is their dot product when l2 norm has
        been applied.
        * 'l1': Sum of absolute values of vector elements is 1.
        See :func:`preprocessing.normalize`

    Attributes
    ----------
    idf : array of shape (n_features)
        The inverse document frequency (IDF) vector;

    Examples
    --------
    >>> from sklearn.feature_extraction.text import TfidfTransformer
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> from sklearn.pipeline import Pipeline
    >>> import numpy as np
    >>> corpus = ['this is the first document',
    ...           'this document is the second document',
    ...           'and this is the third one',
    ...           'is this the first document']
    >>> vocabulary = ['this', 'document', 'first', 'is', 'second', 'the',
    ...               'and', 'one']
    >>> pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
    ...                  ('tfid', TfidfTransformer())]).fit(corpus)
    >>> pipe['count'].transform(corpus).toarray()
    array([[1, 1, 1, 1, 0, 1, 0, 0],
           [1, 2, 0, 1, 1, 1, 0, 0],
           [1, 0, 0, 1, 0, 1, 1, 1],
           [1, 1, 1, 1, 0, 1, 0, 0]])
    >>> pipe['tfid'].idf
    array([1.        , 1.22314355, 1.51082562, 1.        , 1.91629073,
           1.        , 1.91629073, 1.91629073])
    >>> pipe.transform(corpus).shape
    (4, 8)
    """

    def __init__(self, norm="l2", k1=1.2, b=0.75):
        self.norm = norm
        self.k1 = k1
        self.b = b
        self.avdl = None

    def fit(self, X, y=None):
        """Learn(Calculate) the idf vector (global term weights).
        Parameters
        ----------
        X : sparse matrix of shape n_samples, n_features)
            A matrix of term/token counts. (CountVector)
        """
        X = check_array(X, accept_sparse=("csr", "csc"))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64

        # Calc idf
        n_samples, n_features = X.shape

        # document_length
        dl = X.sum(axis=1)
        # average of document length
        self.avgdl = np.mean(dl)

        document_frequency = self._document_frequency(X)
        # NOTE: copy=False is necessary for scipy >= 1.1
        document_frequency = document_frequency.astype(dtype, copy=False)
        idf = np.log(
            (n_samples - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        self._idf_diag = sp.diags(
            idf, offsets=0, shape=(n_features, n_features), format="csr", dtype=dtype
        )

        return self

    def transform(self, X, copy=True):
        """Transform a count matrix to a tf or tf-idf representation
        Parameters
        ----------
        X : sparse matrix of (n_samples, n_features)
            a matrix of term/token counts (CountVector)
        copy : bool, default=True
            Whether to copy X and operate on the copy or perform in-place
            operations.
        Returns
        -------
        vectors : sparse matrix of shape (n_samples, n_features)
        """
        X = check_array(X, accept_sparse="csr", dtype=FLOAT_DTYPES, copy=copy)
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float64)

        n_samples, n_features = X.shape

        check_is_fitted(self, attributes=["idf"], msg="idf vector is not fitted")

        expected_n_features = self._idf_diag.shape[0]
        if n_features != expected_n_features:
            raise ValueError(
                "Input has n_features=%d while the model"
                " has been trained with n_features=%d"
                % (n_features, expected_n_features)
            )

        # document length: Shape=(n_sample)
        dl = np.sum(X, axis=1)
        sz = X.indptr[1:] - X.indptr[0:-1]
        rep = np.repeat(np.asarray(dl), sz)

        denominator = X.data + self.k1 * (1 - self.b + self.b * (rep / self.avgdl))

        X[X.nonzero()] = (X[X.nonzero()] * (self.k1 + 1)) / (denominator)
        X = X * self._idf_diag
        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

    @classmethod
    def _document_frequency(self, X):
        """Count the number of non-zero values for each feature in sparse X."""
        if sp.isspmatrix_csr(X):
            return np.bincount(X.indices, minlength=X.shape[1])
        else:
            return np.diff(X.indptr)

    @property
    def idf(self):
        # if _idf_diag is not set, this will raise an attribute error,
        # which means hasattr(self, "idf") is False
        return np.ravel(self._idf_diag.sum(axis=0))


class BM25Vectorizer(CountVectorizer):
    r"""Convert a collection of raw documents to a matrix of BM25 features.

    See Also
    --------
    CountVectorizer : Transforms text into a sparse matrix of n-gram counts.

    Notes
    -----
    The ``stop_words_`` attribute can get large and increase the model size
    when pickling. This attribute is provided only for introspection and can
    be safely removed using delattr or set to None before pickling.
    Examples
    --------
    >>> from sklearn.feature_extraction.text import TfidfVectorizer
    >>> corpus = [
    ...     'This is the first document.',
    ...     'This document is the second document.',
    ...     'And this is the third one.',
    ...     'Is this the first document?',
    ... ]
    >>> vectorizer = TfidfVectorizer()
    >>> X = vectorizer.fit_transform(corpus)
    >>> print(vectorizer.get_feature_names())
    ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
    >>> print(X.shape)
    (4, 9)
    """

    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        analyzer="word",
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.float64,
        norm="l2"
    ):

        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )

        self._bm25 = BM25Transformer(norm=norm)

    # Broadcast the TF-IDF parameters to the underlying transformer instance
    # for easy grid search and repr

    @property
    def norm(self):
        return self._bm25.norm

    @norm.setter
    def norm(self, value):
        self._bm25.norm = value

    @property
    def idf(self):
        return self._bm25.idf

    @idf.setter
    def idf(self, value):
        self._validate_vocabulary()
        if hasattr(self, "vocabulary_"):
            if len(self.vocabulary_) != len(value):
                raise ValueError(
                    "idf length = %d must be equal "
                    "to vocabulary size = %d" % (len(value), len(self.vocabulary))
                )
        self._bm25.idf = value

    def _check_params(self):
        if self.dtype not in FLOAT_DTYPES:
            warnings.warn(
                "Only {} 'dtype' should be used. {} 'dtype' will "
                "be converted to np.float64.".format(FLOAT_DTYPES, self.dtype),
                UserWarning,
            )

    def fit(self, raw_documents, y=None):
        """Learn vocabulary and idf from training set.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        y : None
            This parameter is not needed to compute tfidf.
        Returns
        -------
        self : object
            Fitted vectorizer.
        """
        self._check_params()
        self._warn_for_unused_params()
        X = super().fit_transform(raw_documents)
        self._bm25.fit(X)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn vocabulary and idf, return document-term matrix.
        This is equivalent to fit followed by transform, but more efficiently
        implemented.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        y : None
            This parameter is ignored.
        Returns
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        self._check_params()
        # fit_transform by CountVectorizer
        X = super().fit_transform(raw_documents)

        self._bm25.fit(X)
        # X is already a transformed view of raw_documents so we set copy to False
        return self._bm25.transform(X, copy=False)

    def transform(self, raw_documents):
        """Transform documents to document-term matrix.
        Uses the vocabulary and document frequencies (df) learned by fit (or
        fit_transform).
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        Returns
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
        """
        check_is_fitted(self, msg="The TF-IDF vectorizer is not fitted")

        # CountVector
        X = super().transform(raw_documents)

        return self._bm25.transform(X, copy=False)
