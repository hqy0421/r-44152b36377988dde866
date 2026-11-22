"""Contains transformers that deal with imputation of missing values."""

from __future__ import annotations

import warnings
from typing import Any

import narwhals as nw
import polars as pl
from beartype import beartype
from typing_extensions import deprecated

from tubular._stats import (
    _get_mean_calculation_expressions,
    _get_median_calculation_expression,
)
from tubular._utils import (
    _collect_frame,
    _convert_dataframe_to_narwhals,
    _convert_series_to_narwhals,
    _is_null,
    _return_narwhals_or_native_dataframe,
    _sort_dict,
    block_from_json,
)
from tubular.base import BaseTransformer, register
from tubular.functions.imputers import (
    get_null_indicator_column_name,
    impute_boolean_columns,
    impute_categorical_columns,
    impute_enum_columns,
    impute_numeric_or_string_nulls,
    indicate_nulls_for_columns,
)
from tubular.mixins import WeightColumnMixin
from tubular.types import DataFrame, LazyFrame, ListOfStrs, NumericTypes, Series

pl.enable_string_cache()


@register
class BaseImputer(BaseTransformer):
    """Contains transform method that will use fill nulls with values from self.impute_values_.

    Other imputers in this module should inherit from this class.

    Attributes
    ----------
    add_indicator : bool, default = False
        Should indicator columns be created for the nulls filled in? If True, a
        boolean column named "<column>_nulls" is added for each column in
        columns, capturing whether the value was null *before* imputation was
        applied. Both the indicator and the imputed value are produced in a
        single transform call, so the indicator always reflects the incoming
        row and cannot be separated from imputation by reordering a pipeline.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Examples
    --------
    ```pycon
    >>> BaseImputer(columns=["a", "b"])
    BaseImputer(columns=['a', 'b'])

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    # this class is not by itself jsonable, as needs attrs
    # which are set in the child classes
    jsonable = False

    FITS = False

    @beartype
    def __init__(
        self,
        columns: ListOfStrs | str,
        add_indicator: bool = False,
        **kwargs: bool | None,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : str or list of strs
            Columns to impute.

        add_indicator : bool, default = False
            Should indicator columns be created for the nulls filled in? If True,
            a boolean column named "<column>_nulls" is added for each column in
            columns, recording whether the incoming value was null before imputation.
            The indicator is computed from the row being transformed, never from
            values learnt at fit, so prediction rows are flagged on their own nulls
            only.

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        Examples
        --------
        ```pycon
        >>> BaseImputer(columns=["a", "b"], add_indicator=True)
        BaseImputer(add_indicator=True, columns=['a', 'b'])

        ```

        """
        super().__init__(columns=columns, **kwargs)

        self.add_indicator = add_indicator

    @property
    def null_indicator_columns_(self) -> dict[str, str]:
        """Mapping of imputed columns to the indicator columns created for them.

        Returns
        -------
        dict[str, str]:
            mapping of column name to indicator column name, one entry per column
            in columns. Useful for interpreting output column by column.

        Examples
        --------
        ```pycon
        >>> imputer = BaseImputer(columns=["a", "b"], add_indicator=True)

        >>> imputer.null_indicator_columns_
        {'a': 'a_nulls', 'b': 'b_nulls'}

        ```

        """
        return {
            column: get_null_indicator_column_name(column) for column in self.columns
        }

    def get_feature_names_out(self) -> list[str]:
        """List features modified/created by the transformer.

        Returns
        -------
        list[str]:
            list of features modified/created by the transformer, including the
            indicator columns if add_indicator is True

        Examples
        --------
        ```pycon
        >>> imputer = BaseImputer(columns=["a", "b"], add_indicator=True)

        >>> imputer.get_feature_names_out()
        ['a', 'b', 'a_nulls', 'b_nulls']

        >>> imputer = BaseImputer(columns=["a", "b"])

        >>> imputer.get_feature_names_out()
        ['a', 'b']

        ```

        """
        if not self.add_indicator:
            return [*self.columns]

        return [*self.columns, *self.null_indicator_columns_.values()]

    def get_indicator_exprs(self) -> list[nw.Expr]:
        """Get expressions creating null indicator columns.

        Returns
        -------
        list[nw.Expr]: indicator expressions, or an empty list if
        add_indicator is False

        Examples
        --------
        ```pycon
        >>> imputer = BaseImputer(columns=["a", "b"], add_indicator=True)

        >>> imputer.null_indicator_columns_
        {'a': 'a_nulls', 'b': 'b_nulls'}

        >>> len(imputer.get_indicator_exprs())
        2

        >>> imputer = BaseImputer(columns=["a", "b"])

        >>> imputer.get_indicator_exprs()
        []

        ```

        """
        if not self.add_indicator:
            return []

        return indicate_nulls_for_columns(columns=self.columns)

    def _apply_transform_exprs(
        self,
        X: nw.DataFrame | nw.LazyFrame,
        transform_exprs: list[nw.Expr],
    ) -> nw.DataFrame | nw.LazyFrame:
        """Apply indicator and imputation expressions to X.

        Indicator expressions are applied in a separate step, before imputation,
        so that indicators always describe the values in the incoming row,
        independently of where the imputer sits in a pipeline.

        Parameters
        ----------
        X: narwhals dataframe
            Data to transform.

        transform_exprs: list[nw.Expr]
            Imputation expressions to apply.

        Returns
        -------
        narwhals dataframe: transformed data

        """
        indicator_exprs = self.get_indicator_exprs()

        if indicator_exprs:
            X = X.with_columns(*indicator_exprs)

        if transform_exprs:
            X = X.with_columns(*transform_exprs)

        return X

    @block_from_json
    def to_json(self) -> dict[str, dict[str, Any]]:
        """Dump transformer to json dict.

        Returns
        -------
        dict[str, dict[str, Any]]:
            jsonified transformer. Nested dict containing levels for attributes
            set at init and fit.

        Raises
        ------
        RuntimeError:
            if class is not jsonable

        Examples
        --------
        ```pycon
        >>> arbitrary_imputer = ArbitraryImputer(columns=["a", "b"], impute_value=1)

        >>> # version will vary for local vs CI, so use ... as generic match
        >>> arbitrary_imputer.to_json()
        {'tubular_version': ..., 'classname': 'ArbitraryImputer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'impute_value': 1, 'add_indicator': False}, 'fit': {'is_fitted_': True, 'impute_values_': {'a': 1, 'b': 1}}}

        >>> mean_imputer = MeanImputer(columns=["a", "b"])

        >>> test_df = pl.DataFrame({"a": [1, None], "b": [None, 2]})

        >>> _ = mean_imputer.fit(test_df)

        >>> mean_imputer.to_json()
        {'tubular_version': ..., 'classname': 'MeanImputer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'weights_column': None, 'add_indicator': False}, 'fit': {'is_fitted_': True, 'impute_values_': {'a': 1.0, 'b': 2.0}}}

        ```

        """
        if not self.jsonable:
            msg = (
                "This transformer has not yet had to/from json functionality developed"
            )
            raise RuntimeError(
                msg,
            )

        self.check_is_fitted("impute_values_")

        json_dict = super().to_json()

        # slightly awkward here as API not fully shared
        # across classes
        if isinstance(
            self,
            (
                MeanImputer,
                MedianImputer,
                ModeImputer,
            ),
        ):
            json_dict["init"]["weights_column"] = self.weights_column
        elif isinstance(
            self,
            (
                ArbitraryImputer,
                NumberImputer,
                StringImputer,
                CategoricalImputer,
                BooleanImputer,
            ),
        ):
            json_dict["init"]["impute_value"] = self.impute_value

        json_dict["init"]["add_indicator"] = self.add_indicator

        json_dict["fit"]["impute_values_"] = _sort_dict(self.impute_values_)

        return json_dict

    def _check_for_failed_fit(self) -> None:
        """Check if fit failed to find needed attrs (if impute_values_ are None).

        Raises
        ------
        ValueError: if impute_values_ have come out as None

        """
        failed_columns = [
            col for col in self.columns if _is_null(self.impute_values_[col])
        ]

        if failed_columns:
            msg = f"fit has failed for columns {failed_columns}, it is possible that all rows are invalid - check for null/negative weights, all null columns, or other invalid conditions listed in the docstring"
            raise ValueError(msg)

    def get_transform_exprs(self) -> list[nw.Expr]:
        """Get transform expressions.

        Returns
        -------
        list[nw.Expr]: transform expressions for class

        """
        return impute_numeric_or_string_nulls(
            columns=self.columns, impute_values=self.impute_values_
        )

    @beartype
    def transform(
        self,
        X: DataFrame,
        return_native_override: bool | None = None,
    ) -> DataFrame:
        """Impute missing values with values calculated from fit method.

        Parameters
        ----------
        X : DataFrame
            Data to impute.

        return_native_override: Optional[bool]
            option to override return_native attr in transformer, useful when calling parent
            methods

        Returns
        -------
        X : DataFrame
            Transformed input X with nulls imputed with the median value for the specified columns.

        Examples
        --------
        ```pycon
        >>> import polars as pl

        >>> imputer = BaseImputer(columns=["a", "b"])

        >>> imputer.impute_values_ = {"a": 2, "b": 3.5}

        >>> test_df = pl.DataFrame({"a": [1, None, 2], "b": [3, None, 4]})

        >>> imputer.transform(test_df)
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ f64 │
        ╞═════╪═════╡
        │ 1   ┆ 3.0 │
        │ 2   ┆ 3.5 │
        │ 2   ┆ 4.0 │
        └─────┴─────┘

        ```

        """
        self.check_is_fitted("impute_values_")

        return_native = self._process_return_native(return_native_override)

        X = _convert_dataframe_to_narwhals(X)

        X = super().transform(X, return_native_override=False)

        transform_expressions = self.get_transform_exprs()

        X = self._apply_transform_exprs(X, transform_expressions)

        return _return_narwhals_or_native_dataframe(X, return_native)


@register
class NumberImputer(BaseImputer):
    """Class to handle arbitrary number imputation.

    Attributes
    ----------
    impute_value : int | float
        Value to impute nulls with.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    """

    polars_compatible = True

    lazyframe_compatible = True

    jsonable = True

    FITS = False

    @beartype
    def __init__(
        self,
        impute_value: int | float,
        columns: str | list[str],
        add_indicator: bool = False,
        **kwargs: bool | None,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        impute_value : int or float
            Value to impute nulls with.
        columns : str or list of strs
            Columns to impute
        add_indicator : bool, default = False
            Should indicator columns be created for the nulls filled in? If True,
            a boolean column named "<column>_nulls" is added for each column in
            columns, recording whether the incoming value was null before imputation.
        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns=columns, add_indicator=add_indicator, **kwargs)

        self.impute_values_ = {}
        self.impute_value = impute_value

        for c in self.columns:
            self.impute_values_[c] = self.impute_value
        self.is_fitted_ = True  # Does not fit

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Impute missing values with the supplied impute_value.

        Parameters
        ----------
        X : DataFrame
            Data containing columns to impute.

        Returns
        -------
        X : DataFrame
            Transformed input X with nulls imputed with the specified impute_value, for the specified columns.

        Raises
        ------
        TypeError: if provided data is incompatible with provided impute value

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> test_df = pl.DataFrame({"a": [1, None, 2], "b": [3, None, 4]})
        >>> imputer = NumberImputer(columns=["a", "b"], impute_value=5)
        >>> imputer.transform(test_df)
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 3   │
        │ 5   ┆ 5   │
        │ 2   ┆ 4   │
        └─────┴─────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)

        schema = X.collect_schema()

        bad_types = [
            schema[col]
            for col in self.columns
            if not isinstance(schema[col], (*NumericTypes, nw.Unknown))
        ]

        if bad_types:
            msg = f"""
                {self.classname()}: transformer can only handle Float/Int/UInt/Unknown type columns
                but got columns with types {bad_types}
                """
            raise TypeError(
                msg,
            )

        X = BaseTransformer.transform(self, X, return_native_override=False)

        transform_exprs = self.get_transform_exprs()

        X = self._apply_transform_exprs(X, transform_exprs)

        return _return_narwhals_or_native_dataframe(X, self.return_native)


@register
class StringImputer(BaseImputer):
    """Class to handle arbitrary string imputation.

    Attributes
    ----------
    impute_value : string
        Value to impute nulls with.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    """

    polars_compatible = True

    lazyframe_compatible = True

    jsonable = True

    FITS = False

    @beartype
    def __init__(
        self,
        impute_value: str,
        columns: str | list[str],
        add_indicator: bool = False,
        **kwargs: bool | None,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        impute_value : str
            Value to impute nulls with.
        columns : str or list of strs
            Columns to impute
        add_indicator : bool, default = False
            Should indicator columns be created for the nulls filled in? If True,
            a boolean column named "<column>_nulls" is added for each column in
            columns, recording whether the incoming value was null before imputation.
        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns=columns, add_indicator=add_indicator, **kwargs)

        self.impute_values_ = {}
        self.impute_value = impute_value

        for c in self.columns:
            self.impute_values_[c] = self.impute_value

        self.is_fitted_ = True  # Does not fit

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Impute missing values with the supplied impute_value.

        Parameters
        ----------
        X : DataFrame
            Data containing columns to impute.

        Returns
        -------
        X : DataFrame
            Transformed input X with nulls imputed with the specified impute_value, for the specified columns.

        Raises
        ------
        TypeError: if provided data is incompatible with provided impute value

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> test_df = pl.DataFrame({"a": ["cat", None, "dog"]})
        >>> imputer = StringImputer(columns=["a"], impute_value="missing")
        >>> imputer.transform(test_df)
        shape: (3, 1)
        ┌─────────┐
        │ a       │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ cat     │
        │ missing │
        │ dog     │
        └─────────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)

        X = BaseTransformer.transform(self, X, return_native_override=False)

        schema = X.collect_schema()

        bad_types = [
            schema[col]
            for col in self.columns
            if not isinstance(schema[col], (nw.String, nw.Unknown))
        ]

        if bad_types:
            msg = f"""
                {self.classname()}: transformer can only handle String/Unknown type columns
                but got columns with types {bad_types}
                """
            raise TypeError(
                msg,
            )

        transform_exprs = self.get_transform_exprs()

        X = self._apply_transform_exprs(X, transform_exprs)

        return _return_narwhals_or_native_dataframe(X, self.return_native)


@register
class BooleanImputer(BaseImputer):
    """Class to handle arbitrary boolean imputation.

    Attributes
    ----------
    impute_value : bool
        Value to impute nulls with.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    """

    polars_compatible = True

    lazyframe_compatible = True

    jsonable = True

    FITS = False

    @beartype
    def __init__(
        self,
        impute_value: bool,
        columns: str | list[str],
        add_indicator: bool = False,
        **kwargs: bool | None,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        impute_value : bool
            Value to impute nulls with.
        columns : str or list of strs
            Columns to impute
        add_indicator : bool, default = False
            Should indicator columns be created for the nulls filled in? If True,
            a boolean column named "<column>_nulls" is added for each column in
            columns, recording whether the incoming value was null before imputation.
        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns=columns, add_indicator=add_indicator, **kwargs)

        self.impute_values_ = {}
        self.impute_value = impute_value

        for c in self.columns:
            self.impute_values_[c] = self.impute_value

        self.is_fitted_ = True  # Does not fit

    def get_transform_exprs(self) -> list[nw.Expr]:
        """Get transform expressions.

        Returns
        -------
        list[nw.Expr]: transform expressions for class

        """
        return impute_boolean_columns(
            columns=self.columns, impute_values=self.impute_values_
        )

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Impute missing values with the supplied impute_value.

        Parameters
        ----------
        X : DataFrame
            Data containing columns to impute.

        Returns
        -------
        X : DataFrame
            Transformed input X with nulls imputed with the specified impute_value, for the specified columns.

        Raises
        ------
        TypeError: if provided data is incompatible with provided impute value

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> test_df = pl.DataFrame({"a": [True, None, False]})
        >>> imputer = BooleanImputer(columns=["a"], impute_value=True)
        >>> imputer.transform(test_df)
        shape: (3, 1)
        ┌───────┐
        │ a     │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ true  │
        │ true  │
        │ false │
        └───────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)

        schema = X.collect_schema()

        allowed_types = [nw.Boolean, nw.Unknown]
        allowed_types_str = "Boolean/Unknown"

        if nw.get_native_namespace(X).__name__ == "pandas":
            allowed_types += [nw.Object]
            allowed_types_str += "/Object"

        bad_types = [
            schema[col] for col in self.columns if schema[col] not in allowed_types
        ]

        if bad_types:
            msg = f"""
                {self.classname()}: transformer can only handle {allowed_types_str} type columns
                but got columns with types {bad_types}
                """
            raise TypeError(
                msg,
            )

        X = BaseTransformer.transform(self, X, return_native_override=False)

        transform_exprs = self.get_transform_exprs()

        X = self._apply_transform_exprs(X, transform_exprs)

        return _return_narwhals_or_native_dataframe(X, self.return_native)


@register
class CategoricalImputer(BaseImputer):
    """Class to handle arbitrary imputation into categorical columns.

    Attributes
    ----------
    impute_value : string
        Value to impute nulls with.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    """

    polars_compatible = True

    lazyframe_compatible = True

    jsonable = True

    FITS = False

    @beartype
    def __init__(
        self,
        impute_value: str,
        columns: str | list[str],
        add_indicator: bool = False,
        **kwargs: bool | None,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        impute_value : str
            Value to impute nulls with.
        columns : str or list of strs
            Columns to impute
        add_indicator : bool, default = False
            Should indicator columns be created for the nulls filled in? If True,
            a boolean column named "<column>_nulls" is added for each column in
            columns, recording whether the incoming value was null before imputation.
        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns=columns, add_indicator=add_indicator, **kwargs)

        self.impute_values_ = {}
        self.impute_value = impute_value

        for c in self.columns:
            self.impute_values_[c] = self.impute_value

        self.is_fitted_ = True  # Does not fit

    def get_transform_exprs(self) -> list[nw.Expr]:
        """Get transform expressions.

        Returns
        -------
        list[nw.Expr]: transform expressions for class

        """
        return impute_categorical_columns(
            columns=self.columns, impute_values=self.impute_values_
        )

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Impute missing values with the supplied impute_value.

        Parameters
        ----------
        X : DataFrame
            Data containing columns to impute.

        Returns
        -------
        X : DataFrame
            Transformed input X with nulls imputed with the specified impute_value, for the specified columns.

        Raises
        ------
        TypeError: if provided data is incompatible with provided impute value

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> test_df = pl.DataFrame({"a": ["cat", None, "dog"]})
        >>> test_df = test_df.with_columns(pl.col("a").cast(pl.Categorical))
        >>> imputer = CategoricalImputer(columns=["a"], impute_value="missing")
        >>> imputer.transform(test_df)
        shape: (3, 1)
        ┌─────────┐
        │ a       │
        │ ---     │
        │ cat     │
        ╞═════════╡
        │ cat     │
        │ missing │
        │ dog     │
        └─────────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)

        X = BaseTransformer.transform(self, X, return_native_override=False)

        schema = X.collect_schema()

        bad_types = [
            schema[col]
            for col in self.columns
            if not isinstance(schema[col], (nw.Categorical, nw.Unknown))
        ]

        if bad_types:
            msg = f"""
                {self.classname()}: transformer can only handle Categorical/Unknown type columns
                but got columns with types {bad_types}
                """
            raise TypeError(
                msg,
            )

        transform_exprs = self.get_transform_exprs()

        X = self._apply_transform_exprs(X, transform_exprs)

        return _return_narwhals_or_native_dataframe(X, self.return_native)


@register
class _EnumImputer(BaseImputer):
    """Private class to handle arbitrary imputation into enum columns.

    Currently keeping this private, as it's not easy to make a stateless
    get_transform_exprs method for this one, as knowledge of the enum
    categories for each column are required for the transform.

    Attributes
    ----------
    impute_value : string
        Value to impute nulls with.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    """

    polars_compatible = True

    lazyframe_compatible = True

    jsonable = True

    FITS = False

    @beartype
    def __init__(
        self,
        impute_value: str,
        columns: str | list[str],
        add_indicator: bool = False,
        **kwargs: bool | None,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        impute_value : str
            Value to impute nulls with.
        columns : str or list of strs
            Columns to impute
        add_indicator : bool, default = False
            Should indicator columns be created for the nulls filled in? If True,
            a boolean column named "<column>_nulls" is added for each column in
            columns, recording whether the incoming value was null before imputation.
        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns=columns, add_indicator=add_indicator, **kwargs)

        self.impute_values_ = {}
        self.impute_value = impute_value

        for c in self.columns:
            self.impute_values_[c] = self.impute_value

        self.is_fitted_ = True  # Does not fit

    def get_transform_exprs(
        self, columns_to_categories: dict[str, list[Any]]
    ) -> list[nw.Expr]:
        """Get transform expressions.

        Parameters
        ----------
        columns_to_categories:
            dict mapping columns to the categories in their enum class

        Returns
        -------
        list[nw.Expr]: transform expressions for class

        """
        return impute_enum_columns(
            columns=self.columns,
            impute_values=self.impute_values_,
            columns_to_categories=columns_to_categories,
        )

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Impute missing values with the supplied impute_value.

        Parameters
        ----------
        X : DataFrame
            Data containing columns to impute.

        Returns
        -------
        X : DataFrame
            Transformed input X with nulls imputed with the specified impute_value, for the specified columns.

        Raises
        ------
        TypeError: if provided data is incompatible with provided impute value

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> test_df = pl.DataFrame({"a": ["cat", None, "dog"]})
        >>> test_df = test_df.with_columns(pl.col("a").cast(pl.Enum(categories=["cat", "dog"])))
        >>> imputer = _EnumImputer(columns=["a"], impute_value="missing")
        >>> imputer.transform(test_df)
        shape: (3, 1)
        ┌─────────┐
        │ a       │
        │ ---     │
        │ enum    │
        ╞═════════╡
        │ cat     │
        │ missing │
        │ dog     │
        └─────────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)

        X = BaseTransformer.transform(self, X, return_native_override=False)

        schema = X.collect_schema()

        bad_types = [
            schema[col]
            for col in self.columns
            if not isinstance(schema[col], (nw.Enum, nw.Unknown))
        ]

        if bad_types:
            msg = f"""
                {self.classname()}: transformer can only handle Enum/Unknown type columns
                but got columns with types {bad_types}
                """
            raise TypeError(
                msg,
            )

        columns_to_categories = {}
        for col in self.columns:
            columns_to_categories[col] = schema[col].categories

        transform_exprs = self.get_transform_exprs(
            columns_to_categories=columns_to_categories
        )

        X = self._apply_transform_exprs(X, transform_exprs)

        return _return_narwhals_or_native_dataframe(X, self.return_native)


@register
class MedianImputer(BaseImputer, WeightColumnMixin):
    """Transformer to impute missing values with the median of the supplied columns.

    Attributes
    ----------
    impute_values_ : dict
        Created during fit method. Dictionary of float / int (median) values of columns
        in the columns attribute. Keys of impute_values_ give the column names.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Examples
    --------
    ```pycon
    >>> median_imputer = MedianImputer(
    ...     columns=["a", "b"],
    ... )
    >>> median_imputer
    MedianImputer(columns=['a', 'b'])

    >>> # once fit, transformer can also be dumped to json and reinitialised

    >>> test_df = pl.DataFrame({"a": [0, None], "b": [None, 1]})

    >>> _ = median_imputer.fit(test_df)

    >>> json_dump = median_imputer.to_json()
    >>> json_dump
    {'tubular_version': ..., 'classname': 'MedianImputer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'weights_column': None, 'add_indicator': False}, 'fit': {'is_fitted_': True, 'impute_values_': {'a': 0.0, 'b': 1.0}}}

    >>> MedianImputer.from_json(json_dump)
    MedianImputer(columns=['a', 'b'])

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    jsonable = True

    FITS = True

    @beartype
    def __init__(
        self,
        columns: str | list[str],
        weights_column: str | None = None,
        add_indicator: bool = False,
        **kwargs: bool,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : None or str or list, default = None
            Columns to impute, if the default of None is supplied all columns in X are used
            when the transform method is called.

        weights_column: None or str, default=None
            Column containing weights

        add_indicator : bool, default = False
            Should indicator columns be created for the nulls filled in? If True,
            a boolean column named "<column>_nulls" is added for each column in
            columns, recording whether the incoming value was null before imputation.

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns=columns, add_indicator=add_indicator, **kwargs)
        self.weights_column = weights_column

    @block_from_json
    @beartype
    def fit(self, X: DataFrame, y: Series | LazyFrame | None = None) -> MedianImputer:
        """Calculate median values to impute with from X.

        Parameters
        ----------
        X : DataFrame
            Data to "learn" the median values from.

        y : Series or LazyFrame or None, default = None
            Not required.

        Returns
        -------
            MedianImputer:
                fitted class instance.

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> test_df = pl.DataFrame({"a": [1, None, 2], "b": [3, None, 4]})
        >>> imputer = MedianImputer(columns=["a", "b"])
        >>> imputer = imputer.fit(test_df)
        >>> imputer.transform(test_df)
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ f64 ┆ f64 │
        ╞═════╪═════╡
        │ 1.0 ┆ 3.0 │
        │ 1.5 ┆ 3.5 │
        │ 2.0 ┆ 4.0 │
        └─────┴─────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)
        y = _convert_series_to_narwhals(y)

        super().fit(X, y)

        self.impute_values_ = {}

        # as median depends on data ordering, it is less amenable to writing in
        # pure expression form, so implementation here is still
        # slightly pandas-like
        # also, the weighted median approach is genuinely different to the unweighted
        # approach, so have left as two separate logic flows
        if self.weights_column is not None:
            WeightColumnMixin.check_weights_column(self, X, self.weights_column)
            valid_weights_filter_expr = WeightColumnMixin.get_valid_weights_filter_expr(
                self.weights_column, self.verbose
            )
            X_temp = X.filter(valid_weights_filter_expr)

            for c in self.columns:
                col_not_null_expr = ~nw.col(c).is_null()

                X_c = X_temp.filter(col_not_null_expr)

                median_expr = _get_median_calculation_expression(
                    values_column=c,
                    weights_column=self.weights_column,
                )

                # impute value is weighted median
                self.impute_values_[c] = _collect_frame(X_c.select(median_expr)).item(
                    0, 0
                )

        else:
            median_exprs = {
                c: _get_median_calculation_expression(
                    values_column=c, weights_column=None
                )
                for c in self.columns
            }
            results_dict = (
                _collect_frame(X)
                .select(
                    **median_exprs,
                )
                .to_dict(as_series=False)
            )

            self.impute_values_.update(
                {col: value[0] for col, value in results_dict.items()},
            )

        self._check_for_failed_fit()
        self.is_fitted_ = True

        return self


@register
class MeanImputer(WeightColumnMixin, BaseImputer):
    """Transformer to impute missing values with the mean of the supplied columns.

    Attributes
    ----------
    impute_values_ : dict
        Created during fit method. Dictionary of float / int (mean) values of columns
        in the columns attribute. Keys of impute_values_ give the column names.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Examples
    --------
    ```pycon
    >>> mean_imputer = MeanImputer(
    ...     columns=["a", "b"],
    ... )
    >>> mean_imputer
    MeanImputer(columns=['a', 'b'])

    >>> # once fit, transformer can also be dumped to json and reinitialised

    >>> test_df = pl.DataFrame({"a": [0, None], "b": [None, 1]})

    >>> _ = mean_imputer.fit(test_df)

    >>> json_dump = mean_imputer.to_json()
    >>> json_dump
    {'tubular_version': ..., 'classname': 'MeanImputer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'weights_column': None, 'add_indicator': False}, 'fit': {'is_fitted_': True, 'impute_values_': {'a': 0.0, 'b': 1.0}}}

    >>> MeanImputer.from_json(json_dump)
    MeanImputer(columns=['a', 'b'])

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    jsonable = True

    FITS = True

    @beartype
    def __init__(
        self,
        columns: str | list[str],
        weights_column: str | None = None,
        add_indicator: bool = False,
        **kwargs: bool,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : None or str or list, default = None
            Columns to impute, if the default of None is supplied all columns in X are used
            when the transform method is called.

        weights_column : None or str, default = None
            Column containing weights.

        add_indicator : bool, default = False
            Should indicator columns be created for the nulls filled in? If True,
            a boolean column named "<column>_nulls" is added for each column in
            columns, recording whether the incoming value was null before imputation.

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns=columns, add_indicator=add_indicator, **kwargs)
        self.weights_column = weights_column

    @block_from_json
    @beartype
    def fit(self, X: DataFrame, y: Series | LazyFrame | None = None) -> MeanImputer:
        """Calculate mean values to impute with from X.

        Parameters
        ----------
        X : DataFrame
            Data to "learn" the mean values from.

        y : Series or LazyFrame or None, default = None
            Not required.

        Returns
        -------
            MeanImputer:
                fitted class instance.

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> test_df = pl.DataFrame({"a": [1, None, 2], "b": [3, None, 4]})
        >>> imputer = MeanImputer(columns=["a", "b"])
        >>> imputer = imputer.fit(test_df)
        >>> imputer.transform(test_df)
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ f64 ┆ f64 │
        ╞═════╪═════╡
        │ 1.0 ┆ 3.0 │
        │ 1.5 ┆ 3.5 │
        │ 2.0 ┆ 4.0 │
        └─────┴─────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)
        y = _convert_series_to_narwhals(y)

        super().fit(X, y)

        self.impute_values_ = {}

        if not self.columns:
            return self

        weights_column = self.weights_column
        if self.weights_column is None:
            X, weights_column = WeightColumnMixin._create_unit_weights_column(
                X,
                return_native=False,
                verbose=self.verbose,
            )

        WeightColumnMixin.check_weights_column(self, X, weights_column)
        valid_weights_filter_expr = WeightColumnMixin.get_valid_weights_filter_expr(
            weights_column, self.verbose
        )
        X = X.filter(valid_weights_filter_expr)

        weighted_mean_exprs = _get_mean_calculation_expressions(
            self.columns,
            weights_column,
        )

        results_dict = _collect_frame(X.select(**weighted_mean_exprs)).to_dict(
            as_series=False
        )

        # results looks like {key: [value]} so extract value from list
        self.impute_values_.update(
            {col: value[0] for col, value in results_dict.items()},
        )

        self._check_for_failed_fit()
        self.is_fitted_ = True

        return self


@register
class ModeImputer(BaseImputer, WeightColumnMixin):
    """Transformer to impute missing values with the mode of the supplied columns.

    If mode is NaN, a warning will be raised.

    Attributes
    ----------
    impute_values_ : dict
        Created during fit method. Dictionary of float / int (mode) values of columns
        in the columns attribute. Keys of impute_values_ give the column names.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Examples
    --------
    ```pycon
    >>> mode_imputer = ModeImputer(
    ...     columns=["a", "b"],
    ... )
    >>> mode_imputer
    ModeImputer(columns=['a', 'b'])

    >>> # once fit, transformer can also be dumped to json and reinitialised

    >>> test_df = pl.DataFrame({"a": [0, None], "b": [None, 1]})

    >>> _ = mode_imputer.fit(test_df)

    >>> json_dump = mode_imputer.to_json()
    >>> json_dump
    {'tubular_version': ..., 'classname': 'ModeImputer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'weights_column': None, 'add_indicator': False}, 'fit': {'is_fitted_': True, 'impute_values_': {'a': 0, 'b': 1}}}

    >>> ModeImputer.from_json(json_dump)
    ModeImputer(columns=['a', 'b'])

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    jsonable = True

    FITS = True

    @beartype
    def __init__(
        self,
        columns: str | list[str],
        weights_column: str | None = None,
        add_indicator: bool = False,
        **kwargs: bool,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : None or str or list, default = None
            Columns to impute, if the default of None is supplied all columns in X are used
            when the transform method is called.

        weights_column : str
            Name of weights columns to use if mode should be in terms of sum of weights
            not count of rows.

        add_indicator : bool, default = False
            Should indicator columns be created for the nulls filled in? If True,
            a boolean column named "<column>_nulls" is added for each column in
            columns, recording whether the incoming value was null before imputation.

        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns=columns, add_indicator=add_indicator, **kwargs)
        self.weights_column = weights_column

    @block_from_json
    @beartype
    def fit(self, X: DataFrame, y: Series | LazyFrame | None = None) -> ModeImputer:
        """Calculate mode values to impute with from X.

        In the event of a tie, the highest modal value will be returned.

        Parameters
        ----------
        X : DataFrame
            Data to "learn" the mode values from.

        y : Series or LazyFrame or None, default = None
            Not required.

        Returns
        -------
        ModeImputer:
            fitted class instance

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> test_df = pl.DataFrame({"a": [1, None, 2], "b": [3, None, 4]})
        >>> imputer = ModeImputer(columns=["a", "b"])
        >>> imputer = imputer.fit(test_df)
        >>> imputer.transform(test_df)
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 3   │
        │ 2   ┆ 4   │
        │ 2   ┆ 4   │
        └─────┴─────┘

        ```

        """
        X = _convert_dataframe_to_narwhals(X)
        y = _convert_series_to_narwhals(y)

        super().fit(X, y)

        self.impute_values_ = {}

        weights_column = self.weights_column
        if self.weights_column is None:
            X, weights_column = WeightColumnMixin._create_unit_weights_column(
                X,
                return_native=False,
                verbose=self.verbose,
            )

        WeightColumnMixin.check_weights_column(self, X, weights_column)
        valid_weights_filter_expr = WeightColumnMixin.get_valid_weights_filter_expr(
            weights_column, self.verbose
        )
        X = X.filter(valid_weights_filter_expr)

        self.impute_values_ = {}

        for c in self.columns:
            group = (
                X.filter(~nw.col(c).is_null())
                .group_by(c)
                .agg(nw.col(weights_column).sum().alias(f"{c}_total_weight"))
                .filter(
                    nw.col(f"{c}_total_weight") == nw.col(f"{c}_total_weight").max()
                )
            )

            results_dict = _collect_frame(group).to_dict(as_series=True)

            mode_values = results_dict[c].sort(descending=True).to_list()

            n_mode_vals = len(mode_values)

            if n_mode_vals == 0:
                mode_value = None

            elif n_mode_vals == 1:
                mode_value = mode_values[0]

            elif n_mode_vals > 1:
                warnings.warn(
                    f"ModeImputer: The Mode of column {c} is tied, will sort in descending order and return first candidate",
                    stacklevel=2,
                )

                mode_value = mode_values[0]

            self.impute_values_[c] = mode_value

        self._check_for_failed_fit()
        self.is_fitted_ = True

        return self


@register
class NullIndicator(BaseTransformer):
    """Class to create a binary indicator column for null values.

    Attributes
    ----------
    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Notes
    -----
    The indicators created by this transformer reflect the values in the data
    passed to transform. If a pipeline imputes nulls before this transformer,
    the indicators will see the imputed (non-null) values and come out all
    False. Where both the imputed value and the original null indicator are
    needed, use the ``add_indicator`` option on the relevant imputer instead;
    it creates both from the same incoming row, so the pairing survives
    reordering the pipeline and never carries training null patterns into
    prediction rows.

    Examples
    --------
    ```pycon
    >>> null_indicator = NullIndicator(
    ...     columns=["a", "b"],
    ... )
    >>> null_indicator
    NullIndicator(columns=['a', 'b'])

    >>> # transformer can also be dumped to json and reinitialised
    >>> json_dump = null_indicator.to_json()
    >>> json_dump
    {'tubular_version': ..., 'classname': 'NullIndicator', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True}, 'fit': {'is_fitted_': True}}

    >>> NullIndicator.from_json(json_dump)
    NullIndicator(columns=['a', 'b'])

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    FITS = False

    jsonable = True

    @beartype
    def __init__(
        self,
        columns: ListOfStrs | str,
        **kwargs: bool | None,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        columns : None or str or list, default = None
            Columns to produce indicator columns for, if the default of None is supplied all columns in X are used
            when the transform method is called.

        kwargs: bool
            arguments for base class, e.g. verbose.

        """
        super().__init__(columns=columns, **kwargs)
        self.is_fitted_ = True  # does not fit

    @property
    def null_indicator_columns_(self) -> dict[str, str]:
        """Mapping of input columns to the indicator columns created for them.

        Returns
        -------
        dict[str, str]:
            mapping of column name to indicator column name, one entry per column
            in columns. Useful for interpreting output column by column, and
            matches the naming used by the imputers' add_indicator option.

        Examples
        --------
        ```pycon
        >>> indicator = NullIndicator(columns=["a", "b"])

        >>> indicator.null_indicator_columns_
        {'a': 'a_nulls', 'b': 'b_nulls'}

        ```

        """
        return {
            column: get_null_indicator_column_name(column) for column in self.columns
        }

    def get_feature_names_out(self) -> list[str]:
        """List features modified/created by the transformer.

        Returns
        -------
        list[str]:
            list of indicator columns created by the transformer

        Examples
        --------
        ```pycon
        >>> indicator = NullIndicator(columns=["a", "b"])

        >>> indicator.get_feature_names_out()
        ['a_nulls', 'b_nulls']

        ```

        """
        return list(self.null_indicator_columns_.values())

    def get_transform_exprs(self) -> list[nw.Expr]:
        """Get transform expressions.

        Returns
        -------
        list[nw.Expr]: transform expressions for class

        """
        return indicate_nulls_for_columns(
            columns=self.columns,
        )

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Create new columns indicating the position of null values for each variable in self.columns.

        Parameters
        ----------
        X : DataFrame
            Data to add indicators to.

        Returns
        -------
        DataFrame:
            dataframe with null indicator columns added

        Examples
        --------
        --------,
        ```pycon
        >>> import polars as pl
        >>> test_df = pl.DataFrame({"a": [1, None, 2], "b": [3, None, 4]})
        >>> imputer = NullIndicator(columns=["a", "b"])
        >>> imputer.transform(test_df)
        shape: (3, 4)
        ┌──────┬──────┬─────────┬─────────┐
        │ a    ┆ b    ┆ a_nulls ┆ b_nulls │
        │ ---  ┆ ---  ┆ ---     ┆ ---     │
        │ i64  ┆ i64  ┆ bool    ┆ bool    │
        ╞══════╪══════╪═════════╪═════════╡
        │ 1    ┆ 3    ┆ false   ┆ false   │
        │ null ┆ null ┆ true    ┆ true    │
        │ 2    ┆ 4    ┆ false   ┆ false   │
        └──────┴──────┴─────────┴─────────┘

        ```

        """
        X = super().transform(X, return_native_override=False)

        X = _convert_dataframe_to_narwhals(X)

        transform_exprs = self.get_transform_exprs()

        X = X.with_columns(*transform_exprs) if transform_exprs else X

        return X if not self.return_native else X.to_native()


# DEPRECATED TRANSFORMERS


@deprecated(
    """This transformer has been deprecated and replaced with type specific transformer
    classes, please use NumberImputer/BooleanImputer/StringImputer/CategoricalImputer from
    this file instead.
    """,
)
@register
class ArbitraryImputer(BaseImputer):
    """Transformer to impute null values with an arbitrary pre-defined value.

    Attributes
    ----------
    impute_value : int or float or str or bool
        Value to impute nulls with.

    built_from_json: bool
        indicates if transformer was reconstructed from json, which limits it's supported
        functionality to .transform

    polars_compatible : bool
        class attribute, indicates whether transformer has been converted to polars/pandas agnostic narwhals framework

    return_native: bool, default = True
        Controls whether transformer returns narwhals or native pandas/polars type

    jsonable: bool
        class attribute, indicates if transformer supports to/from_json methods

    FITS: bool
        class attribute, indicates whether transform requires fit to be run first

    lazyframe_compatible: bool
        class attribute, indicates whether transformer works with lazyframes

    Examples
    --------
    ```pycon
    >>> arbitrary_imputer = ArbitraryImputer(columns=["a", "b"], impute_value=5)
    >>> arbitrary_imputer
    ArbitraryImputer(columns=['a', 'b'], impute_value=5)

    >>> # transformer can also be dumped to json and reinitialised
    >>> json_dump = arbitrary_imputer.to_json()
    >>> json_dump
    {'tubular_version': ..., 'classname': 'ArbitraryImputer', 'init': {'columns': ['a', 'b'], 'copy': False, 'verbose': False, 'return_native': True, 'impute_value': 5, 'add_indicator': False}, 'fit': {'is_fitted_': True, 'impute_values_': {'a': 5, 'b': 5}}}

    >>> ArbitraryImputer.from_json(json_dump)
    ArbitraryImputer(columns=['a', 'b'], impute_value=5)

    ```

    """

    polars_compatible = True

    lazyframe_compatible = True

    jsonable = True

    FITS = False

    @beartype
    def __init__(
        self,
        impute_value: int | float | str | bool,
        columns: str | list[str],
        add_indicator: bool = False,
        **kwargs: bool | None,
    ) -> None:
        """Initialise class instance.

        Parameters
        ----------
        impute_value : int or float or str or bool
            Value to impute nulls with.
        columns : str or list of strs
            Columns to impute
        add_indicator : bool, default = False
            Should indicator columns be created for the nulls filled in? If True,
            a boolean column named "<column>_nulls" is added for each column in
            columns, recording whether the incoming value was null before imputation.
        **kwargs
            Arbitrary keyword arguments passed onto BaseTransformer.init method.

        """
        super().__init__(columns=columns, add_indicator=add_indicator, **kwargs)

        self.impute_values_ = {}
        self.impute_value = impute_value

        for c in self.columns:
            self.impute_values_[c] = self.impute_value
        self.is_fitted_ = True  # Does not fit

    @beartype
    def transform(self, X: DataFrame) -> DataFrame:
        """Impute missing values with the supplied impute_value.

        Parameters
        ----------
        X : DataFrame
            Data containing columns to impute.

        Returns
        -------
        X : DataFrame
            Transformed input X with nulls imputed with the specified impute_value, for the specified columns.

        Raises
        ------
        TypeError:
            if impute_value and column types do not work together

        Examples
        --------
        ```pycon
        >>> import polars as pl
        >>> test_df = pl.DataFrame({"a": [1, None, 2], "b": [3, None, 4]})
        >>> imputer = ArbitraryImputer(columns=["a", "b"], impute_value=5)
        >>> imputer.transform(test_df)
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 3   │
        │ 5   ┆ 5   │
        │ 2   ┆ 4   │
        └─────┴─────┘

        ```

        """
        self.check_is_fitted("is_fitted_")
        X = _convert_dataframe_to_narwhals(X)
        schema = X.schema

        if isinstance(self.impute_value, (int, float)) and not isinstance(
            self.impute_value, bool
        ):
            imp = NumberImputer(
                columns=self.columns,
                impute_value=self.impute_value,
                add_indicator=self.add_indicator,
                return_native=self.return_native,
            )

        elif isinstance(self.impute_value, str):
            if all(
                isinstance(schema[col], (nw.String, nw.Unknown)) for col in self.columns
            ):
                imp = StringImputer(
                    columns=self.columns,
                    impute_value=self.impute_value,
                    add_indicator=self.add_indicator,
                    return_native=self.return_native,
                )

            elif all(isinstance(schema[col], nw.Categorical) for col in self.columns):
                imp = CategoricalImputer(
                    columns=self.columns,
                    impute_value=self.impute_value,
                    add_indicator=self.add_indicator,
                    return_native=self.return_native,
                )

            elif all(isinstance(schema[col], nw.Enum) for col in self.columns):
                imp = _EnumImputer(
                    columns=self.columns,
                    impute_value=self.impute_value,
                    add_indicator=self.add_indicator,
                    return_native=self.return_native,
                )

            else:
                msg = f"""{self.classname()}: str impute values can only be used on
                String/Categorical/Enum type columns, and types must be consistent
                in each usage, but got:
                impute_value type: {type(self.impute_value)}
                column types: {[schema[col] for col in self.columns]}
                """
                raise TypeError(msg)

        else:
            imp = BooleanImputer(
                columns=self.columns,
                impute_value=self.impute_value,
                add_indicator=self.add_indicator,
                return_native=self.return_native,
            )

        return imp.transform(X)
