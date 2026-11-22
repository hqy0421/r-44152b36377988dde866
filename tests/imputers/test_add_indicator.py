"""Tests for the add_indicator option shared by the imputers."""

import narwhals as nw
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from tests.utils import (
    _collect_frame,
    _convert_to_lazy,
    _handle_from_json,
    assert_frame_equal_dispatch,
    dataframe_init_dispatch,
)
from tubular.functions.imputers import get_null_indicator_column_name
from tubular.imputers import (
    ArbitraryImputer,
    BooleanImputer,
    CategoricalImputer,
    MeanImputer,
    MedianImputer,
    ModeImputer,
    NullIndicator,
    NumberImputer,
    StringImputer,
)
from tubular.pipeline import dump_pipeline_to_json, load_pipeline_from_json

IMPUTER_CASES = [
    # transformer factory, column, raw values, imputed values
    (
        lambda **kwargs: NumberImputer(impute_value=0, **kwargs),
        "a",
        [1, None, 3],
        [1, 0, 3],
    ),
    (
        lambda **kwargs: StringImputer(impute_value="missing", **kwargs),
        "a",
        ["x", None, "z"],
        ["x", "missing", "z"],
    ),
    (
        lambda **kwargs: BooleanImputer(impute_value=True, **kwargs),
        "a",
        [True, None, False],
        [True, True, False],
    ),
]


def _build_df(values, library):
    return dataframe_init_dispatch(
        {"a": values, "keep": list(range(len(values)))},
        library,
    )


class TestIndicatorValues:
    """The indicator should describe the incoming row, not the imputed row."""

    @pytest.mark.parametrize(("factory", "column", "raw", "imputed"), IMPUTER_CASES)
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("add_indicator", [True, False])
    def test_stateless_imputers(
        self,
        factory,
        column,
        raw,
        imputed,
        library,
        add_indicator,
    ):
        df = _build_df(raw, library)
        transformer = factory(columns=[column], add_indicator=add_indicator)

        output = nw.from_native(transformer.transform(df))

        assert output[column].to_list() == imputed

        indicator_name = get_null_indicator_column_name(column)
        if add_indicator:
            assert output[indicator_name].to_list() == [val is None for val in raw]
            assert str(output[indicator_name].dtype) == "Boolean"
        else:
            # default behaviour must not change the output columns
            assert indicator_name not in output.columns

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_categorical_imputer(self, library):
        df = dataframe_init_dispatch(
            {"a": ["x", None, "z"], "keep": [1, 2, 3]},
            library,
        )
        df = (
            nw.from_native(df)
            .with_columns(nw.col("a").cast(nw.Categorical))
            .to_native()
        )

        transformer = CategoricalImputer(
            columns=["a"],
            impute_value="missing",
            add_indicator=True,
        )

        output = nw.from_native(transformer.transform(df))

        assert output["a_nulls"].to_list() == [False, True, False]

    @pytest.mark.parametrize(
        "transformer_class", [MeanImputer, MedianImputer, ModeImputer]
    )
    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_fitted_imputers(self, transformer_class, library):
        """Indicators are created from the rows being transformed, not from fit data."""
        fit_df = _build_df([1.0, 2.0, 3.0], library)
        predict_df = _build_df([None, 5.0, 6.0], library)

        transformer = transformer_class(columns=["a"], add_indicator=True)
        transformer.fit(fit_df)

        output = nw.from_native(transformer.transform(predict_df))

        assert output["a_nulls"].to_list() == [True, False, False]
        # the null is filled with a value learnt from fit data
        assert output["a"].to_list()[0] == transformer.impute_values_["a"]

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_no_training_null_leakage_into_predictions(self, library):
        """A column with no nulls at fit must still flag nulls at predict time."""
        fit_df = _build_df([1.0, 2.0, 3.0], library)
        predict_df = _build_df([None, None, None], library)

        transformer = MeanImputer(columns=["a"], add_indicator=True)
        transformer.fit(fit_df)

        output = nw.from_native(transformer.transform(predict_df))

        assert output["a_nulls"].to_list() == [True, True, True]

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("from_json", [True, False])
    def test_output_matches_expected_frame(self, library, lazy, from_json):
        df = _build_df([1.0, None, 3.0], library)

        transformer = NumberImputer(
            columns=["a"],
            impute_value=0,
            add_indicator=True,
        )

        transformer = _handle_from_json(transformer, from_json)

        # lazy is only meaningful for polars
        use_lazy = lazy and library == "polars"

        output = _collect_frame(
            transformer.transform(_convert_to_lazy(df, use_lazy)),
            use_lazy,
        )

        expected = nw.from_native(_build_df([1.0, 0.0, 3.0], library)).with_columns(
            nw.new_series(
                name="a_nulls",
                values=[False, True, False],
                backend=library,
                dtype=nw.Boolean,
            ),
        )

        assert_frame_equal_dispatch(
            nw.from_native(output).to_native(),
            expected.to_native(),
        )

    def test_empty_columns(self):
        df = _build_df([1.0, None, 3.0], "polars")

        transformer = NumberImputer(
            columns=[],
            impute_value=0,
            add_indicator=True,
        )

        output = transformer.transform(df)

        assert output.columns == df.columns

    @pytest.mark.parametrize(
        ("impute_value", "values"),
        [
            (0, [1, None, 3]),
            ("missing", ["x", None, "z"]),
            (True, [True, None, False]),
        ],
    )
    def test_arbitrary_imputer_forwards_add_indicator(self, impute_value, values):
        """The deprecated wrapper must pass add_indicator to the imputer it delegates to."""
        df = dataframe_init_dispatch(
            {"a": values, "keep": list(range(len(values)))},
            "polars",
        )

        transformer = ArbitraryImputer(
            columns=["a"],
            impute_value=impute_value,
            add_indicator=True,
        )

        output = nw.from_native(transformer.transform(df))

        assert output["a_nulls"].to_list() == [False, True, False]
        assert transformer.null_indicator_columns_ == {"a": "a_nulls"}


class TestColumnInterpretation:
    """The indicator and value columns should be paired and named consistently."""

    @pytest.mark.parametrize(
        "transformer",
        [
            NumberImputer(columns=["a", "b"], impute_value=0, add_indicator=True),
            StringImputer(
                columns=["a", "b"], impute_value="missing", add_indicator=True
            ),
            BooleanImputer(columns=["a", "b"], impute_value=True, add_indicator=True),
            MeanImputer(columns=["a", "b"], add_indicator=True),
            MedianImputer(columns=["a", "b"], add_indicator=True),
            ModeImputer(columns=["a", "b"], add_indicator=True),
        ],
    )
    def test_null_indicator_columns_mapping(self, transformer):
        assert transformer.null_indicator_columns_ == {
            "a": "a_nulls",
            "b": "b_nulls",
        }

    def test_null_indicator_mapping_matches_null_indicator_transformer(self):
        imputer = NumberImputer(columns=["a", "b"], impute_value=0, add_indicator=True)
        indicator = NullIndicator(columns=["a", "b"])

        assert imputer.null_indicator_columns_ == indicator.null_indicator_columns_

    def test_get_feature_names_out(self):
        imputer = NumberImputer(columns=["a", "b"], impute_value=0, add_indicator=True)

        assert imputer.get_feature_names_out() == ["a", "b", "a_nulls", "b_nulls"]

        imputer = NumberImputer(columns=["a", "b"], impute_value=0)

        assert imputer.get_feature_names_out() == ["a", "b"]

    def test_null_indicator_get_feature_names_out(self):
        indicator = NullIndicator(columns=["a", "b"])

        assert indicator.get_feature_names_out() == ["a_nulls", "b_nulls"]

    def test_default_is_false(self):
        assert NumberImputer(columns=["a"], impute_value=0).add_indicator is False


class TestJson:
    """add_indicator must survive a json round trip."""

    @pytest.mark.parametrize(
        "transformer",
        [
            NumberImputer(columns=["a"], impute_value=0, add_indicator=True),
            StringImputer(columns=["a"], impute_value="missing", add_indicator=True),
            BooleanImputer(columns=["a"], impute_value=True, add_indicator=True),
            CategoricalImputer(
                columns=["a"], impute_value="missing", add_indicator=True
            ),
        ],
    )
    def test_add_indicator_in_json(self, transformer):
        json_dict = transformer.to_json()

        assert json_dict["init"]["add_indicator"] is True

        reloaded = type(transformer).from_json(json_dict)

        assert reloaded.add_indicator is True
        assert reloaded.null_indicator_columns_ == transformer.null_indicator_columns_

    def test_fitted_imputer_json_round_trip(self):
        df = _build_df([1.0, 2.0, 3.0], "polars")

        transformer = MedianImputer(columns=["a"], add_indicator=True).fit(df)

        reloaded = MedianImputer.from_json(transformer.to_json())

        assert reloaded.add_indicator is True

        output = reloaded.transform(_build_df([None, 5.0, 6.0], "polars"))

        assert output["a_nulls"].to_list() == [True, False, False]

    def test_pipeline_json_round_trip(self):
        """A pipeline containing an add_indicator imputer must reopen with the
        option enabled, and produce the same null indicators."""
        df = _build_df([1.0, None, 3.0], "polars")

        pipeline = Pipeline(
            [
                (
                    "imputer",
                    NumberImputer(columns=["a"], impute_value=0, add_indicator=True),
                )
            ]
        )

        reloaded = load_pipeline_from_json(dump_pipeline_to_json(pipeline))

        imputer = reloaded.named_steps["imputer"]
        assert imputer.add_indicator is True
        assert imputer.null_indicator_columns_ == {"a": "a_nulls"}

        output = reloaded.transform(df)

        assert output["a_nulls"].to_list() == [False, True, False]


class TestPipelineOrdering:
    """Regression tests for the ordering issue between separate null indicators
    and imputers."""

    def test_separate_null_indicator_is_order_dependent(self):
        df = _build_df([None, 2.0], "polars")

        indicator_first = Pipeline(
            [
                ("indicator", NullIndicator(columns=["a"])),
                ("imputer", NumberImputer(columns=["a"], impute_value=0)),
            ]
        ).transform(df)

        imputer_first = Pipeline(
            [
                ("imputer", NumberImputer(columns=["a"], impute_value=0)),
                ("indicator", NullIndicator(columns=["a"])),
            ]
        ).transform(df)

        assert indicator_first["a_nulls"].to_list() == [True, False]
        # this is the failure mode add_indicator avoids
        assert imputer_first["a_nulls"].to_list() == [False, False]

    def test_add_indicator_is_order_independent(self):
        df = _build_df([None, 2.0], "polars")

        combined = NumberImputer(
            columns=["a"],
            impute_value=0,
            add_indicator=True,
        ).transform(df)

        indicator_first = Pipeline(
            [
                ("indicator", NullIndicator(columns=["a"])),
                ("imputer", NumberImputer(columns=["a"], impute_value=0)),
            ]
        ).transform(df)

        assert combined["a_nulls"].to_list() == indicator_first["a_nulls"].to_list()
        assert combined["a"].to_list() == indicator_first["a"].to_list()

    def test_combined_imputer_in_pipeline(self):
        df = _build_df([None, 2.0], "polars")

        pipeline = Pipeline(
            [
                (
                    "imputer",
                    NumberImputer(columns=["a"], impute_value=0, add_indicator=True),
                ),
                # a downstream indicator on a column that is not imputed, so it
                # cannot overwrite the imputer's own indicator
                ("indicator", NullIndicator(columns=["keep"])),
            ]
        )

        output = pipeline.transform(df)

        assert output["a_nulls"].to_list() == [True, False]
        assert output["a"].to_list() == [0.0, 2.0]
        assert output["keep_nulls"].to_list() == [False, False]

    def test_downstream_indicator_overwrites_same_named_column(self):
        """Documented caveat: a later NullIndicator on the same column wins.

        Users who want the imputer's indicator should not also apply
        NullIndicator to the same columns downstream.
        """
        df = _build_df([None, 2.0], "polars")

        pipeline = Pipeline(
            [
                (
                    "imputer",
                    NumberImputer(columns=["a"], impute_value=0, add_indicator=True),
                ),
                ("indicator", NullIndicator(columns=["a"])),
            ]
        )

        output = pipeline.transform(df)

        assert output["a_nulls"].to_list() == [False, False]


class TestOtherTypes:
    """Spot check add_indicator does not disturb typing/return_native handling."""

    def test_return_native_false(self):
        df = _build_df([1.0, None, 3.0], "polars")

        transformer = NumberImputer(
            columns=["a"],
            impute_value=0,
            add_indicator=True,
            return_native=False,
        )

        output = transformer.transform(df)

        assert isinstance(output, (nw.DataFrame, nw.LazyFrame))

    @pytest.mark.parametrize("library", ["pandas", "polars"])
    def test_original_df_not_modified(self, library):
        df = _build_df([1.0, None, 3.0], library)
        original = df.clone() if isinstance(df, pl.DataFrame) else df.copy(deep=True)

        NumberImputer(columns=["a"], impute_value=0, add_indicator=True).transform(df)

        assert_frame_equal_dispatch(df, original)
