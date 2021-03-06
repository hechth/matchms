import numpy
import pytest
from matchms import Spectrum
from matchms.similarity import PrecursorMzMatch
from matchms.similarity.PrecursorMzMatch import precursormz_scores
from matchms.similarity.PrecursorMzMatch import precursormz_scores_ppm
from matchms.similarity.PrecursorMzMatch import precursormz_scores_symmetric
from matchms.similarity.PrecursorMzMatch import \
    precursormz_scores_symmetric_ppm


def test_precursormz_match():
    """Test with default tolerance."""
    spectrum_1 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 100.0})

    spectrum_2 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 101.0})

    similarity_score = PrecursorMzMatch()
    score = similarity_score.pair(spectrum_1, spectrum_2)
    assert not score, "Expected different score."


def test_precursormz_match_tolerance2():
    """Test with tolerance > difference."""
    spectrum_1 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 100.0})

    spectrum_2 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 101.0})

    similarity_score = PrecursorMzMatch(tolerance=2.0)
    score = similarity_score.pair(spectrum_1, spectrum_2)
    assert score, "Expected different score."


def test_precursormz_match_tolerance_ppm():
    """Test with tolerance > difference in ppm."""
    spectrum_1 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 600.0})

    spectrum_2 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 600.001})

    similarity_score = PrecursorMzMatch(tolerance=2.0, tolerance_type="ppm")
    score = similarity_score.pair(spectrum_1, spectrum_2)
    assert score, "Expected different score."


def test_precursormz_match_missing_precursormz():
    """Test with missing precursormz."""
    spectrum_1 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 100.0})

    spectrum_2 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={})

    similarity_score = PrecursorMzMatch(tolerance=2.0)

    with pytest.raises(AssertionError) as msg:
        _ = similarity_score.pair(spectrum_1, spectrum_2)

    expected_message_part = "Missing precursor m/z."
    assert expected_message_part in str(msg.value), "Expected particular error message."


def test_precursormz_match_array():
    """Test with array and default tolerance."""
    spectrum_1 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 100.0})

    spectrum_2 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 101.0})

    spectrum_a = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 99.0})

    spectrum_b = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 98.0})

    similarity_score = PrecursorMzMatch()
    scores = similarity_score.matrix([spectrum_1, spectrum_2],
                                     [spectrum_a, spectrum_b])
    assert numpy.all(scores == numpy.array([[False, False],
                                            [False, False]])), "Expected different scores."


def test_precursormz_match_tolerance2_array():
    """Test with array and tolerance=2."""
    spectrum_1 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 100.0})

    spectrum_2 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 101.0})

    spectrum_a = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 99.0})

    spectrum_b = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 98.0})

    similarity_score = PrecursorMzMatch(tolerance=2.0)
    scores = similarity_score.matrix([spectrum_1, spectrum_2],
                                     [spectrum_a, spectrum_b])
    assert numpy.all(scores == numpy.array([[True, True],
                                            [True, False]])), "Expected different scores."


def test_precursormz_match_tolerance2_array_ppm():
    """Test with array and tolerance=2 and type=ppm."""
    spectrum_1 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 100.0})

    spectrum_2 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 101.0})

    spectrum_a = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 99.99})

    spectrum_b = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 98.0})

    similarity_score = PrecursorMzMatch(tolerance=101.0, tolerance_type="ppm")
    scores = similarity_score.matrix([spectrum_1, spectrum_2],
                                     [spectrum_a, spectrum_b])
    assert numpy.all(scores == numpy.array([[True, False],
                                            [False, False]])), "Expected different scores."


def test_precursormz_match_array_symmetric():
    """Test with array and is_symmetric=True."""
    spectrum_1 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 100.0})

    spectrum_2 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 101.0})

    spectrum_3 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 99.95})

    spectrum_4 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 98.0})

    spectrums = [spectrum_1, spectrum_2, spectrum_3, spectrum_4]
    similarity_score = PrecursorMzMatch()
    scores = similarity_score.matrix(spectrums, spectrums, is_symmetric=True)
    scores2 = similarity_score.matrix(spectrums, spectrums, is_symmetric=False)

    assert numpy.all(scores == scores2), "Expected identical scores"
    assert numpy.all(scores == numpy.array(
        [[True, False, True, False],
         [False, True, False, False],
         [True, False, True, False],
         [False, False, False, True]])), "Expected different scores"


def test_precursormz_match_array_symmetric_pmm():
    """Test with array and is_symmetric=True for type="ppm"."""
    spectrum_1 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 100.0})

    spectrum_2 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 100.01})

    spectrum_3 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 99.99999})

    spectrum_4 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"precursor_mz": 99.9})

    spectrums = [spectrum_1, spectrum_2, spectrum_3, spectrum_4]
    similarity_score = PrecursorMzMatch(tolerance=5.0, tolerance_type="ppm")
    scores = similarity_score.matrix(spectrums, spectrums, is_symmetric=True)
    scores2 = similarity_score.matrix(spectrums, spectrums, is_symmetric=False)

    assert numpy.all(scores == scores2), "Expected identical scores"
    assert numpy.all(scores == numpy.array(
        [[True, False, True, False],
         [False, True, False, False],
         [True, False, True, False],
         [False, False, False, True]])), "Expected different scores"


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_precursormz_scores(numba_compiled):
    """Test the underlying score function (pure Python and numba compiled)."""
    precursors_ref = numpy.asarray([101, 200, 300])
    precursors_query = numpy.asarray([100, 301])
    if numba_compiled:
        scores = precursormz_scores(precursors_ref, precursors_query, tolerance=2.0)
    else:
        scores = precursormz_scores.py_func(precursors_ref, precursors_query, tolerance=2.0)
    assert numpy.all(scores == numpy.array([[1., 0.],
                                            [0., 0.],
                                            [0., 1.]])), "Expected different scores."


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_precursormz_scores_symmetric(numba_compiled):
    """Test the underlying score function (non-compiled)."""
    precursors = numpy.asarray([101, 100, 200])
    if numba_compiled:
        scores = precursormz_scores_symmetric(precursors, precursors, tolerance=2.0)
    else:
        scores = precursormz_scores_symmetric.py_func(precursors, precursors, tolerance=2.0)
    assert numpy.all(scores == numpy.array([[1., 1., 0.],
                                            [1., 1., 0.],
                                            [0., 0., 1.]])), "Expected different scores."


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_precursormz_scores_ppm(numba_compiled):
    """Test the underlying score function (pure Python and numba compiled)."""
    precursors_ref = numpy.asarray([100.00001, 200, 300])
    precursors_query = numpy.asarray([100, 300.00001])
    if numba_compiled:
        scores = precursormz_scores_ppm(precursors_ref, precursors_query, tolerance_ppm=2.0)
    else:
        scores = precursormz_scores_ppm.py_func(precursors_ref, precursors_query, tolerance_ppm=2.0)
    assert numpy.all(scores == numpy.array([[1., 0.],
                                            [0., 0.],
                                            [0., 1.]])), "Expected different scores."


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_precursormz_scores_symmetric_ppm(numba_compiled):
    """Test the underlying score function (non-compiled)."""
    precursors = numpy.asarray([100.00001, 100, 200])
    if numba_compiled:
        scores = precursormz_scores_symmetric_ppm(precursors, precursors, tolerance_ppm=2.0)
    else:
        scores = precursormz_scores_symmetric_ppm.py_func(precursors, precursors, tolerance_ppm=2.0)
    assert numpy.all(scores == numpy.array([[1., 1., 0.],
                                            [1., 1., 0.],
                                            [0., 0., 1.]])), "Expected different scores."
