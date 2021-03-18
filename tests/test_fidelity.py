import numpy
import pytest
from matchms import Spectrum
from matchms.similarity import Fidelity


@pytest.fixture(autouse=True, scope='function')
def none_spectrum():
    return None


@pytest.fixture
def spectrum1():
    return Spectrum(mz=numpy.array([100, 200, 300, 500, 510], dtype="float"),
                    intensities=numpy.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float"))


@pytest.fixture
def spectrum2():
    return Spectrum(mz=numpy.array([100, 200, 290, 490, 510], dtype="float"),
                    intensities=numpy.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float"))


@pytest.fixture
def empty_spectrum():
    return Spectrum(mz=numpy.array([], dtype="float"),
                    intensities=numpy.array([], dtype="float"))


@pytest.mark.parametrize("tolerance, expected_score, expected_matches", [(0.1, 0.1225, 3), (20, 1.0, 5)])
def test_fidelity_score(tolerance, expected_score, expected_matches, spectrum1, spectrum2):
    """Compare output fidelity distance with scores computed with manual calculator."""

    fidelity = Fidelity(tolerance=tolerance)
    score = fidelity.pair(spectrum1, spectrum2)
    assert score["score"] == pytest.approx(expected_score)
    assert score["matches"] == expected_matches


def test_none_exception(none_spectrum, spectrum1):
    """ Test for exception being thrown if the reference or query spectrum is none. """

    fidelity = Fidelity(tolerance=0.0)
    with pytest.raises(AttributeError) as exception:
        fidelity.pair(none_spectrum, spectrum1)

    message = exception.value.args[0]
    assert message == "'NoneType' object has no attribute 'peaks'"
