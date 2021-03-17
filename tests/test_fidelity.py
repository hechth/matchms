import numpy
import pytest
from matchms import Spectrum
from matchms.similarity import Fidelity


@pytest.mark.parametrize("tolerance, expected_score, expected_matches", [(0.1, 0.1225, 3)])
def test_fidelity(tolerance, expected_score, expected_matches):
    """Compare output fidelity distance with scores computed via daphnis package implementation."""
    spectrum_1 = Spectrum(mz=numpy.array([100, 200, 300, 500, 510], dtype="float"),
                          intensities=numpy.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float"))

    spectrum_2 = Spectrum(mz=numpy.array([100, 200, 290, 490, 510], dtype="float"),
                          intensities=numpy.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float"))

    fidelity = Fidelity(tolerance=tolerance)
    score = fidelity.pair(spectrum_1, spectrum_2)
    assert score["score"] == pytest.approx(expected_score)
    assert score["matches"] == expected_matches


def test_none_exception():
    reference = None
    query = Spectrum(mz=numpy.array([100, 200, 290, 490, 510], dtype="float"),
                     intensities=numpy.array([0.1, 0.2, 1.0, 0.3, 0.4],
                                             dtype="float"))

    fidelity = Fidelity(tolerance=0.0)
    with pytest.raises(AttributeError) as exception:
        fidelity.pair(reference, query)

    message = exception.value.args[0]
    assert message == "'NoneType' object has no attribute 'peaks'"
