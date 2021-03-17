from typing import List
from typing import Tuple
import numpy
from matchms.similarity.spectrum_similarity_functions import find_matches
from matchms.similarity.spectrum_similarity_functions import fidelity_score
from matchms.similarity.spectrum_similarity_functions import get_peaks_array
from matchms.typing import SpectrumType
from .BaseSimilarity import BaseSimilarity


class Fidelity(BaseSimilarity):
    """Calculate 'fidelity distance' between two spectra.

    TODO: Add documentation here.

    For example

    .. testcode::

        import numpy as np
        from matchms import Spectrum
        from matchms.similarity import Fidelity

        reference = Spectrum(mz=np.array([100, 150, 200.]),
                             intensities=np.array([0.7, 0.2, 0.1]))
        query = Spectrum(mz=np.array([100, 140, 190.]),
                         intensities=np.array([0.4, 0.2, 0.1]))

        # Use factory to construct a similarity function
        fidelity = Fidelity()

        score = fidelity.pair(reference, query)

        print(f"Fidelity score is {score['score']:.2f} with {score['matches']} matched peaks")

    Should output

    .. testoutput::

        TODO: Describe test output.

    """
    # Set key characteristics as class attributes
    is_commutative = True
    # Set output data type, e.g. ("score", "float") or [("score", "float"), ("matches", "int")]
    score_datatype = [("score", numpy.float64), ("matches", int)]

    def __init__(self, tolerance: float = 0.1):
        """
        Parameters
        ----------
        tolerance:
            Peaks will be considered a match when <= tolerance apart. Default is 0.1.
        """
        self.tolerance = tolerance

    def pair(self, reference: SpectrumType,
             query: SpectrumType) -> Tuple[float, int]:
        """Calculate fidelity between two spectra.

        Parameters
        ----------
        reference
            Single reference spectrum.
        query
            Single query spectrum.

        Returns
        -------
        Score
            Tuple with fidelity score and number of matched peaks.
        """

        # Get m/z and intensity data as arrays
        ref_peaks = get_peaks_array(reference)
        query_peaks = get_peaks_array(query)

        # Find matching pairs, since we only compute the matching score using those
        matching_pairs = find_matches(ref_peaks,
                                      query_peaks,
                                      tolerance=self.tolerance,
                                      shift=0.0)
        if matching_pairs is None:
            return numpy.asarray((float(0), 0), dtype=self.score_datatype)

        p, q = _compute_p_q_for_matches(matching_pairs,
                                        reference.peaks.intensities,
                                        query.peaks.intensities)

        score = fidelity_score(p, q)
        matches = len(matching_pairs)

        return numpy.asarray((score, matches), dtype=self.score_datatype)


def _compute_p_q_for_matches(
        matching_pairs: List[Tuple[int, int]],
        intensities1: numpy.ndarray, intensities2: numpy.ndarray
) -> (numpy.ndarray, numpy.ndarray):
    norm_intensities1 = _to_pdf(intensities1)
    norm_intensities2 = _to_pdf(intensities2)
    p = norm_intensities1.take([match[0] for match in matching_pairs])
    q = norm_intensities2.take([match[1] for match in matching_pairs])
    return (p, q)


def _to_pdf(x: numpy.array) -> numpy.array:
    """ Convert vector to PDF.
    Convert a vector to a probability density function (pdf), so that all values sum up to 1.

    Parameters
    ----------
    x : numpy.array
        The values which should be rescaled in order to sum up to 1.

    Returns
    -------
    output: numpy.array
        The rescaled intensities, so that sum(x) = 1.
    """

    return x / x.sum()
