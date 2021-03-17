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

    The fidelity score computes the similarity of two discrete probability distributions P, Q.
    These are described by probability density functions p(x) and q(x).
    The event x is the occurence of a peak at a given m/z value,
    while p(x) and q(x) denote the likelyhood of this event,
    given by the intensity of the peak.

    For example

    .. testcode::

        import numpy as np
        from matchms import Spectrum
        from matchms.similarity import Fidelity

        reference = Spectrum(mz=numpy.array([100, 200, 300, 500, 510], dtype="float"),
                          intensities=numpy.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float"))

        query = Spectrum(mz=numpy.array([100, 200, 290, 490, 510], dtype="float"),
                          intensities=numpy.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float"))

        # Use factory to construct a similarity function
        fidelity = Fidelity()

        score = fidelity.pair(reference, query)

        print(f"Fidelity score is {score['score']:.2f} with {score['matches']} matched peaks")

    Should output

    .. testoutput::
        Fidelity score is 0.122 with 3 matched peaks.
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
        """Calculate fidelity score and number of matched peaks between two spectra.

        This function computes the matching peaks between the two spectra and
        computes the fidelity score after transforming the intensities to pdfs.

        Parameters
        ----------
        reference
            Single reference spectrum.
        query
            Single query spectrum.

        Returns
        -------
        Score : Tuple[float, int]
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

        # If no matching pairs are found, shortcut and return 0.
        if matching_pairs is None:
            return numpy.asarray((float(0), 0), dtype=self.score_datatype)

        # Compute random variable distributions p and q for reference and query intensities.
        p, q = _compute_p_q_for_matches(matching_pairs,
                                        reference.peaks.intensities,
                                        query.peaks.intensities)

        # Compute score and number of matches
        score = fidelity_score(p, q)
        matches = len(matching_pairs)

        return numpy.asarray((score, matches), dtype=self.score_datatype)


def _compute_p_q_for_matches(
        matching_pairs: List[Tuple[int, int]],
        ref_ints: numpy.array, query_ints: numpy.array
) -> (numpy.array, numpy.array):
    """ Compute PDFs for spectra for given matching peaks.

    The intensity values of reference and query spectrum are rescaled to sum up to 1 to be used as
    pdfs.

    Parameters
    ----------
    matching_pairs : List[Tuple[int, int]]
        List of matching peak indices.
    ref_ints : numpy.array
        Intensities of peaks in reference spectrum.
    query_ints : numpy.array
        Intensities of peaks in query spectrum.

    Returns
    -------
    p, q : (numpy.array, numpy.array)
        Probability density functions of query and reference spectra intensities.
    """

    # Transform reference and query intensities to probability density functions
    ref_pdf = _to_pdf(ref_ints)
    query_pdf = _to_pdf(query_ints)

    # Extract the probabilitites for matching peaks
    p = ref_pdf.take([match[0] for match in matching_pairs])
    q = query_pdf.take([match[1] for match in matching_pairs])
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
