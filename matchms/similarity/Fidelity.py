from typing import Tuple
import numpy
from matchms.similarity.spectrum_similarity_functions import collect_peak_pairs
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
    score_datatype = [("score", numpy.float64), ("matches", "int")]

    def __init__(self, tolerance: float = 0.1):
        """
        Parameters
        ----------
        tolerance:
            Peaks will be considered a match when <= tolerance apart. Default is 0.1.
        """
        self.tolerance = tolerance

    def pair(self, reference: SpectrumType, query: SpectrumType) -> Tuple[float, int]:
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
        def get_matching_pairs():
            """Get pairs of peaks that match within the given tolerance."""
            matching_pairs = collect_peak_pairs(spec1, spec2, self.tolerance,
                                                shift=0.0)
            if matching_pairs is None:
                return None
            matching_pairs = matching_pairs[numpy.argsort(matching_pairs[:, 2])[::-1], :]
            return matching_pairs

        spec1 = get_peaks_array(reference)
        spec2 = get_peaks_array(query)
        matching_pairs = get_matching_pairs()
        if matching_pairs is None:
            return numpy.asarray((float(0), 0), dtype=self.score_datatype)

        def fidelity_score(matching_pairs: numpy.ndarray, spec1: numpy.ndarray, spec2: numpy.ndarray):
            #TODO: Implementation
            return 0

        score = fidelity_score(matching_pairs, spec1, spec2)
        return numpy.asarray(score, dtype=self.score_datatype)
