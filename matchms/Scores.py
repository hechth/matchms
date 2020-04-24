from numpy import empty, unravel_index, asarray


class Scores:
    def __init__(self, references=None, queries=None, similarity_function=None):
        self.n_rows = asarray(references).flatten().size
        self.n_cols = asarray(queries).flatten().size
        self.references = asarray(references).flatten().reshape(self.n_rows, 1)
        self.queries = asarray(queries).flatten().reshape(1, self.n_cols)
        self.similarity_function = similarity_function
        self._scores = empty([self.n_rows, self.n_cols], dtype="object")
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < self.scores.size:
            # pylint: disable=unbalanced-tuple-unpacking
            r, c = unravel_index(self._index, self._scores.shape)
            self._index += 1
            result = self._scores[r, c]
            if not isinstance(result, tuple):
                result = (result,)
            return (self.references[r, 0], self.queries[0, c]) + result
        self._index = 0
        raise StopIteration

    def __str__(self):
        return self._scores.__str__()

    def calculate(self):
        for i_ref, reference in enumerate(self.references[:self.n_rows, 0]):
            for i_query, query in enumerate(self.queries[0, :self.n_cols]):
                self._scores[i_ref][i_query] = self.similarity_function(reference, query)
        return self

    def reset_iterator(self):
        self._index = 0

    @property
    def scores(self):
        """getter method for scores private variable"""
        return self._scores.copy()

    @scores.setter
    def scores(self, value):
        """setter method for scores private variable"""
        self._scores = value
