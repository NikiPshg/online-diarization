import torch

from .base import RecognizerMemory


def name_generator(dtype=str):
    i = 0
    while True:
        i += 1
        yield dtype(i)


class OnlineClusteringMemory(RecognizerMemory):
    """
    Performs online clustering given a sequence of feature vectors.
    Clusters are represented by sets of feature vectors.

    Args:
        similarity_score (callable): a function computing the similarity between
        a pair of feature vectors.
        threshold (float): similarity threshold to detect a known class.
        threshold_update (float): similarity threshold to update a class.
        average_scores (bool): if True, average similarity scores across all objects
        representing a class; otherwise, it is assumed that ``similarity_score`` can
        handle this within itself.
    """

    def __init__(
        self,
        similarity_score,
        threshold,
        threshold_update=None,
        average_scores=True,
        **params,
    ):
        super().__init__(similarity_score, average_scores)
        self.threshold = threshold
        self.threshold_update = (
            threshold if threshold_update is None else threshold_update
        )
        self.name_generator = name_generator(int)
        self.params = params
        self.predictions = []

    def reset(self):
        self.name_generator = name_generator(int)
        self.representations.clear()
        self.predictions = []

    def add_class(self, X):
        class_id = next(self.name_generator)
        self.representations[class_id] = X
        return class_id

    def update_class(self, class_id, x):
        X_class = self.representations[class_id]
        self.representations[class_id] = torch.cat([X_class, x])

    def verify_all(self, x):
        classes = list(self.representations.keys())
        scores = []
        for c in classes:
            scores += [self.verify(c, x)]
        scores = torch.cat(scores)
        assert len(classes) == scores.numel()
        return scores

    def processing_one_emb(
        self, emb: torch.Tensor, is_short: bool = False, is_silence: bool = False
    ) -> str:
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)

        if not self.predictions:
            class_id = self.add_class(emb)
            self.predictions = [class_id]
            self.last_speaker = class_id
            return class_id

        if is_short and not is_silence:
            return self.last_speaker

        scores = self.verify_all(emb).view(-1)
        idx_max = torch.argmax(scores).item()
        s_max = scores[idx_max]
        classes = list(self.representations.keys())
        class_id = classes[idx_max]

        if s_max > self.threshold:
            if s_max > self.threshold_update:
                self.update_class(class_id, emb)
            self.last_speaker = class_id
            self.predictions += [class_id]
        else:
            class_id = self.add_class(emb)
            self.last_speaker = class_id
            self.predictions += [class_id]

        return class_id
