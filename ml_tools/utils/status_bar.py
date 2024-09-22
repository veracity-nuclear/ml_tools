class StatusBar(object):
    def __init__(self, total_count: int, length: int = 50) -> None:
        self._total = total_count
        self._length = length

    def update(self, i: int) -> None:
        frac = (i + 1) / self._total
        completed = int(frac * self._length)
        remaining = self._length - completed
        print('[' + '|' * completed + '.' * remaining + '] {0:5.1f}%'.format(frac*100.), end='\r')
        #print(frac)

    def finalize(self) -> None:
        print('[' + '|' * self._length + '] 100.0%')